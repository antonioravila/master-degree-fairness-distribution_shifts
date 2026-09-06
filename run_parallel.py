import subprocess
import concurrent.futures
import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

MODELS = ['logistic_regression', 'xgboost']
TECHNIQUES_WITH_WINDOW = ['sma']
TECHNIQUES_NO_WINDOW = ['ema', 'oracle']
WINDOWS = [3, 5, 10]
BATCH_TYPES = ['monthly', 'weekly']

# Fairness-correction axis. 'kamiran_rw' is Kamiran & Calders reweighing;
# 'adaptative_rw' is APR (Hu et al.). Each writes to its own folder (…_kamiran_rw /
# …_adaptative_rw). APR uses main.py's default fairness measure (demographic
# parity) — the measure is recorded in the CSVs, not the folder name.
FAIRNESS_CORRECTIONS = ['kamiran_rw', 'adaptative_rw']

# Must match general_utils.py defaults
ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SENSITIVE_ATTRIBUTE = 'IDADE_BIN'
DEFAULT_FORECAST_WINDOW = 5  # used by ema/oracle (no --forecast_window passed)
DEFAULT_ALPHA_EMA = 0.5

RESULTS_BASE = Path('src/results')

# Conservative defaults — safe for RTX 3050 (4 GB VRAM).
# For RTX Blackwell 8 GB: --max_xgb_workers 2 --max_lr_workers 4
DEFAULT_XGB_WORKERS = 1
DEFAULT_LR_WORKERS = 2
# APR retrains T times per fit (default T=20), so an APR job is ~20x a kamiran one.
# This caps APR jobs running at once *across both model pools* (a shared gate on
# top of the LR/XGB pools), independent of --max_lr_workers / --max_xgb_workers.
DEFAULT_APR_WORKERS = 1


def _fmt_alpha(alpha):
    """Normalize alpha to int when whole number: 1.0 → 1, 0.1 → 0.1."""
    f = float(alpha)
    return int(f) if f == int(f) else f


def _fairness_correction_suffix(fairness_correction):
    """Mirror of metrics_utils._fairness_correction_suffix — keep in sync."""
    return f'_{fairness_correction}'


def _run_folder_name(model, alpha, technique, window, batch_type, fairness_correction='kamiran_rw'):
    """Mirror of metrics_utils._run_folder_name — keep in sync."""
    ema_suffix = f'_ema_{DEFAULT_ALPHA_EMA}' if technique == 'ema' else ''
    fc_suffix = _fairness_correction_suffix(fairness_correction)
    return f'{model}_alpha_{_fmt_alpha(alpha)}_{technique}_window_{window}_{batch_type}{ema_suffix}{fc_suffix}'


def _folder_exists(results_dir, model, alpha, technique, window, batch_type, fairness_correction):
    """Check both alpha_1 and alpha_1.0 formats for backward compat with existing results."""
    primary = results_dir / _run_folder_name(model, alpha, technique, window, batch_type, fairness_correction)
    alt = results_dir / _run_folder_name(model, float(alpha), technique, window, batch_type, fairness_correction)
    return primary.exists() or alt.exists()


def _missing_alphas(entry):
    results_dir = RESULTS_BASE / SENSITIVE_ATTRIBUTE
    return [
        alpha for alpha in ALPHAS
        if not _folder_exists(
            results_dir, entry['model'], alpha, entry['technique'], entry['window'],
            entry['batch_type'], entry['fairness_correction'],
        )
    ]


def resolve_jobs(scenarios):
    """Expand scenarios into runnable jobs.

    - Complete (0 missing alphas): skipped.
    - Not started (all 11 missing): full command submitted — one data load covers all alphas.
    - Incomplete (1–10 missing): one job per missing alpha to avoid redoing finished work.
    """
    jobs = []
    n_complete = 0

    for entry in scenarios:
        missing = _missing_alphas(entry)

        if not missing:
            n_complete += 1
            continue

        if len(missing) == len(ALPHAS):
            jobs.append(entry)
        else:
            for alpha in missing:
                jobs.append({
                    **entry,
                    'label': f'{entry["label"]} | alpha={alpha}',
                    'cmd': entry['cmd'] + ['--alpha', str(alpha)],
                })

    return jobs, n_complete


def _scenario_cmd(model, technique, batch_type, window, with_window):
    cmd = [
        'uv', 'run', 'src/main.py',
        '--model_name', model,
        '--forecasting_technique', technique,
        '--batch_type', batch_type,
    ]
    if with_window:
        cmd += ['--forecast_window', str(window)]
    return cmd


def build_commands(fairness_corrections):
    commands = []

    def _add(model, technique, window, batch_type, with_window):
        base_label = (
            f'{model} | {technique} | window={window} | {batch_type}'
            if with_window else f'{model} | {technique} | {batch_type}'
        )
        base_cmd = _scenario_cmd(model, technique, batch_type, window, with_window)
        for fc in fairness_corrections:
            commands.append({
                'model': model,
                'technique': technique,
                'window': window,
                'batch_type': batch_type,
                'fairness_correction': fc,
                'label': f'{base_label} | fc={fc}',
                'cmd': base_cmd + ['--fairness_correction', fc],
            })

    for model in MODELS:
        for technique in TECHNIQUES_WITH_WINDOW:
            for window in WINDOWS:
                for batch_type in BATCH_TYPES:
                    _add(model, technique, window, batch_type, with_window=True)
        for technique in TECHNIQUES_NO_WINDOW:
            for batch_type in BATCH_TYPES:
                _add(model, technique, DEFAULT_FORECAST_WINDOW, batch_type, with_window=False)
    return commands


def run_command(entry, log_dir: Path, apr_gate: threading.Semaphore):
    label = entry['label']
    log_path = log_dir / (label.replace(' | ', '_').replace('=', '-').replace(' ', '') + '.log')

    is_apr = entry.get('fairness_correction') == 'adaptative_rw'
    if is_apr:
        apr_gate.acquire()
    try:
        print(f'[{_now()}] START  {label}')
        start = datetime.now()

        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                entry['cmd'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = int((datetime.now() - start).total_seconds())
        status = 'DONE ' if result.returncode == 0 else 'FAILED'
        print(f'[{_now()}] {status} {label} ({elapsed}s) — log: {log_path}')
        return result.returncode
    finally:
        if is_apr:
            apr_gate.release()


def _now():
    return datetime.now().strftime('%H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment grid with separate parallelism for XGBoost (GPU) and LR (CPU).'
    )
    parser.add_argument(
        '--max_xgb_workers', type=int, default=DEFAULT_XGB_WORKERS,
        help=f'Max parallel XGBoost jobs (GPU-bound). Default: {DEFAULT_XGB_WORKERS}. Use 2 if VRAM >= 8 GB.',
    )
    parser.add_argument(
        '--max_lr_workers', type=int, default=DEFAULT_LR_WORKERS,
        help=f'Max parallel logistic regression jobs (CPU-bound). Default: {DEFAULT_LR_WORKERS}.',
    )
    parser.add_argument(
        '--max_apr_workers', type=int, default=DEFAULT_APR_WORKERS,
        help=(
            f'Max APR (adaptative_rw) jobs running at once, across both model pools. '
            f'Default: {DEFAULT_APR_WORKERS}. APR retrains ~20x per fit, so keep this low.'
        ),
    )
    parser.add_argument(
        '--fairness_corrections', nargs='+', default=FAIRNESS_CORRECTIONS, choices=FAIRNESS_CORRECTIONS,
        help=f'Which fairness corrections to run. Default: {" ".join(FAIRNESS_CORRECTIONS)}.',
    )
    args = parser.parse_args()

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    scenarios = build_commands(args.fairness_corrections)
    jobs, n_complete = resolve_jobs(scenarios)

    if n_complete:
        print(f'Skipping {n_complete} already-complete scenario(s).')
    if not jobs:
        print('All scenarios complete. Nothing to run.')
        return

    # Submit kamiran jobs before APR jobs so short kamiran runs are not stuck
    # behind APR jobs waiting on the APR gate.
    jobs.sort(key=lambda j: j['fairness_correction'] == 'adaptative_rw')

    apr_gate = threading.Semaphore(max(1, args.max_apr_workers))

    xgb_jobs = [j for j in jobs if j['model'] == 'xgboost']
    lr_jobs = [j for j in jobs if j['model'] == 'logistic_regression']
    n_apr = sum(1 for j in jobs if j['fairness_correction'] == 'adaptative_rw')

    print(
        f'Launching {len(lr_jobs)} LR jobs (max_workers={args.max_lr_workers}) '
        f'and {len(xgb_jobs)} XGBoost jobs (max_workers={args.max_xgb_workers}) '
        f'in parallel pools; {n_apr} of them are APR, gated to {args.max_apr_workers} at a time. '
        f'Logs → {log_dir}/\n'
    )

    futures = []
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_lr_workers) as lr_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_xgb_workers) as xgb_pool,
    ):
        for job in lr_jobs:
            futures.append(lr_pool.submit(run_command, job, log_dir, apr_gate))
        for job in xgb_jobs:
            futures.append(xgb_pool.submit(run_command, job, log_dir, apr_gate))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    failed = sum(1 for r in results if r != 0)
    total = len(results)
    print(f'\nFinished. {total - failed}/{total} succeeded.')
    if failed:
        print(f'{failed} job(s) failed — check logs in {log_dir}/')
        sys.exit(1)


if __name__ == '__main__':
    main()
