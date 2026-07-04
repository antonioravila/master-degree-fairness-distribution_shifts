import subprocess
import concurrent.futures
import argparse
import sys
from datetime import datetime
from pathlib import Path

MODELS = ['logistic_regression', 'xgboost']
TECHNIQUES_WITH_WINDOW = ['sma']
TECHNIQUES_NO_WINDOW = ['ema', 'oracle']
WINDOWS = [3, 5, 10]
BATCH_TYPES = ['monthly', 'weekly']

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


def _fmt_alpha(alpha):
    """Normalize alpha to int when whole number: 1.0 → 1, 0.1 → 0.1."""
    f = float(alpha)
    return int(f) if f == int(f) else f


def _run_folder_name(model, alpha, technique, window, batch_type):
    ema_suffix = f'_ema_{DEFAULT_ALPHA_EMA}' if technique == 'ema' else ''
    return f'{model}_alpha_{_fmt_alpha(alpha)}_{technique}_window_{window}_{batch_type}{ema_suffix}'


def _folder_exists(results_dir, model, alpha, technique, window, batch_type):
    """Check both alpha_1 and alpha_1.0 formats for backward compat with existing results."""
    primary = results_dir / _run_folder_name(model, alpha, technique, window, batch_type)
    alt = results_dir / _run_folder_name(model, float(alpha), technique, window, batch_type)
    return primary.exists() or alt.exists()


def _missing_alphas(entry):
    results_dir = RESULTS_BASE / SENSITIVE_ATTRIBUTE
    return [
        alpha for alpha in ALPHAS
        if not _folder_exists(
            results_dir, entry['model'], alpha, entry['technique'], entry['window'], entry['batch_type']
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
                    'label': f'{entry["label"]} | α={alpha}',
                    'cmd': entry['cmd'] + ['--alpha', str(alpha)],
                })

    return jobs, n_complete


def build_commands():
    commands = []
    for model in MODELS:
        for technique in TECHNIQUES_WITH_WINDOW:
            for window in WINDOWS:
                for batch_type in BATCH_TYPES:
                    commands.append({
                        'model': model,
                        'technique': technique,
                        'window': window,
                        'batch_type': batch_type,
                        'label': f'{model} | {technique} | window={window} | {batch_type}',
                        'cmd': [
                            'uv', 'run', 'src/main.py',
                            '--model_name', model,
                            '--forecasting_technique', technique,
                            '--forecast_window', str(window),
                            '--batch_type', batch_type,
                        ],
                    })
        for technique in TECHNIQUES_NO_WINDOW:
            for batch_type in BATCH_TYPES:
                commands.append({
                    'model': model,
                    'technique': technique,
                    'window': DEFAULT_FORECAST_WINDOW,
                    'batch_type': batch_type,
                    'label': f'{model} | {technique} | {batch_type}',
                    'cmd': [
                        'uv', 'run', 'src/main.py',
                        '--model_name', model,
                        '--forecasting_technique', technique,
                        '--batch_type', batch_type,
                    ],
                })
    return commands


def run_command(entry, log_dir: Path):
    label = entry['label']
    log_path = log_dir / (label.replace(' | ', '_').replace('=', '').replace('.', '') + '.log')

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
    args = parser.parse_args()

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    scenarios = build_commands()
    jobs, n_complete = resolve_jobs(scenarios)

    if n_complete:
        print(f'Skipping {n_complete} already-complete scenario(s).')
    if not jobs:
        print('All scenarios complete. Nothing to run.')
        return

    xgb_jobs = [j for j in jobs if j['model'] == 'xgboost']
    lr_jobs = [j for j in jobs if j['model'] == 'logistic_regression']

    print(
        f'Launching {len(lr_jobs)} LR jobs (max_workers={args.max_lr_workers}) '
        f'and {len(xgb_jobs)} XGBoost jobs (max_workers={args.max_xgb_workers}) '
        f'in parallel pools. Logs → {log_dir}/\n'
    )

    futures = []
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_lr_workers) as lr_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_xgb_workers) as xgb_pool,
    ):
        for job in lr_jobs:
            futures.append(lr_pool.submit(run_command, job, log_dir))
        for job in xgb_jobs:
            futures.append(xgb_pool.submit(run_command, job, log_dir))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    failed = sum(1 for r in results if r != 0)
    total = len(results)
    print(f'\nFinished. {total - failed}/{total} succeeded.')
    if failed:
        print(f'{failed} job(s) failed — check logs in {log_dir}/')
        sys.exit(1)


if __name__ == '__main__':
    main()
