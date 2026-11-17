import subprocess
import concurrent.futures
import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

MODELS = ['logistic_regression', 'xgboost', 'random_forest', 'lightgbm']

# Model → worker pool. Anything not listed runs in the CPU pool, so a new model
# lands in a known pool instead of silently never being submitted.
POOL_BY_MODEL = {
    'xgboost': 'gpu',
    'random_forest': 'rf',
    # lightgbm also uses n_jobs=-1: it shares RF's heavy-CPU pool
    # and still runs alongside xgboost, which is on the GPU.
    'lightgbm': 'rf',
}
DEFAULT_POOL = 'cpu'
TECHNIQUES_WITH_WINDOW = ['sma']
TECHNIQUES_NO_WINDOW = ['ema', 'oracle']
WINDOWS = [3, 5, 10]
BATCH_TYPES = ['monthly', 'weekly']

# Fairness-correction axis. 'kamiran_rw' is Kamiran & Calders reweighing;
# 'adaptative_rw' is APR (Hu et al.). Each writes to its own folder (…_kamiran_rw /
# …_adaptative_rw). APR uses main.py's default fairness measure (demographic
# parity) — the measure is recorded in the CSVs, not the folder name.
FAIRNESS_CORRECTIONS = ['kamiran_rw', 'adaptative_rw']

# train_window axis (months bounding dynamic/abcinml's training history).
# dynamic/abcinml are the whole reason train_window exists (they're the slow
# baselines) — so unlike leaving this at [None], the default grid sweeps
# these candidate values for BOTH fairness corrections. This multiplies the
# grid 4x — pass --train_windows to override with a narrower set if you want
# to run just an exploration subset first.
DEFAULT_TRAIN_WINDOWS = [None, 3, 6, 12]

# Must match general_utils.py defaults
ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SENSITIVE_ATTRIBUTE = 'IDADE_BIN'
DEFAULT_FORECAST_WINDOW = 5  # used by ema/oracle (no --forecast_window passed)
DEFAULT_ALPHA_EMA = 0.5

RESULTS_BASE = Path('src/results')

# Conservative defaults — safe for a GPU with 4 GB of VRAM.
# With 8 GB or more: --max_xgb_workers 2 --max_lr_workers 4
DEFAULT_XGB_WORKERS = 1
DEFAULT_LR_WORKERS = 2
# random_forest and lightgbm use n_jobs=-1 (every core per process): more than one
# of their jobs at once oversubscribes the CPU and runs slower than one alone.
DEFAULT_RF_WORKERS = 1
# APR retrains T times per fit (default T=20), so an APR job is ~20x a kamiran one.
# This caps APR jobs running at once *across all model pools* (a shared gate on
# top of the LR/XGB/RF pools), independent of the per-pool worker limits.
DEFAULT_APR_WORKERS = 1


def _fmt_alpha(alpha):
    """Normalize alpha to int when whole number: 1.0 → 1, 0.1 → 0.1."""
    f = float(alpha)
    return int(f) if f == int(f) else f


def _fairness_correction_suffix(fairness_correction):
    """Mirror of metrics_utils._fairness_correction_suffix — keep in sync."""
    return f'_{fairness_correction}'


def _train_window_suffix(train_window):
    """Mirror of metrics_utils._train_window_suffix — keep in sync."""
    token = 'none' if train_window is None else str(train_window)
    return f'_trainwindow_{token}'


def _run_folder_name(model, alpha, technique, window, batch_type, fairness_correction='kamiran_rw', train_window=None):
    """Mirror of metrics_utils._run_folder_name — keep in sync."""
    ema_suffix = f'_ema_{DEFAULT_ALPHA_EMA}' if technique == 'ema' else ''
    fc_suffix = _fairness_correction_suffix(fairness_correction)
    tw_suffix = _train_window_suffix(train_window)
    return f'{model}_alpha_{_fmt_alpha(alpha)}_{technique}_window_{window}_{batch_type}{ema_suffix}{fc_suffix}{tw_suffix}'


def _folder_exists(results_dir, model, alpha, technique, window, batch_type, fairness_correction, train_window):
    """Check both alpha_1 and alpha_1.0 formats for backward compat with existing results."""
    primary = results_dir / _run_folder_name(model, alpha, technique, window, batch_type, fairness_correction, train_window)
    alt = results_dir / _run_folder_name(model, float(alpha), technique, window, batch_type, fairness_correction, train_window)
    return primary.exists() or alt.exists()


def _missing_alphas(entry):
    results_dir = RESULTS_BASE / SENSITIVE_ATTRIBUTE
    return [
        alpha for alpha in ALPHAS
        if not _folder_exists(
            results_dir, entry['model'], alpha, entry['technique'], entry['window'],
            entry['batch_type'], entry['fairness_correction'], entry['train_window'],
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


def build_commands(fairness_corrections, train_windows=None, models=None):
    models = models if models is not None else MODELS
    train_windows = train_windows if train_windows is not None else DEFAULT_TRAIN_WINDOWS
    commands = []

    def _add(model, technique, window, batch_type, with_window):
        base_label = (
            f'{model} | {technique} | window={window} | {batch_type}'
            if with_window else f'{model} | {technique} | {batch_type}'
        )
        base_cmd = _scenario_cmd(model, technique, batch_type, window, with_window)
        for fc in fairness_corrections:
            for tw in train_windows:
                cmd = base_cmd + ['--fairness_correction', fc]
                if tw is not None:
                    cmd = cmd + ['--train_window', str(tw)]
                commands.append({
                    'model': model,
                    'technique': technique,
                    'window': window,
                    'batch_type': batch_type,
                    'fairness_correction': fc,
                    'train_window': tw,
                    'label': f"{base_label} | fc={fc} | trainwindow={'none' if tw is None else tw}",
                    'cmd': cmd,
                })

    for model in models:
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
        description='Run experiment grid with separate parallelism for XGBoost (GPU), LR (CPU) and RF/LightGBM (CPU, all cores).'
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
        '--max_rf_workers', type=int, default=DEFAULT_RF_WORKERS,
        help=f'Max parallel random forest + LightGBM jobs, shared pool (each uses every core). Default: {DEFAULT_RF_WORKERS}.',
    )
    parser.add_argument(
        '--max_apr_workers', type=int, default=DEFAULT_APR_WORKERS,
        help=(
            f'Max APR (adaptative_rw) jobs running at once, across all model pools. '
            f'Default: {DEFAULT_APR_WORKERS}. APR retrains ~20x per fit, so keep this low.'
        ),
    )
    parser.add_argument(
        '--models', nargs='+', default=MODELS, choices=MODELS,
        help=f'Which models to run. Default: {" ".join(MODELS)}.',
    )
    parser.add_argument(
        '--fairness_corrections', nargs='+', default=FAIRNESS_CORRECTIONS, choices=FAIRNESS_CORRECTIONS,
        help=f'Which fairness corrections to run. Default: {" ".join(FAIRNESS_CORRECTIONS)}.',
    )
    parser.add_argument(
        '--train_windows', nargs='+', default=None,
        help=(
            "Which --train_window values (months) to sweep for dynamic/abcinml, in addition to "
            "the existing model/technique/window/batch_type/fairness_correction grid. Use the "
            f"literal 'none' for unbounded history. Default: {['none' if w is None else w for w in DEFAULT_TRAIN_WINDOWS]} "
            "for every fairness_correction requested."
        ),
    )
    parser.add_argument(
        '--techniques', nargs='+', default=['sma', 'ema', 'oracle'], choices=['sma', 'ema', 'oracle'],
        help='Which forecasting techniques to run. Default: sma ema oracle.',
    )
    args = parser.parse_args()

    train_windows = (
        DEFAULT_TRAIN_WINDOWS if args.train_windows is None
        else [None if tw.lower() == 'none' else int(tw) for tw in args.train_windows]
    )

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    scenarios = build_commands(args.fairness_corrections, train_windows, args.models)
    scenarios = [s for s in scenarios if s['technique'] in args.techniques]
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

    pool_jobs = {'cpu': [], 'gpu': [], 'rf': []}
    for j in jobs:
        pool_jobs[POOL_BY_MODEL.get(j['model'], DEFAULT_POOL)].append(j)
    assert sum(len(p) for p in pool_jobs.values()) == len(jobs), 'some jobs were not routed to a pool'
    n_apr = sum(1 for j in jobs if j['fairness_correction'] == 'adaptative_rw')

    print(
        f'Launching {len(pool_jobs["cpu"])} LR jobs (max_workers={args.max_lr_workers}), '
        f'{len(pool_jobs["gpu"])} XGBoost jobs (max_workers={args.max_xgb_workers}) '
        f'and {len(pool_jobs["rf"])} RF/LightGBM jobs (max_workers={args.max_rf_workers}) '
        f'in parallel pools; {n_apr} of them are APR, gated to {args.max_apr_workers} at a time. '
        f'Logs → {log_dir}/\n'
    )

    futures = []
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_lr_workers) as lr_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_xgb_workers) as xgb_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_rf_workers) as rf_pool,
    ):
        pools = {'cpu': lr_pool, 'gpu': xgb_pool, 'rf': rf_pool}
        for name, pool in pools.items():
            for job in pool_jobs[name]:
                futures.append(pool.submit(run_command, job, log_dir, apr_gate))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    failed = sum(1 for r in results if r != 0)
    total = len(results)
    print(f'\nFinished. {total - failed}/{total} succeeded.')
    if failed:
        print(f'{failed} job(s) failed — check logs in {log_dir}/')
        sys.exit(1)


if __name__ == '__main__':
    main()
