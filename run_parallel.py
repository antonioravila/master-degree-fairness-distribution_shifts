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

# Conservative defaults — safe for RTX 3050 (4 GB VRAM).
# For RTX Blackwell 8 GB: --max_xgb_workers 2 --max_lr_workers 4
DEFAULT_XGB_WORKERS = 1
DEFAULT_LR_WORKERS = 2


def build_commands():
    commands = []
    for model in MODELS:
        for technique in TECHNIQUES_WITH_WINDOW:
            for window in WINDOWS:
                for batch_type in BATCH_TYPES:
                    commands.append({
                        'model': model,
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
    log_path = log_dir / (label.replace(' | ', '_').replace('=', '') + '.log')

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

    all_commands = build_commands()
    xgb_commands = [c for c in all_commands if c['model'] == 'xgboost']
    lr_commands = [c for c in all_commands if c['model'] == 'logistic_regression']

    print(
        f'Launching {len(lr_commands)} LR jobs (max_workers={args.max_lr_workers}) '
        f'and {len(xgb_commands)} XGBoost jobs (max_workers={args.max_xgb_workers}) '
        f'in parallel pools. Logs → {log_dir}/\n'
    )

    futures = []
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_lr_workers) as lr_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_xgb_workers) as xgb_pool,
    ):
        for cmd in lr_commands:
            futures.append(lr_pool.submit(run_command, cmd, log_dir))
        for cmd in xgb_commands:
            futures.append(xgb_pool.submit(run_command, cmd, log_dir))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    failed = sum(1 for r in results if r != 0)
    total = len(results)
    print(f'\nFinished. {total - failed}/{total} succeeded.')
    if failed:
        print(f'{failed} job(s) failed — check logs in {log_dir}/')
        sys.exit(1)


if __name__ == '__main__':
    main()
