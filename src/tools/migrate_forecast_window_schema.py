"""
Appends the `_fw_{forecast_window}` suffix to existing vanilla/static and
dynamic result folders.

Why: `forecast_window` sets the warm-up skip in main.run_baseline, which decides
which batches every baseline is evaluated on — and, for vanilla/static, which
single batch they train on. But `baseline_non_abc_*` / `baseline_dynamic_*`
folder names never recorded it, while main.py uses those names as a cache key
(`if os.path.exists(...): load`). So the first run to write a folder froze its
forecast_window there, and any later run with a different one silently reused
it. Two folders under IDADE_BIN were already stale this way (both
logistic_regression/weekly, holding forecast_window=5 data while every sibling
holds 3).

abcinml folders are untouched: their names already carry `_window_{fw}_`, and
that branch has no cache lookup at all.

The rename is deterministic because the true forecast_window is already stored
in every CSV (main._attach_metadata stamps it at write time) — this script reads
it back rather than guessing.

Idempotent: a folder whose name already contains `_fw_` is left alone. A folder
whose CSV holds more than one distinct forecast_window is skipped and reported,
never guessed at.

Usage:
    uv run python src/tools/migrate_forecast_window_schema.py --dry-run
    uv run python src/tools/migrate_forecast_window_schema.py
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / 'results'
SKIP_DIRS = {'all', 'forecast_quality'}

CSV_ASSESSMENT = 'assessment_all_baselines.csv'
MIGRATED_PREFIXES = ('baseline_non_abc_', 'baseline_dynamic_')


def _migrate_folder(folder: Path, dry_run: bool) -> str:
    if '_fw_' in folder.name:
        return 'already migrated, nothing to do'

    assessment_path = folder / CSV_ASSESSMENT
    if not assessment_path.exists():
        return 'MISSING assessment CSV, skipped'

    df = pd.read_csv(assessment_path, usecols=['forecast_window'])
    values = sorted(df['forecast_window'].dropna().unique())
    if len(values) != 1:
        return f'AMBIGUOUS forecast_window {values}, skipped (fix by hand)'

    forecast_window = int(values[0])
    target = folder.with_name(f'{folder.name}_fw_{forecast_window}')
    if target.exists():
        return f'SKIP — target {target.name} already exists'

    if not dry_run:
        folder.rename(target)
    return f'renamed -> {target.name}'


def migrate_attribute_dir(sensitive_attribute_dir: Path, dry_run: bool) -> None:
    print(f'\n{sensitive_attribute_dir.name}:')
    touched = False
    for folder in sorted(sensitive_attribute_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(MIGRATED_PREFIXES):
            continue
        touched = True
        print(f'  {folder.name}: {_migrate_folder(folder, dry_run)}')
    if not touched:
        print('  (no vanilla/static or dynamic folders found)')


def main():
    parser = argparse.ArgumentParser(
        description='Append _fw_{forecast_window} to vanilla/static and dynamic result folders.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dry-run', action='store_true', help='Report what would change; write nothing.')
    parser.add_argument(
        '--results-root', default=str(RESULTS_ROOT),
        help='Override the src/results root (mainly for testing).',
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f'{root} does not exist')

    attr_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name not in SKIP_DIRS]
    if not attr_dirs:
        raise SystemExit(f'No sensitive-attribute folders found under {root}')

    if args.dry_run:
        print('--- DRY RUN: no files will be changed ---')

    for attr_dir in attr_dirs:
        migrate_attribute_dir(attr_dir, args.dry_run)

    print(
        '\nDone.'
        + (' (dry run — nothing written)' if args.dry_run else '')
        + '\nNext: regenerate the aggregate with `uv run python src/tools/concatenate_results.py`.'
    )


if __name__ == '__main__':
    main()
