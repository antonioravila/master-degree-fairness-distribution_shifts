"""
Migrates this branch's existing results (all run before `--train_window`
existed, all kamiran-only) to the schema introduced by that hyperparameter.

Two independent, idempotent changes, applied per sensitive-attribute folder
under src/results/ (e.g. src/results/IDADE_BIN/):

1. `baseline_non_abc_{model}_{batch_type}_{fc}` folders (vanilla+static+dynamic
   combined — the shape migrate_legacy_schema.py already normalized): split by
   the `baseline` column.
   - vanilla/static rows stay in the same folder, same name — they never
     depend on train_window, so nothing about their storage changes.
   - dynamic rows move to a new
     `baseline_dynamic_{model}_{batch_type}_{fc}_trainwindow_none` folder,
     with their own 3 CSVs. 'none' because these rows were produced with
     unbounded history — exactly what `train_window=None` means today.

2. Every other run folder (`{model}_alpha_..._{technique}_window_..._{batch}...`,
   abcinml-only): renamed to append `_trainwindow_none`.

Both changes also backfill a `train_window` column into every CSV touched:
'none' for dynamic/abcinml rows (meaningful), and 'none' for the vanilla/
static rows left behind too (not meaningful, but keeps the column free of
NaN for any downstream tool that reads it unconditionally).

Idempotent: a `baseline_non_abc_*` folder whose `baseline` column no longer
contains 'dynamic' is treated as already split (only its CSVs are checked/
patched for the train_window column). Any other folder whose name already
contains '_trainwindow_' is left alone.

Does NOT touch src/results/all/ or src/results/forecast_quality/. Regenerate
the aggregate afterwards with `uv run python src/tools/concatenate_results.py`.

Usage:
    uv run python src/tools/migrate_train_window_schema.py --dry-run
    uv run python src/tools/migrate_train_window_schema.py
"""

import argparse
import re
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / 'results'
SKIP_DIRS = {'all', 'forecast_quality'}

CSV_ASSESSMENT = 'assessment_all_baselines.csv'
CSV_TEMPORAL = 'temporal_results_df.csv'
CSV_ABSOLUTE = 'absolute_results_df.csv'
CSV_FILES = [CSV_ASSESSMENT, CSV_TEMPORAL, CSV_ABSOLUTE]

# Matches a scenario folder that does NOT yet have a _trainwindow_ suffix —
# same shape as concatenate_results.py's _ABC_RE, minus that optional group.
_PRE_TRAINWINDOW_ABC_RE = re.compile(
    r'^(?P<model>.+)_alpha_(?P<alpha>[\d.]+)_(?P<technique>.+)_window_(?P<window>\d+)'
    r'_(?P<batch>monthly|weekly)(?:_ema_(?P<alpha_ema>[\d.]+))?'
    r'(?:_(?P<fairness_correction>kamiran_rw|adaptative_rw))?$'
)
_NON_ABC_RE = re.compile(
    r'^baseline_non_abc_(?P<model>.+)_(?P<batch>monthly|weekly)(?:_(?P<fairness_correction>kamiran_rw|adaptative_rw))?$'
)


def _backfill_train_window(df: pd.DataFrame, value: str = 'none') -> bool:
    if 'train_window' in df.columns:
        return False
    df['train_window'] = value
    return True


def _split_non_abc_folder(folder: Path, dry_run: bool) -> str:
    """Returns a short status string for logging."""
    assessment_path = folder / CSV_ASSESSMENT
    if not assessment_path.exists():
        return 'MISSING assessment CSV, skipped'

    assessment_df = pd.read_csv(assessment_path)
    if 'baseline' not in assessment_df.columns:
        return "no 'baseline' column, skipped"

    if 'dynamic' not in assessment_df['baseline'].unique():
        # Already split (or never had dynamic here) — just backfill the column.
        changed = False
        for csv_name in CSV_FILES:
            csv_path = folder / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if _backfill_train_window(df) and not dry_run:
                df.to_csv(csv_path, index=False)
                changed = True
        return 'already split — backfilled train_window' if changed else 'already split, nothing to do'

    m = _NON_ABC_RE.match(folder.name)
    if not m:
        return 'unrecognized baseline_non_abc_ name, skipped'

    dynamic_folder = folder.with_name(
        f"baseline_dynamic_{m.group('model')}_{m.group('batch')}"
        + (f"_{m.group('fairness_correction')}" if m.group('fairness_correction') else '')
        + '_trainwindow_none'
    )
    if dynamic_folder.exists():
        return f'SKIP — target {dynamic_folder.name} already exists'

    dfs = {}
    for csv_name in CSV_FILES:
        csv_path = folder / csv_name
        if csv_path.exists():
            dfs[csv_name] = pd.read_csv(csv_path)

    if not dry_run:
        dynamic_folder.mkdir(parents=True, exist_ok=True)

    for csv_name, df in dfs.items():
        dynamic_rows = df[df['baseline'] == 'dynamic'].copy()
        remaining_rows = df[df['baseline'] != 'dynamic'].copy()
        _backfill_train_window(dynamic_rows)
        _backfill_train_window(remaining_rows)
        if not dry_run:
            dynamic_rows.to_csv(dynamic_folder / csv_name, index=False)
            remaining_rows.to_csv(folder / csv_name, index=False)

    assessment_df = dfs.get(CSV_ASSESSMENT)
    n_dynamic = int((assessment_df['baseline'] == 'dynamic').sum()) if assessment_df is not None else 0
    return f'split -> {dynamic_folder.name} ({n_dynamic} assessment rows moved)'


def _rename_abc_folder(folder: Path, dry_run: bool) -> str:
    if '_trainwindow_' in folder.name:
        # Already migrated — just backfill the column in case it predates it.
        changed = False
        for csv_name in CSV_FILES:
            csv_path = folder / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if _backfill_train_window(df) and not dry_run:
                df.to_csv(csv_path, index=False)
                changed = True
        return 'already migrated — backfilled train_window' if changed else 'already migrated, nothing to do'

    if not _PRE_TRAINWINDOW_ABC_RE.match(folder.name):
        return 'unrecognized name, skipped'

    target = folder.with_name(folder.name + '_trainwindow_none')
    if target.exists():
        return f'SKIP — target {target.name} already exists'

    for csv_name in CSV_FILES:
        csv_path = folder / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        _backfill_train_window(df)
        if not dry_run:
            df.to_csv(csv_path, index=False)

    if not dry_run:
        folder.rename(target)
    return f'renamed -> {target.name}'


def migrate_attribute_dir(sensitive_attribute_dir: Path, dry_run: bool) -> None:
    print(f'\n{sensitive_attribute_dir.name}:')
    for folder in sorted(sensitive_attribute_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith('baseline_dynamic_'):
            continue  # already-migrated target, never a source
        if folder.name.startswith('baseline_non_abc_'):
            status = _split_non_abc_folder(folder, dry_run)
        else:
            status = _rename_abc_folder(folder, dry_run)
        print(f'  {folder.name}: {status}')


def main():
    parser = argparse.ArgumentParser(
        description='Migrate existing results to the train_window schema.',
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
