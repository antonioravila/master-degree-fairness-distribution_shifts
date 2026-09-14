"""
Migrates pre-APR-refactor results (the shape still found on `main`) to the
current schema, so they can be consumed by run_parallel.py,
concatenate_results.py, metrics_utils.py and the analysis notebooks.

Two independent, idempotent changes, applied per sensitive-attribute folder
under src/results/ (e.g. src/results/IDADE_BIN/):

1. CSV columns, inside each of the three per-run files:
   - assessment_all_baselines.csv: sp -> dp, tpr -> tpr_diff, fpr -> fpr_diff
     (only renamed if the old name is present and the new one isn't yet).
   - all three files: adds fairness_correction='kamiran_rw' and
     apr_fairness_measure='' when missing. main never ran APR, so every
     legacy result is unambiguously kamiran_rw (see metrics_utils.py and
     main.py's _attach_metadata for the convention this mirrors).

2. Folder name: appends the _kamiran_rw fairness-correction suffix that
   metrics_utils._fairness_correction_suffix expects, e.g.
     baseline_non_abc_xgboost_monthly          -> ..._kamiran_rw
     xgboost_alpha_0.5_sma_window_10_monthly   -> ..._kamiran_rw
   Folders already ending in _kamiran_rw or _adaptative_rw are left alone
   (their CSVs are still checked/patched, in case they came from an
   in-between version of the code).

Does NOT touch src/results/all/ (the concatenated aggregate) or
src/results/forecast_quality/. Regenerate the aggregate afterwards with
`uv run python src/tools/concatenate_results.py`.

IMPORTANT — this is a schema migration only. It does not and cannot fix the
fact that main's vanilla/static baselines were computed under an older,
different run_baseline warm-up convention (trained on batch 0, not on the
first post-warm-up batch that the current pipeline uses). Those rows are
migrated as-is; treat them as a distinct historical artifact if you need a
strict apples-to-apples comparison against rows produced by the current code.

Usage:
    uv run python src/tools/migrate_legacy_schema.py --dry-run
    uv run python src/tools/migrate_legacy_schema.py
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

_FC_SUFFIXES = ('_kamiran_rw', '_adaptative_rw')

# Same two shapes as metrics_utils._run_folder_name / baseline_non_abc_folder_name,
# WITHOUT the trailing fairness-correction suffix — used to recognize a legacy
# (pre-suffix) folder name. Kept in sync manually with metrics_utils.py.
_LEGACY_ABC_RE = re.compile(
    r'^(?P<model>.+?)_alpha_(?P<alpha>[\d.]+)_(?P<technique>.+?)'
    r'_window_(?P<window>\d+)_(?P<batch>monthly|weekly)(?:_ema_(?P<alpha_ema>[\d.]+))?$'
)
_LEGACY_BASELINE_RE = re.compile(r'^baseline_non_abc_(?P<model>.+?)_(?P<batch>monthly|weekly)$')


def _is_legacy_scenario_folder(name: str) -> bool:
    if name.endswith(_FC_SUFFIXES):
        return False
    return bool(_LEGACY_ABC_RE.match(name) or _LEGACY_BASELINE_RE.match(name))


def _ensure_metadata_columns(df: pd.DataFrame) -> bool:
    changed = False
    if 'fairness_correction' not in df.columns:
        df['fairness_correction'] = 'kamiran_rw'
        changed = True
    if 'apr_fairness_measure' not in df.columns:
        df['apr_fairness_measure'] = ''
        changed = True
    return changed


def _migrate_assessment_csv(path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(path)
    rename_map = {'sp': 'dp', 'tpr': 'tpr_diff', 'fpr': 'fpr_diff'}
    to_rename = {old: new for old, new in rename_map.items() if old in df.columns and new not in df.columns}
    changed = bool(to_rename)
    if to_rename:
        df = df.rename(columns=to_rename)
    changed |= _ensure_metadata_columns(df)
    if changed and not dry_run:
        df.to_csv(path, index=False)
    return changed


def _migrate_plain_csv(path: Path, dry_run: bool) -> bool:
    df = pd.read_csv(path)
    changed = _ensure_metadata_columns(df)
    if changed and not dry_run:
        df.to_csv(path, index=False)
    return changed


def _migrate_folder_csvs(folder: Path, dry_run: bool) -> list[str]:
    touched = []
    for csv_name in CSV_FILES:
        csv_path = folder / csv_name
        if not csv_path.exists():
            continue
        migrate_fn = _migrate_assessment_csv if csv_name == CSV_ASSESSMENT else _migrate_plain_csv
        if migrate_fn(csv_path, dry_run):
            touched.append(csv_name)
    return touched


def migrate_attribute_dir(sensitive_attribute_dir: Path, dry_run: bool) -> None:
    renamed, csv_only, skipped = [], [], []

    for folder in sorted(sensitive_attribute_dir.iterdir()):
        if not folder.is_dir():
            continue

        name = folder.name
        if name.endswith(_FC_SUFFIXES):
            touched = _migrate_folder_csvs(folder, dry_run)
            if touched:
                csv_only.append((name, touched))
            continue

        if not _is_legacy_scenario_folder(name):
            skipped.append(name)
            continue

        target = folder.with_name(name + '_kamiran_rw')
        if target.exists():
            print(f'  SKIP  {name} -> {target.name} already exists, not touching either')
            skipped.append(name)
            continue

        touched = _migrate_folder_csvs(folder, dry_run)
        if not dry_run:
            folder.rename(target)
        renamed.append((name, target.name, touched))

    verb = 'would be' if dry_run else 'were'
    print(f'\n{sensitive_attribute_dir.name}:')
    print(f'  {len(renamed)} folder(s) {verb} renamed + migrated')
    print(f'  {len(csv_only)} already-suffixed folder(s) had their CSVs patched')
    if skipped:
        print(f'  {len(skipped)} folder(s) skipped (unrecognized name or collision): {skipped}')


def main():
    parser = argparse.ArgumentParser(
        description='Migrate pre-APR-refactor results (main-branch shape) to the current schema.',
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
