"""
Translates the epidemiological period labels stored in every mae_by_period.csv
under src/results/forecast_quality/ from Portuguese to English, matching the
labels metrics_utils._PERIOD_SPANS now uses:

    pré-pandemia -> pre-pandemic        pandemia aguda 1 -> acute pandemic 1
    transição    -> transition          pandemia aguda 2 -> acute pandemic 2
    endêmico     -> endemic

mae_by_period.csv (written by analyze_forecast_quality.py) is the only result
file that stores these labels; forecast_quality.csv and mae_overall.csv have no
period column, and the pipeline CSVs never stored them.

Idempotent: English labels are left alone. Only the `period` column is
touched; every other column is written back byte-identical.

Usage:
    uv run python src/tools/migrate_period_labels_en.py --dry-run
    uv run python src/tools/migrate_period_labels_en.py
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / 'results'
CSV_NAME = 'mae_by_period.csv'
LABELS_EN = {
    'pré-pandemia': 'pre-pandemic',
    'pandemia aguda 1': 'acute pandemic 1',
    'pandemia aguda 2': 'acute pandemic 2',
    'transição': 'transition',
    'endêmico': 'endemic',
}


def migrate(csv_path: Path, dry_run: bool) -> str:
    # Read as text so the float columns are written back unchanged
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if 'period' not in df.columns:
        print(f'  SKIP {csv_path}: no period column')
        return 'skipped'

    n = int(df['period'].isin(LABELS_EN).sum())
    if n == 0:
        return 'unchanged'
    if dry_run:
        print(f'  would translate {n} label(s): {csv_path}')
        return 'migrated'

    df['period'] = df['period'].replace(LABELS_EN)
    df.to_csv(csv_path, index=False)
    return 'migrated'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='List the files that would change; write nothing.')
    parser.add_argument('--results-root', default=str(RESULTS_ROOT), help=f'Default: {RESULTS_ROOT}')
    args = parser.parse_args()

    counts = {'migrated': 0, 'unchanged': 0, 'skipped': 0}
    for csv_path in sorted(Path(args.results_root).rglob(CSV_NAME)):
        counts[migrate(csv_path, args.dry_run)] += 1

    prefix = '[dry-run] ' if args.dry_run else ''
    print(
        f'\n{prefix}{sum(counts.values())} file(s): {counts["migrated"]} migrated, '
        f'{counts["unchanged"]} already in English, {counts["skipped"]} skipped (no period column)'
    )


if __name__ == '__main__':
    main()
