"""
Adds the eq_odds_diff column (Equalized Odds difference) to every
assessment_all_baselines.csv under src/results/, computed from the columns
already saved — no retraining:

    eq_odds_diff = max(tpr_diff, fpr_diff)

This is the same value metrics_utils.compute_snapshot_metrics now writes for
new runs (NaN in one of the two is ignored, as in pandas max(axis=1)).

Idempotent: if eq_odds_diff already exists it is recomputed and overwritten,
so the script can be re-run whenever results produced by older code arrive.
The column is placed right after fpr_diff, matching the order of new runs.

Only assessment_all_baselines.csv is touched (temporal_results_df.csv and
absolute_results_df.csv have no tpr_diff/fpr_diff). src/results/all/ is
migrated too; rerunning concatenate_results.py afterwards gives the same result.

Usage:
    uv run python src/tools/migrate_add_eq_odds_diff.py --dry-run
    uv run python src/tools/migrate_add_eq_odds_diff.py
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / 'results'
CSV_ASSESSMENT = 'assessment_all_baselines.csv'
COLUMN = 'eq_odds_diff'


def migrate(csv_path: Path, dry_run: bool) -> str:
    # Read as text so every other column is written back byte-identical (a numeric round-trip
    # would drop float digits and turn a mixed int/float alpha column into 0.0, 1.0, ...)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in ('tpr_diff', 'fpr_diff') if c not in df.columns]
    if missing:
        print(f'  SKIP {csv_path}: missing {", ".join(missing)}')
        return 'skipped'

    status = 'recomputed' if COLUMN in df.columns else 'migrated'
    action = 'would recompute' if status == 'recomputed' else 'would add'
    if dry_run:
        print(f'  {action} {COLUMN} ({len(df)} rows): {csv_path}')
        return status

    # float() parses exactly; pd.to_numeric can be off by one ULP, so the max wouldn't match its source
    eq_odds = df[['tpr_diff', 'fpr_diff']].map(lambda s: float(s) if s else float('nan')).max(axis=1)
    df = df.drop(columns=COLUMN, errors='ignore')
    df.insert(df.columns.get_loc('fpr_diff') + 1, COLUMN, eq_odds.map(lambda v: '' if pd.isna(v) else repr(float(v))))
    df.to_csv(csv_path, index=False)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='List the files that would change; write nothing.')
    args = parser.parse_args()

    counts = {'migrated': 0, 'recomputed': 0, 'skipped': 0}
    for csv_path in sorted(RESULTS_ROOT.rglob(CSV_ASSESSMENT)):
        counts[migrate(csv_path, args.dry_run)] += 1

    prefix = '[dry-run] ' if args.dry_run else ''
    print(
        f'\n{prefix}{sum(counts.values())} file(s): {counts["migrated"]} migrated (column added), '
        f'{counts["recomputed"]} already had {COLUMN} (recomputed), '
        f'{counts["skipped"]} skipped (no tpr_diff/fpr_diff)'
    )


if __name__ == '__main__':
    main()
