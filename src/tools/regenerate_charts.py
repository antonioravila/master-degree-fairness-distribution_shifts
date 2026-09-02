"""
Regenerates all charts from saved CSV results without re-running the experiment.
SHAP charts are skipped — SHAP values are not persisted to disk.

Usage:
    # single run folder
    uv run python src/tools/regenerate_charts.py --results_dir src/results/IDADE_BIN/logistic_regression_alpha_0.5_sma_window_5_monthly

    # all run folders under a sensitive-attribute directory
    uv run python src/tools/regenerate_charts.py --results_dir src/results/IDADE_BIN
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from utils import metrics_utils

_CSV_ASSESSMENT = 'assessment_all_baselines.csv'
_CSV_TEMPORAL = 'temporal_results_df.csv'
_CSV_ABSOLUTE = 'absolute_results_df.csv'

# Matches the optional _ema_{alpha_ema} suffix that appears after the batch_type token,
# tolerating an optional trailing _{fairness_correction} suffix (e.g. _adaptative_rw).
_EMA_SUFFIX_RE = re.compile(r'(?:monthly|weekly)_ema_([\d.]+)(?:_[a-z_]+)?$')


def _parse_alpha_ema(folder_name: str) -> float | None:
    m = _EMA_SUFFIX_RE.search(folder_name)
    return float(m.group(1)) if m else None


def _load_non_abc(run_dir: Path, model_name: str, batch_type: str, fairness_correction: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    non_abc_folder = metrics_utils.baseline_non_abc_folder_name(model_name, batch_type, fairness_correction)
    non_abc_dir = run_dir.parent / non_abc_folder
    if not (non_abc_dir / _CSV_ASSESSMENT).exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    print(f'  merging non-ABC results from {non_abc_dir.name}')
    return (
        pd.read_csv(non_abc_dir / _CSV_ASSESSMENT),
        pd.read_csv(non_abc_dir / _CSV_TEMPORAL),
        pd.read_csv(non_abc_dir / _CSV_ABSOLUTE),
    )


def _regenerate_one(run_dir: Path, sensitive_attribute: str) -> None:
    assessment_path = run_dir / _CSV_ASSESSMENT
    temporal_path = run_dir / _CSV_TEMPORAL
    absolute_path = run_dir / _CSV_ABSOLUTE

    missing = [p.name for p in (assessment_path, temporal_path, absolute_path) if not p.exists()]
    if missing:
        print(f'Skipping {run_dir.name}: missing {", ".join(missing)}')
        return

    all_assessment_df = pd.read_csv(assessment_path)
    all_temporal_df = pd.read_csv(temporal_path)
    all_absolute_df = pd.read_csv(absolute_path)

    if all_assessment_df.empty:
        print(f'Skipping {run_dir.name}: assessment CSV is empty')
        return

    row = all_assessment_df.iloc[0]
    model_name = str(row['model_name'])
    forecast_window = int(row['forecast_window'])
    forecasting_technique = str(row['forecasting_technique'])
    batch_type = str(row.get('batch_type', 'monthly'))
    alpha = float(all_assessment_df['alpha'].iloc[0])
    alpha_ema = _parse_alpha_ema(run_dir.name)
    fairness_correction = str(row.get('fairness_correction', 'kamiran_rw'))

    is_non_abc = run_dir.name.startswith('baseline_non_abc')
    subfolder_override = os.path.join(sensitive_attribute, run_dir.name) if is_non_abc else None

    # For per-alpha (abcinml) folders, merge non-abc results so comparative charts include all baselines
    if not is_non_abc:
        non_abc_assessment, non_abc_temporal, non_abc_absolute = _load_non_abc(run_dir, model_name, batch_type, fairness_correction)
        if not non_abc_assessment.empty:
            all_assessment_df = pd.concat([non_abc_assessment, all_assessment_df], ignore_index=True)
            all_temporal_df = pd.concat([non_abc_temporal, all_temporal_df], ignore_index=True)
            all_absolute_df = pd.concat([non_abc_absolute, all_absolute_df], ignore_index=True)

    baselines = all_assessment_df['baseline'].unique().tolist()
    baseline_type = 'all' if len(baselines) > 1 else baselines[0]

    print(f'Regenerating: {run_dir.name}')
    print(f'  model={model_name}  alpha={alpha}  baselines={baselines}  batch_type={batch_type}  alpha_ema={alpha_ema}')

    metrics_utils.generate_all_charts(
        baseline_type,
        baselines,
        alpha,
        sensitive_attribute,
        all_assessment_df,
        all_temporal_df,
        all_absolute_df,
        [],  # SHAP values not persisted; shap_summary.png will be skipped
        model_name,
        forecast_window,
        forecasting_technique,
        batch_type,
        alpha_ema,
        fairness_correction=fairness_correction,
        subfolder_override=subfolder_override,
    )
    plt.close('all')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Regenerate charts from saved CSV results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results_dir',
        required=True,
        help=(
            'Path to a single run folder (contains the three CSVs directly) '
            'or to a sensitive-attribute folder to process all sub-runs.'
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f'Error: {results_dir} does not exist.')
        sys.exit(1)

    if (results_dir / _CSV_ASSESSMENT).exists():
        # Pointed directly at a run folder
        sensitive_attribute = results_dir.parent.name
        _regenerate_one(results_dir, sensitive_attribute)
    else:
        # Treat as a sensitive-attribute parent; process every child run folder
        sensitive_attribute = results_dir.name
        run_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
        if not run_dirs:
            print(f'No run sub-folders found under {results_dir}')
            sys.exit(1)
        for run_dir in run_dirs:
            _regenerate_one(run_dir, sensitive_attribute)

    print('Done.')


if __name__ == '__main__':
    main()
