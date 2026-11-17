"""
Analyzes the impact of the alpha parameter on ABCInML by generating a figure
with three statistical tables (max, min, mean) of per-batch metrics grouped by alpha,
with each table divided into subsections per model.

Usage:
    uv run python src/tools/alpha_analysis.py
    uv run python src/tools/alpha_analysis.py --results_dir src/results/all --output src/tools/alpha_analysis.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

_DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results', 'all')
_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'alpha_analysis.png')

_METRICS = {
    'acc': 'Accuracy',
    'auc': 'AUC',
    'dp': 'Dem. Parity Diff.',
    'tpr_diff': 'TPR Difference',
    'fpr_diff': 'FPR Difference',
}

_STAT_LABELS = {
    'max': 'Maximum',
    'min': 'Minimum',
    'mean': 'Mean',
}

_HEADER_COLOR = '#2C3E50'
_MODEL_HEADER_COLOR = '#1A5276'
_ROW_COLORS = ['#EAF2FB', '#FDFEFE']
_HEADER_TEXT_COLOR = 'white'
_CELL_TEXT_COLOR = '#2C3E50'


def _load_abcinml(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, 'assessment_all_baselines.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    df = pd.read_csv(path)
    df = df[df['baseline'] == 'abcinml'].copy()
    if df.empty:
        raise ValueError(f'No abcinml rows found in {path}')
    if 'model_name' not in df.columns:
        raise KeyError(
            "'model_name' column not found. Run src/tools/concatenate_results.py first."
        )
    # This script's groupby('alpha') silently mixes rows across any other axis
    # that varies (fairness_correction, train_window) — warn instead of failing,
    # since older aggregates may not have these columns at all.
    for col in ('fairness_correction', 'train_window'):
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            print(
                f"Warning: multiple distinct '{col}' values found "
                f'({sorted(df[col].astype(str).unique())}) — the tables below mix them.'
            )
    return df


def _build_stat_table(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    grouped = df.groupby('alpha')[list(_METRICS.keys())]
    if stat == 'max':
        table = grouped.max()
    elif stat == 'min':
        table = grouped.min()
    else:
        table = grouped.mean()
    table.index = [f'α = {a}' for a in table.index]
    table.columns = list(_METRICS.values())
    return table


def _render_table(ax: plt.Axes, table: pd.DataFrame, model_label: str) -> None:
    ax.axis('off')

    col_labels = list(table.columns)
    row_labels = list(table.index)
    cell_values = [[f'{v:.4f}' for v in row] for row in table.values]
    row_colors = [_ROW_COLORS[i % 2] for i in range(len(row_labels))]

    tbl = ax.table(
        cellText=cell_values,
        rowLabels=row_labels,
        colLabels=col_labels,
        rowColours=row_colors,
        cellLoc='center',
        rowLoc='center',
        loc='center',
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if row == 0:
            cell.set_facecolor(_HEADER_COLOR)
            cell.set_text_props(color=_HEADER_TEXT_COLOR, fontweight='bold')
        elif col == -1:
            cell.set_facecolor('#D5DBDB')
            cell.set_text_props(color=_CELL_TEXT_COLOR, fontweight='bold')
        else:
            cell.set_text_props(color=_CELL_TEXT_COLOR)

    ax.set_title(model_label, fontsize=11, fontweight='bold',
                 color=_MODEL_HEADER_COLOR, pad=10)


def main() -> None:
    parser = argparse.ArgumentParser(description='Alpha impact analysis for ABCInML')
    parser.add_argument('--results_dir', default=_DEFAULT_RESULTS,
                        help='Directory containing assessment_all_baselines.csv')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT,
                        help='Output image path')
    args = parser.parse_args()

    df = _load_abcinml(args.results_dir)
    models = sorted(df['model_name'].unique())
    alphas = sorted(df['alpha'].unique())
    print(f'ABCInML data loaded — models: {models}, alphas: {alphas}')

    n_stats = len(_STAT_LABELS)
    n_models = len(models)

    # Layout: one row per stat, one column per model
    fig, axes = plt.subplots(
        n_stats, n_models,
        figsize=(max(13, 8 * n_models), 4.5 * n_stats),
        squeeze=False,
    )

    for row_idx, (stat, stat_label) in enumerate(_STAT_LABELS.items()):
        # Left-side stat label spanning all model columns
        fig.text(
            0.01,
            1 - (row_idx + 0.5) / n_stats,
            stat_label,
            va='center', ha='left',
            fontsize=13, fontweight='bold', color=_HEADER_COLOR,
            rotation=90,
        )

        for col_idx, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            table = _build_stat_table(model_df, stat)
            _render_table(axes[row_idx][col_idx], table, model)

    fig.suptitle(
        'Impact of the α parameter on ABCInML — Per-batch statistics',
        fontsize=15, fontweight='bold', color=_HEADER_COLOR, y=1.02,
    )

    plt.tight_layout(h_pad=4, w_pad=3)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {out_path}')


if __name__ == '__main__':
    main()
