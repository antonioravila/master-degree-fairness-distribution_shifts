"""
Vanilla baseline under distribution shift: monthly vs. weekly batching.

Shows the never-retrained `vanilla` baseline's fairness metrics (demographic
parity, TPR, and FPR differences) degrading over time as the underlying data
distribution shifts, using the same visual identity (period shading, baseline
colors, moving averages) as `src/notebooks/04_results_overview.ipynb`.

Usage:
    uv run python src/tools/vanilla_shift_chart.py --sensitive_attribute IDADE_BIN --model_name logistic_regression
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from utils import metrics_utils
from utils.metrics_utils import (
    _BASELINE_COLORS,
    _BATCH_FORMAT,
    _batch_to_datetime,
    _build_ma_series,
    _add_period_spans,
    _apply_bin_xticks,
)

_FAIRNESS_METRICS = [
    ('dp', 'Demographic Parity Diff.'),
    ('tpr_diff', 'TPR Difference'),
    ('fpr_diff', 'FPR Difference'),
]

_VANILLA_COLOR = _BASELINE_COLORS['vanilla']


def _load_vanilla(model_name: str, sensitive_attribute: str, batch_type: str, forecast_window: int) -> pd.DataFrame:
    folder = metrics_utils.baseline_non_abc_folder_name(model_name, batch_type, forecast_window=forecast_window)
    path = _SRC / 'results' / sensitive_attribute / folder / 'assessment_all_baselines.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found — run baseline_type=vanilla for batch_type={batch_type} first.'
        )
    df = pd.read_csv(path)
    return df[df['baseline'] == 'vanilla'].copy()


def _plot_row(axes, df: pd.DataFrame, batch_type: str, ma_window: int):
    df = df.copy()
    df['batch_dt'] = _batch_to_datetime(df['batch'].astype(str).tolist(), batch_type)
    df = df.sort_values('batch_dt')
    batch_labels = sorted(df['batch'].astype(str).unique())
    fmt = _BATCH_FORMAT[batch_type]

    metric_ylims = {}
    for col, _ in _FAIRNESS_METRICS:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        vmin, vmax = values.min(), values.max()
        padding = (vmax - vmin) * 0.05 or 0.01
        metric_ylims[col] = (vmin - padding, vmax + padding)

    period_handles = []
    raw_handle = ma_handle = None

    for col_idx, (col, label) in enumerate(_FAIRNESS_METRICS):
        ax = axes[col_idx]

        handles = _add_period_spans(ax, batch_labels, batch_type)
        if not period_handles:
            period_handles = handles

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5, zorder=2)

        x_pos_raw = [batch_labels.index(b) for b in df['batch'].astype(str)]
        raw_handle = ax.scatter(
            x_pos_raw, df[col], color=_VANILLA_COLOR, alpha=0.30, s=18, zorder=2, label='vanilla (raw)'
        )

        ma = _build_ma_series(df, col, ma_window)
        x_pos_ma = [batch_labels.index(d.strftime(fmt)) for d in ma.index if d.strftime(fmt) in batch_labels]
        ma_handle, = ax.plot(
            x_pos_ma, ma.values, color=_VANILLA_COLOR, linewidth=2.75, zorder=3,
            label='vanilla (moving avg.)',
        )

        ax.set_title(label, fontsize=22, fontweight='bold')
        ax.set_ylabel(label, fontsize=20)
        if col in metric_ylims:
            ax.set_ylim(metric_ylims[col])
        _apply_bin_xticks(ax, batch_labels)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

    return period_handles, raw_handle, ma_handle


def build_chart(
    sensitive_attribute: str,
    model_name: str,
    forecast_window: int,
    ma_window_monthly: int = 5,
    ma_window_weekly: int = 20,
) -> plt.Figure:
    monthly_df = _load_vanilla(model_name, sensitive_attribute, 'monthly', forecast_window)
    weekly_df = _load_vanilla(model_name, sensitive_attribute, 'weekly', forecast_window)

    fig = plt.figure(figsize=(18, 13), layout='constrained')
    subfigs = fig.subfigures(2, 1)

    rows = [
        ('monthly', monthly_df, ma_window_monthly),
        ('weekly', weekly_df, ma_window_weekly),
    ]

    period_handles, raw_handle, ma_handle = [], None, None
    for subfig, (batch_type, df_row, ma_window) in zip(subfigs, rows):
        axes = subfig.subplots(1, len(_FAIRNESS_METRICS))
        subfig.suptitle(f'batch type = {batch_type}', fontsize=22, fontweight='bold')
        ph, rh, mh = _plot_row(axes, df_row, batch_type, ma_window)
        if not period_handles:
            period_handles = ph
        raw_handle, ma_handle = rh, mh

    # Same post-plot alpha treatment as the 04_results_overview notebook
    for ax in fig.axes:
        for patch in ax.patches:
            patch.set_alpha(0.20)
        for text in ax.texts:
            text.set_fontsize(18)

    all_handles = [raw_handle, ma_handle] + period_handles
    legend_labels = (
        [raw_handle.get_label(), ma_handle.get_label()]
        + [h.get_label() for h in period_handles]
    )
    leg = fig.legend(
        all_handles, legend_labels,
        loc='outside lower center', ncol=len(all_handles), fontsize=20, framealpha=0.9,
    )
    for legend_patch in leg.get_patches():
        legend_patch.set_alpha(0.4)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sensitive_attribute', default='IDADE_BIN', choices=['CS_RACA', 'IDADE_BIN'])
    parser.add_argument('--model_name', default='logistic_regression', choices=['logistic_regression', 'xgboost', 'random_forest', 'lightgbm'])
    parser.add_argument('--forecast_window', type=int, default=3, help='Selects which baseline_non_abc_*_fw_{n} folder to read.')
    args = parser.parse_args()

    fig = build_chart(args.sensitive_attribute, args.model_name, args.forecast_window)

    out_dir = Path('src') / 'figures' / args.sensitive_attribute
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'vanilla_distribution_shift_{args.model_name}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Chart saved to {out_path.resolve()}')
    plt.close(fig)


if __name__ == '__main__':
    main()
