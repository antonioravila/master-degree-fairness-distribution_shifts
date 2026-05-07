import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference,
    true_positive_rate_difference,
    false_positive_rate_difference,
)

_SNAPSHOT_METRICS = [
    ('acc', 'Accuracy'),
    ('auc', 'AUC-ROC'),
    ('sp', 'Demographic Parity Diff.'),
    ('tpr', 'TPR Difference'),
    ('fpr', 'FPR Difference'),
]

_TEMPORAL_METRICS = ['Maximum Bias', 'Temporal Stability', 'Maximum Bias Difference']

_FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
_N_BINS = 6

_PERIOD_SPANS = [
    ('2019-02-01', '2020-03-01', 'pré-pandemia',     '#AED6F1'),
    ('2020-03-01', '2022-01-01', 'pandemia aguda 1', '#F1948A'),
    ('2022-01-01', '2023-01-01', 'pandemia aguda 2', '#F0B27A'),
    ('2023-01-01', '2024-01-01', 'transição',         '#A9DFBF'),
    ('2024-01-01', '2026-03-01', 'endêmico',          '#D2B4DE'),
]

_VLINES = [
    (pd.Timestamp('2020-03-01'), 'mar/2020'),
    (pd.Timestamp('2022-01-01'), 'jan/2022'),
    (pd.Timestamp('2023-01-01'), 'jan/2023'),
    (pd.Timestamp('2024-01-01'), 'jan/2024'),
]


def _apply_bin_xticks(ax, labels: list, n_bins: int = _N_BINS) -> None:
    n = len(labels)
    positions = np.linspace(0, n - 1, min(n_bins, n), dtype=int)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[p] for p in positions], rotation=45, ha='right')


def _add_period_spans(ax, batch_labels: list) -> list:
    try:
        batch_dates = pd.to_datetime([f'{b}-01' for b in batch_labels])
    except Exception:
        return []

    handles = []
    for start, end, name, color in _PERIOD_SPANS:
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
        idxs = [i for i, d in enumerate(batch_dates) if start_dt <= d < end_dt]
        if not idxs:
            continue
        patch = ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5,
                           alpha=0.18, color=color, zorder=0, label=name)
        handles.append(patch)

    for vline_dt, vline_label in _VLINES:
        idxs = [i for i, d in enumerate(batch_dates) if d >= vline_dt]
        if idxs:
            x = min(idxs)
            ax.axvline(x=x, color='#555555', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
            ax.text(x, 1.0, vline_label, transform=ax.get_xaxis_transform(),
                    rotation=90, fontsize=6, va='top', ha='right', color='#555555')

    return handles


def generate_charts(
    assessment_results_df: pd.DataFrame,
    temporal_results_df: pd.DataFrame,
    baseline_type: str = '',
    save: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    alphas = assessment_results_df['alpha'].unique()

    batch_labels = assessment_results_df['batch'].astype(str).unique().tolist()

    _MA_WINDOW = 10
    period_handles = []

    for ax, (col, label) in zip(axes, _SNAPSHOT_METRICS):
        handles = _add_period_spans(ax, batch_labels)
        if not period_handles:
            period_handles = handles

        for alpha in alphas:
            subset = assessment_results_df[assessment_results_df['alpha'] == alpha].reset_index(drop=True)
            x_str = subset['batch'].astype(str).tolist()
            y = subset[col]
            ax.plot(x_str, y, marker='o', alpha=0.3, label=f'α={alpha}')
            ma = y.rolling(window=_MA_WINDOW, min_periods=1).mean()
            ax.plot(x_str, ma, linewidth=2, alpha=0.9, color='green', label=f'MA-{_MA_WINDOW} α={alpha}')
        ax.set_title(label)
        ax.set_ylabel(label)
        _apply_bin_xticks(ax, batch_labels)
        if len(alphas) > 1:
            ax.legend(fontsize=7)

    title = 'Model Performance & Fairness Metrics'
    if baseline_type:
        title += f'\nBaseline: {baseline_type}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if period_handles:
        fig.legend(handles=period_handles, loc='lower center', ncol=len(_PERIOD_SPANS),
                   fontsize=16, framealpha=0.5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        os.makedirs(_FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(_FIGURES_DIR, 'model_performance_metrics.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Chart saved to {os.path.abspath(out_path)}')


def compute_temporal_metrics(
    assessment_metrics: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    mb = assessment_metrics['sp'].max()
    ts = abs(assessment_metrics['sp'].diff().mean())
    mbd = abs(assessment_metrics['sp'].diff().max())

    temporal_metrics = pd.DataFrame(
        {
            'Maximum Bias': [mb],
            'Temporal Stability': [ts],
            'Maximum Bias Difference': [mbd],
        }
    )

    return temporal_metrics


def compute_snapshot_metrics(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_pred_proba: np.array,
    sensitive_features: str,
) -> tuple[float]:

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sp = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )
    tpr = true_positive_rate_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )
    fpr = false_positive_rate_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )

    return acc, auc, sp, tpr, fpr
