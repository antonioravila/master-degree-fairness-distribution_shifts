import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
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


def generate_all_charts(baseline_type, baselines, alpha, sensitive_attribute, all_assessment_df, all_temporal_df, all_absolute_df, all_shap_records, model_name, forecast_window, forecasting_technique, batch_type):
    chart_label = 'all' if baseline_type == 'all' else baseline_type
    run_folder = f'{model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}_{batch_type}'
    subfolder = os.path.join(sensitive_attribute, run_folder)
    _generate_performance_charts(all_assessment_df, all_temporal_df, baseline_type=chart_label, save=True, subfolder=subfolder, model_name=model_name, forecast_window=forecast_window)
    if model_name != 'xgboost':
        _generate_shap_charts(all_shap_records, save=True, subfolder=subfolder, model_name=model_name, forecast_window=forecast_window)

    # Save consolidated results
    results_dir = os.path.join('src', 'results', sensitive_attribute, run_folder)
    os.makedirs(results_dir, exist_ok=True)
    all_assessment_df.to_csv(os.path.join(results_dir, 'assessment_all_baselines.csv'), index=False)
    all_temporal_df.to_csv(os.path.join(results_dir, 'temporal_results_df.csv'), index=False)
    all_absolute_df.to_csv(os.path.join(results_dir, 'absolute_results_df.csv'), index=False)

    # Comparative charts (only when multiple baselines were run)
    if len(baselines) > 1:
        _generate_comparative_chart(all_assessment_df, moving_avg_window=5, save=True, subfolder=subfolder, model_name=model_name, forecast_window=forecast_window)
        _generate_absolute_rates_chart(all_absolute_df, save=True, subfolder=subfolder, model_name=model_name, forecast_window=forecast_window)


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


def _generate_performance_charts(
    assessment_results_df: pd.DataFrame,
    temporal_results_df: pd.DataFrame,
    baseline_type: str = '',
    save: bool = False,
    subfolder: str = '',
    model_name: str = '',
    forecast_window: int = 0,
) -> None:
    _MA_WINDOW = 10
    N_COLS = len(_SNAPSHOT_METRICS)

    has_baselines = 'baseline' in assessment_results_df.columns
    baselines = assessment_results_df['baseline'].unique().tolist() if has_baselines else [None]
    n_rows = len(baselines)

    batch_labels = assessment_results_df['batch'].astype(str).unique().tolist()
    alphas = assessment_results_df['alpha'].unique()

    metric_ylims = {}
    for col, _ in _SNAPSHOT_METRICS:
        values = assessment_results_df[col].dropna()
        if len(values) == 0:
            continue
        vmin, vmax = values.min(), values.max()
        padding = (vmax - vmin) * 0.05
        metric_ylims[col] = (vmin - padding, vmax + padding)

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(30, 5 * n_rows), squeeze=False)

    period_handles = []

    for row, bt in enumerate(baselines):
        row_df = assessment_results_df[assessment_results_df['baseline'] == bt] if has_baselines else assessment_results_df

        for col_idx, (col, label) in enumerate(_SNAPSHOT_METRICS):
            ax = axes[row][col_idx]

            handles = _add_period_spans(ax, batch_labels)
            if not period_handles:
                period_handles = handles

            for alpha in alphas:
                subset = row_df[row_df['alpha'] == alpha].reset_index(drop=True)
                x_str = subset['batch'].astype(str).tolist()
                y = subset[col]
                ax.plot(x_str, y, marker='o', alpha=0.3, label=f'α={alpha}')
                ma = y.rolling(window=_MA_WINDOW, min_periods=1).mean()
                ax.plot(x_str, ma, linewidth=2, alpha=0.9, color='green', label=f'MA-{_MA_WINDOW} α={alpha}')

            ax.set_title(label)
            ax.set_ylabel(label)

            if col in metric_ylims:
                ax.set_ylim(metric_ylims[col])

            _apply_bin_xticks(ax, batch_labels)

            if len(alphas) > 1:
                ax.legend(fontsize=7)

        # baseline label centered above the row
        if has_baselines and bt:
            mid_ax = axes[row][N_COLS // 2]
            mid_ax.annotate(
                f'Baseline: {bt}',
                xy=(0.5, 1.08),
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#888888', alpha=0.9),
            )

    title = f'Model Performance & Fairness Metrics — {model_name} | window={forecast_window}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if period_handles:
        fig.legend(handles=period_handles, loc='lower center', ncol=len(_PERIOD_SPANS),
                   fontsize=16, framealpha=0.5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save:
        save_dir = os.path.join(_FIGURES_DIR, subfolder) if subfolder else _FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'model_performance_metrics.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Chart saved to {os.path.abspath(out_path)}')


def compute_temporal_metrics(
    assessment_metrics: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    mb = assessment_metrics['sp'].abs().max()
    ts = assessment_metrics['sp'].diff().abs().mean()
    mbd = assessment_metrics['sp'].diff().abs().max()

    temporal_metrics = pd.DataFrame(
        {
            'Maximum Bias': [mb],
            'Temporal Stability': [ts],
            'Maximum Bias Difference': [mbd],
        }
    )

    return temporal_metrics


def compute_snapshot_metrics(
    y_test: pd.Series,
    y_pred: np.array,
    y_pred_proba: np.array,
    sensitive_features: pd.Series,
) -> tuple[float]:
    
    def _group_rates(y_test, y_pred, sensitive_features, sensitive_value):
        mask = sensitive_features == sensitive_value
        tn, fp, fn, tp = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return tpr, fpr

    tpr_S0, fpr_S0 = _group_rates(y_test, y_pred, sensitive_features, 0)
    tpr_S1, fpr_S1 = _group_rates(y_test, y_pred, sensitive_features, 1)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )
    tpr_diff = true_positive_rate_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )
    fpr_diff = false_positive_rate_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features
    )

    return acc, auc, sp_diff, tpr_diff, fpr_diff, tpr_S0, fpr_S0, tpr_S1, fpr_S1


def compute_shap_values(model_name, model, X_test: pd.DataFrame) -> dict:
    if model_name.lower() == 'xgboost':
        explainer = shap.TreeExplainer(model)
    elif model_name.lower() == 'logistic_regression':
        explainer = shap.LinearExplainer(model, X_test)
    else: raise ValueError(f"Unsupported model_name: {model_name}")

    values = explainer.shap_values(X_test)
    return {'shap_values': values, 'feature_names': list(X_test.columns)}


_BASELINE_COLORS = {
    'vanilla':  '#555555',
    'static':   '#1f77b4',
    'dynamic':  '#ff7f0e',
    'abcinml':  '#2ca02c',
}

_BASELINE_ORDER = ['vanilla', 'static', 'dynamic', 'abcinml']


def _build_ma_series(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Return moving-average series sorted by batch, with datetime index."""
    ordered = df.sort_values('batch_dt')
    return ordered.set_index('batch_dt')[col].rolling(window=window, min_periods=1).mean()


def _generate_comparative_chart(
    assessment_results_df: pd.DataFrame,
    moving_avg_window: int = 5,
    save: bool = False,
    subfolder: str = '',
    model_name: str = '',
    forecast_window: int = 0,
) -> None:
    """Overlay moving-average lines for all baselines, one subplot per metric."""
    if 'baseline' not in assessment_results_df.columns:
        raise ValueError("assessment_results_df must have a 'baseline' column")

    df = assessment_results_df.copy()
    df['batch_dt'] = pd.to_datetime([f'{b}-01' for b in df['batch'].astype(str)])

    # For baselines where multiple alphas exist (abcinml), average across alphas first
    df = df.groupby(['baseline', 'batch', 'batch_dt'], as_index=False)[
        [col for col, _ in _SNAPSHOT_METRICS]
    ].mean()

    present_baselines = [b for b in _BASELINE_ORDER if b in df['baseline'].unique()]

    # Common batch range across all baselines
    common_dates = None
    for bt in present_baselines:
        dates = set(df[df['baseline'] == bt]['batch_dt'])
        common_dates = dates if common_dates is None else common_dates & dates
    common_dates = sorted(common_dates) if common_dates else []

    metric_ylims = {}
    for col, _ in _SNAPSHOT_METRICS:
        values = assessment_results_df[col].dropna()
        if len(values) == 0:
            continue
        vmin, vmax = values.min(), values.max()
        padding = (vmax - vmin) * 0.05
        metric_ylims[col] = (vmin - padding, vmax + padding)

    N_COLS = len(_SNAPSHOT_METRICS)
    fig, axes = plt.subplots(1, N_COLS, figsize=(28, 6), squeeze=False)
    axes = axes[0]

    period_handles = []
    baseline_handles = []

    # Use common batch labels list for xticks
    all_batch_labels = sorted(df['batch'].astype(str).unique())

    for col_idx, (col, label) in enumerate(_SNAPSHOT_METRICS):
        ax = axes[col_idx]

        handles = _add_period_spans(ax, all_batch_labels)
        if not period_handles:
            period_handles = handles

        # Fairness reference lines
        is_fairness = col in ('sp', 'tpr', 'fpr')
        if is_fairness:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5, zorder=2)

        for bt in present_baselines:
            sub = df[df['baseline'] == bt]
            ma = _build_ma_series(sub, col, moving_avg_window)
            # restrict to common dates
            ma = ma[ma.index.isin(common_dates)] if common_dates else ma

            x_pos = [all_batch_labels.index(d.strftime('%Y-%m')) for d in ma.index if d.strftime('%Y-%m') in all_batch_labels]
            line, = ax.plot(
                x_pos, ma.values,
                color=_BASELINE_COLORS.get(bt, None),
                linewidth=2.5,
                label=bt,
                zorder=3,
            )
            if col_idx == 0:
                baseline_handles.append(line)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=9)

        if col in metric_ylims:
            ax.set_ylim(metric_ylims[col])

        _apply_bin_xticks(ax, all_batch_labels)
        ax.tick_params(axis='x', labelsize=8)

    fig.suptitle(
        f'Comparação de Baselines — Médias Móveis (janela = {moving_avg_window} batches)',
        fontsize=13, fontweight='bold',
    )

    all_legend_handles = baseline_handles + period_handles
    all_legend_labels = [h.get_label() for h in baseline_handles] + [h.get_label() for h in period_handles]
    fig.legend(
        all_legend_handles, all_legend_labels,
        loc='lower center', ncol=len(all_legend_handles),
        fontsize=9, framealpha=0.6, bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save:
        save_dir = os.path.join(_FIGURES_DIR, subfolder) if subfolder else _FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'model_performance_comparison.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f'Comparative chart saved to {os.path.abspath(out_path)}')


def _generate_absolute_rates_chart(
    all_absolute_df: pd.DataFrame,
    save: bool = False,
    subfolder: str = '',
    model_name: str = '',
    forecast_window: int = 0,
) -> None:
    """Grid N×2: each row = one baseline, col 1 = TPR by group, col 2 = FPR by group."""
    _MA_WINDOW = 10
    _S0_COLOR = '#ff7f0e'
    _S1_COLOR = '#1f77b4'

    df = all_absolute_df.copy()
    df['batch_dt'] = pd.to_datetime([f'{b}-01' for b in df['batch'].astype(str)])

    rate_cols = ['tpr_S0', 'fpr_S0', 'tpr_S1', 'fpr_S1']
    df = df.groupby(['baseline', 'batch', 'batch_dt'], as_index=False)[rate_cols].mean()

    present_baselines = [b for b in _BASELINE_ORDER if b in df['baseline'].unique()]
    n_rows = len(present_baselines)
    all_batch_labels = sorted(df['batch'].astype(str).unique())

    tpr_values = pd.concat([df['tpr_S0'], df['tpr_S1']])
    tpr_min, tpr_max = tpr_values.min(), tpr_values.max()
    tpr_padding = (tpr_max - tpr_min) * 0.05
    tpr_ylim = (tpr_min - tpr_padding, tpr_max + tpr_padding)

    fpr_values = pd.concat([df['fpr_S0'], df['fpr_S1']])
    fpr_min, fpr_max = fpr_values.min(), fpr_values.max()
    fpr_padding = (fpr_max - fpr_min) * 0.05
    fpr_ylim = (fpr_min - fpr_padding, fpr_max + fpr_padding)

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows), squeeze=False)

    period_handles = []

    _RATE_PAIRS = [
        ('tpr_S0', 'tpr_S1', 'TPR (True Positive Rate)', tpr_ylim),
        ('fpr_S0', 'fpr_S1', 'FPR (False Positive Rate)', fpr_ylim),
    ]

    for row, bt in enumerate(present_baselines):
        sub = df[df['baseline'] == bt].sort_values('batch_dt').reset_index(drop=True)
        x_str = sub['batch'].astype(str).tolist()
        x_idx = [all_batch_labels.index(b) for b in x_str if b in all_batch_labels]

        for col_idx, (col_s0, col_s1, col_label, ylim) in enumerate(_RATE_PAIRS):
            ax = axes[row][col_idx]

            handles = _add_period_spans(ax, all_batch_labels)
            if not period_handles:
                period_handles = handles

            ma_s0 = sub[col_s0].rolling(window=_MA_WINDOW, min_periods=1).mean().values
            ma_s1 = sub[col_s1].rolling(window=_MA_WINDOW, min_periods=1).mean().values

            ax.plot(x_idx, ma_s0, color=_S0_COLOR, linewidth=2.0, label='S0 (privileged)')
            ax.plot(x_idx, ma_s1, color=_S1_COLOR, linewidth=2.0, label='S1 (sensitive)')
            ax.fill_between(x_idx, ma_s0, ma_s1, alpha=0.15, color='gray')

            ax.set_title(col_label, fontsize=10)
            ax.set_ylabel(col_label, fontsize=9)
            ax.set_ylim(ylim)
            ax.legend(fontsize=8)
            _apply_bin_xticks(ax, all_batch_labels)
            ax.tick_params(axis='x', labelsize=8)

        # Baseline label above the row (on right column, matching generate_charts style)
        axes[row][1].annotate(
            f'Baseline: {bt}',
            xy=(0.5, 1.08),
            xycoords='axes fraction',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#888888', alpha=0.9),
        )

    fig.suptitle('Taxas Absolutas TPR e FPR por Grupo Sensível', fontsize=14, fontweight='bold')

    if period_handles:
        fig.legend(
            handles=period_handles,
            loc='lower center',
            ncol=len(_PERIOD_SPANS),
            fontsize=9,
            framealpha=0.5,
            bbox_to_anchor=(0.5, -0.05),
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        save_dir = os.path.join(_FIGURES_DIR, subfolder) if subfolder else _FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'absolute_rates_by_group.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Absolute rates chart saved to {os.path.abspath(out_path)}')


def _generate_shap_charts(shap_records: list, save: bool = False, subfolder: str = '') -> None:
    if not shap_records:
        return

    feature_names = shap_records[0]['feature_names']
    shap_records = sorted(shap_records, key=lambda r: pd.to_datetime(r['batch']))
    batches = [r['batch'] for r in shap_records]

    # mean |SHAP| per feature per batch → shape (n_batches, n_features)
    mean_abs = np.array([
        np.abs(r['shap_values']).mean(axis=0) for r in shap_records
    ])

    global_importance = mean_abs.mean(axis=0)
    sorted_idx = np.argsort(global_importance)[::-1]

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(20, 12))

    # --- global bar chart ---
    ax_bar.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        global_importance[sorted_idx[::-1]],
        color='steelblue',
    )
    ax_bar.set_xlabel('Mean |SHAP value|')
    ax_bar.set_title('Global Feature Importance (all batches)')

    # --- temporal heatmap ---
    heatmap_data = mean_abs[:, sorted_idx].T  # (n_features, n_batches)
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax_heat.set_yticks(range(len(feature_names)))
    ax_heat.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)

    n_batches = len(batches)
    tick_positions = np.linspace(0, n_batches - 1, min(_N_BINS, n_batches), dtype=int)
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels([batches[p] for p in tick_positions], rotation=45, ha='right', fontsize=8)
    ax_heat.set_title('SHAP Feature Importance Over Time')
    fig.colorbar(im, ax=ax_heat, label='Mean |SHAP value|')

    batch_dates = pd.to_datetime([f'{b}-01' for b in batches])
    for vline_dt, vline_label in _VLINES:
        idxs = [i for i, d in enumerate(batch_dates) if d >= vline_dt]
        if idxs:
            x = min(idxs) - 0.5
            ax_heat.axvline(x=x, color='#555555', linestyle='--', linewidth=0.8, alpha=0.5)
            ax_heat.text(x, -0.5, vline_label, rotation=90, fontsize=6,
                         va='bottom', ha='right', color='#555555')

    fig.suptitle('SHAP Value Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        save_dir = os.path.join(_FIGURES_DIR, subfolder) if subfolder else _FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'shap_summary.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'SHAP chart saved to {os.path.abspath(out_path)}')
