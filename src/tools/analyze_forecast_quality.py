"""
Evaluates the forecast quality of SMA and EMA by comparing predicted
joint-distribution proportions against observed values for each batch.

Usage:
    uv run python src/tools/analyze_forecast_quality.py \\
        --sensitive_attribute CS_RACA \\
        --forecast_window 10 \\
        --ema_alpha 0.3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from utils import data_utils, general_utils, metrics_utils, model_utils

_PROPORTIONS = ['S0_Y0', 'S0_Y1', 'S1_Y0', 'S1_Y1']
_FORECASTERS = ['sma', 'ema']
_FORECASTER_COLORS = {'sma': '#1f77b4', 'ema': '#ff7f0e'}


def _batch_type_from_window_column(window_column: str) -> str:
    return 'weekly' if window_column == 'YEAR-WEEK' else 'monthly'


def _assign_period(dt: pd.Timestamp) -> str:
    for start, end, name, _ in metrics_utils._PERIOD_SPANS:
        if pd.Timestamp(start) <= dt < pd.Timestamp(end):
            return name
    return 'unknown'


def run_forecast_quality(
    df: pd.DataFrame,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    forecast_window: int,
    ema_alpha: float,
) -> pd.DataFrame:
    batches = sorted(df[window_column].unique())
    historical_proportions = {k: [] for k in _PROPORTIONS}
    records = []

    for batch in batches:
        current_batch_df = df[df[window_column] == batch].copy()
        real_props = model_utils.compute_batch_proportions(
            current_batch_df, sensitive_attribute, target_label
        )

        if len(historical_proportions[_PROPORTIONS[0]]) >= forecast_window:
            for forecaster in _FORECASTERS:
                forecasted = model_utils.forecast_joint_distributions(
                    historical_proportions,
                    window=forecast_window,
                    forecasting_technique=forecaster,
                    alpha_ema=ema_alpha,
                )
                if forecasted is not None:
                    for prop in _PROPORTIONS:
                        fv = forecasted[prop]
                        rv = real_props[prop]
                        records.append({
                            'batch': str(batch),
                            'forecaster': forecaster,
                            'proportion': prop,
                            'forecast_value': fv,
                            'real_value': rv,
                            'abs_error': abs(fv - rv),
                            'bias': fv - rv,
                        })

        for key in historical_proportions:
            historical_proportions[key].append(real_props[key])

    return pd.DataFrame(records)


def compute_mae_by_period(df: pd.DataFrame, batch_type: str) -> pd.DataFrame:
    unique_batches = df['batch'].unique().tolist()
    batch_dates = metrics_utils._batch_to_datetime(unique_batches, batch_type)
    batch_to_dt = dict(zip(unique_batches, batch_dates))

    df = df.copy()
    df['period'] = df['batch'].map(lambda b: _assign_period(batch_to_dt[b]))

    agg = (
        df.groupby(['forecaster', 'period'])['abs_error']
        .agg(mae_mean='mean', mae_std='std')
        .reset_index()
    )
    return agg


def compute_mae_overall(df: pd.DataFrame) -> pd.DataFrame:
    def _rmse(x: pd.Series) -> float:
        return float(np.sqrt(np.mean(x**2)))

    agg = (
        df.groupby(['forecaster', 'proportion'])
        .agg(
            mae=('abs_error', 'mean'),
            rmse=('abs_error', _rmse),
            bias=('bias', 'mean'),
        )
        .reset_index()
    )
    return agg


def plot_error_over_time(df: pd.DataFrame, batch_type: str, out_path: Path) -> None:
    all_batch_labels = sorted(df['batch'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), squeeze=True)
    axes = axes.flatten()

    period_handles = []

    for idx, prop in enumerate(_PROPORTIONS):
        ax = axes[idx]

        handles = metrics_utils._add_period_spans(ax, all_batch_labels, batch_type)
        if not period_handles:
            period_handles = handles

        for forecaster in _FORECASTERS:
            sub = (
                df[(df['proportion'] == prop) & (df['forecaster'] == forecaster)]
                .set_index('batch')
                .reindex(all_batch_labels)
            )
            y = sub['abs_error'].values
            x_indices = [i for i, v in enumerate(y) if not np.isnan(v)]
            y_values = y[~np.isnan(y)]

            ax.plot(
                x_indices,
                y_values,
                label=forecaster.upper(),
                color=_FORECASTER_COLORS[forecaster],
                linewidth=1.8,
                alpha=0.85,
            )

        ax.set_title(f'Erro Absoluto — {prop}', fontsize=11, fontweight='bold')
        ax.set_ylabel('|forecast − real|', fontsize=9)
        metrics_utils._apply_bin_xticks(ax, all_batch_labels)
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(fontsize=9)

    fig.suptitle(
        'Forecast Quality: Absolute Error per Proportion over Time',
        fontsize=13,
        fontweight='bold',
    )

    if period_handles:
        fig.legend(
            handles=period_handles,
            loc='lower center',
            ncol=len(metrics_utils._PERIOD_SPANS),
            fontsize=9,
            framealpha=0.5,
            bbox_to_anchor=(0.5, -0.05),
        )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved to {out_path.resolve()}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate forecast quality (SMA vs EMA)')
    parser.add_argument('--sensitive_attribute', type=str, default='IDADE_BIN')
    parser.add_argument('--target_label', type=str, default='EVOLUCAO')
    parser.add_argument('--window_column', type=str, default='YEAR-MONTH')
    parser.add_argument('--forecast_window', type=int, default=10)
    parser.add_argument('--ema_alpha', type=float, default=0.3)
    args = parser.parse_args()

    batch_type = _batch_type_from_window_column(args.window_column)

    # forecast_window is given in months; convert to batches for the chosen batch_type.
    forecast_window_batches = general_utils.effective_forecast_window(args.forecast_window, batch_type)

    print('Loading data...')
    df = data_utils.pre_process_df(args.sensitive_attribute, args.window_column)

    print('Computing forecast quality...')
    quality_df = run_forecast_quality(
        df,
        window_column=args.window_column,
        sensitive_attribute=args.sensitive_attribute,
        target_label=args.target_label,
        forecast_window=forecast_window_batches,
        ema_alpha=args.ema_alpha,
    )

    out_dir = _SRC / 'results' / 'forecast_quality' / args.sensitive_attribute / f'{batch_type}_forecast_window_{args.forecast_window}_ema_alpha_{args.ema_alpha}'
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_df.to_csv(out_dir / 'forecast_quality.csv', index=False)
    print(f'Saved {len(quality_df)} rows -> {(out_dir / "forecast_quality.csv").resolve()}')

    mae_period = compute_mae_by_period(quality_df, batch_type)
    mae_period.to_csv(out_dir / 'mae_by_period.csv', index=False)
    print(f'Saved mae_by_period.csv ({len(mae_period)} rows)')

    mae_overall = compute_mae_overall(quality_df)
    mae_overall.to_csv(out_dir / 'mae_overall.csv', index=False)
    print(f'Saved mae_overall.csv ({len(mae_overall)} rows)')

    plot_error_over_time(quality_df, batch_type, out_dir / 'forecast_error_over_time.png')

    print('Done.')


if __name__ == '__main__':
    main()
