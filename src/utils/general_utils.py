import argparse
import pandas as pd

WEEKS_PER_MONTH = 52 / 12

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument(
        '--baseline_type',
        type=str,
        default='all',
        choices=['vanilla', 'static', 'dynamic', 'abcinml', 'all'],
    )
    parser.add_argument('--sensitive_attribute', type=str, default='IDADE_BIN')
    parser.add_argument('--target_label', type=str, default='EVOLUCAO')
    parser.add_argument('--model_name', type=str, choices=['logistic_regression', 'xgboost', 'random_forest', 'lightgbm'])
    parser.add_argument('--test_run', type=str, default='False', choices=['True', 'False'])
    parser.add_argument(
        '--forecasting_technique',
        type=str,
        default='sma',
        choices=['sma', 'ema', 'oracle'],
    )
    parser.add_argument('--forecast_window', type=int, default=5)
    parser.add_argument(
        '--train_window', type=int, default=None,
        help=(
            'Bound (in months) on how much history dynamic/abcinml train on. '
            'Default: None = unbounded (all prior batches, current behavior). '
            'vanilla/static ignore this — they always train on a single batch.'
        ),
    )
    parser.add_argument('--batch_type', type=str, default='monthly', choices=['monthly', 'weekly'])
    parser.add_argument('--alpha_ema', type=float, default=0.5)
    parser.add_argument(
        '--fairness_correction',
        type=str,
        default='kamiran_rw',
        choices=['kamiran_rw', 'adaptative_rw'],
    )
    parser.add_argument('--apr_iterations', type=int, default=20)
    parser.add_argument('--apr_eta', type=float, default=1.0)
    parser.add_argument('--apr_subgroup_alpha', type=float, default=100.0)
    parser.add_argument('--apr_decision_boundary', type=float, default=0.5)
    parser.add_argument(
        '--apr_fairness_measure',
        type=str,
        default='demographic_parity',
        choices=['demographic_parity', 'equalized_odds', 'equal_opportunity'],
    )

    args = parser.parse_args()

    alpha = args.alpha
    baseline_type = args.baseline_type
    sensitive_attribute = args.sensitive_attribute
    target_label = args.target_label
    model_name = args.model_name
    test_run = eval(args.test_run)
    forecasting_technique = args.forecasting_technique
    forecast_window = args.forecast_window
    train_window = args.train_window
    batch_type = args.batch_type
    alpha_ema = args.alpha_ema
    fairness_correction = args.fairness_correction
    apr_iterations = args.apr_iterations
    apr_eta = args.apr_eta
    apr_subgroup_alpha = args.apr_subgroup_alpha
    apr_decision_boundary = args.apr_decision_boundary
    apr_fairness_measure = args.apr_fairness_measure

    if alpha == None:
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        alphas = [alpha]

    return (
        alphas,
        baseline_type,
        sensitive_attribute,
        target_label,
        model_name,
        test_run,
        forecasting_technique,
        forecast_window,
        train_window,
        batch_type,
        alpha_ema,
        fairness_correction,
        apr_iterations,
        apr_eta,
        apr_subgroup_alpha,
        apr_decision_boundary,
        apr_fairness_measure,
    )


def get_window_column(batch_type):
    window_col_map = {
        'monthly': 'YEAR-MONTH',
        'weekly': 'YEAR-WEEK',
    }

    window_column = window_col_map.get(batch_type)
    return window_column




def effective_forecast_window(forecast_window_months, batch_type):
    """Convert forecast_window (in months) to the number of batches for batch_type."""
    if batch_type == 'weekly':
        return max(1, round(forecast_window_months * WEEKS_PER_MONTH))
    return forecast_window_months


def initialize_variables(df, window_column, alpha_ema=0.5):
    batches = sorted(df[window_column].unique())

    historical_proportions = {
        'S0_Y1': [],
        'S0_Y0': [],
        'S1_Y1': [],
        'S1_Y0': [],
    }

    assessment_metrics = pd.DataFrame(
        {
            'acc': [],
            'auc': [],
            'dp': [],
            'tpr_diff': [],
            'fpr_diff': [],
            'eq_odds_diff': [],
            'batch': [],
            'alpha': [],
        }
    )

    absolute_rates = pd.DataFrame(
        {
            'batch': [],
            'alpha': [],
            'tpr_S0': [],
            'fpr_S0': [],
            'tpr_S1': [],
            'fpr_S1': [],
        }
    )

    model = None
    sample_weights = None
    forecasted_distributions = None

    return (
        batches,
        historical_proportions,
        assessment_metrics,
        model,
        sample_weights,
        forecasted_distributions,
        absolute_rates,
    )
