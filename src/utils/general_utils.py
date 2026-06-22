import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--baseline_type', type=str, default='abcinml')
    parser.add_argument('--sensitive_attribute', type=str)
    parser.add_argument('--target_label', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--test_run', type=str, default='False')
    parser.add_argument('--forecasting_technique', type=str, default='sma')
    parser.add_argument('--forecast_window', type=int, default=5)
    parser.add_argument('--batch_type', type=str, default='monthly')

    parser.add_argument('--alpha_ema', type=float, default=0.5)

    args = parser.parse_args()

    alpha = args.alpha
    baseline_type = args.baseline_type
    sensitive_attribute = args.sensitive_attribute
    target_label = args.target_label
    model_name = args.model_name
    test_run = eval(args.test_run)
    forecasting_technique = args.forecasting_technique
    forecast_window = args.forecast_window
    batch_type = args.batch_type
    alpha_ema = args.alpha_ema

    if alpha == None:
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        alphas = [alpha]

    return alphas, baseline_type, sensitive_attribute, target_label, model_name, test_run, forecasting_technique, forecast_window, batch_type, alpha_ema


def get_window_column(batch_type):
    window_col_map = {
        'monthly': 'YEAR-MONTH',
        'weekly': 'YEAR-WEEK' ,
    }

    window_column = window_col_map.get(batch_type)
    return window_column


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
            'sp': [],
            'tpr': [],
            'fpr': [],
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
    shap_records = []

    return batches, historical_proportions, assessment_metrics, model, sample_weights, forecasted_distributions, shap_records, absolute_rates
