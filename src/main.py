import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from fairlearn.metrics import (
    demographic_parity_difference,
    true_positive_rate_difference,
    false_positive_rate_difference,
)


def forecast_joint_distributions(
    historical_proportions: dict[str, list[float]],
    window: int = 3,
) -> dict[str, float]:
    forecasted = {}
    for key, history in historical_proportions.items():
        series = pd.Series(history)
        rolling_avg = series.rolling(window=window, min_periods=1).mean()
        forecasted[key] = float(rolling_avg.iloc[-1])

    return forecasted


def pre_process_df(
    df: pd.DataFrame,
    sensitive_attribute: str,
    sensitive_value: str | int,
    favored_value: str | int,
) -> pd.DataFrame:
    df[sensitive_attribute] = df[sensitive_attribute].replace(
        {sensitive_value: 1, favored_value: 0}
    )
    return df


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


def weight_estimation(
    df: pd.DataFrame,
    sensitive_attribute: str,
    target_label: str,
) -> np.array:

    n = df.shape[0]
    positve = df[df[target_label] == 1].shape[0]
    negative = df[df[target_label] == 0].shape[0]

    deprived = (df[df[sensitive_attribute] == 0]).shape[0]
    favored = (df[df[sensitive_attribute] == 1]).shape[0]

    DP = df[(df[sensitive_attribute] == 0) & (df[target_label] == 1)].shape[0]
    DN = df[(df[sensitive_attribute] == 0) & (df[target_label] == 0)].shape[0]
    FP = df[(df[sensitive_attribute] == 1) & (df[target_label] == 1)].shape[0]
    FN = df[(df[sensitive_attribute] == 1) & (df[target_label] == 0)].shape[0]

    W_DP = (deprived * positve) / (n * DP)
    W_DN = (deprived * negative) / (n * DN)
    W_FP = (favored * positve) / (n * FP)
    W_FN = (favored * negative) / (n * FN)

    return np.array([W_DP, W_DN, W_FP, W_FN])


def weight_estimation_anticipated(
    forecasted: dict[str, float],
) -> np.array:
    p_S0_Y1 = forecasted['S0_Y1']
    p_S0_Y0 = forecasted['S0_Y0']
    p_S1_Y1 = forecasted['S1_Y1']
    p_S1_Y0 = forecasted['S1_Y0']

    n = p_S0_Y1 + p_S0_Y0 + p_S1_Y1 + p_S1_Y0

    deprived = p_S0_Y1 + p_S0_Y0
    favored = p_S1_Y1 + p_S1_Y0
    positive = p_S0_Y1 + p_S1_Y1
    negative = p_S0_Y0 + p_S1_Y0

    W_DP = (deprived * positive) / (n * p_S0_Y1)
    W_DN = (deprived * negative) / (n * p_S0_Y0)
    W_FP = (favored * positive) / (n * p_S1_Y1)
    W_FN = (favored * negative) / (n * p_S1_Y0)

    return np.array([W_DP, W_DN, W_FP, W_FN])


def reweighing(
    df: pd.DataFrame, sensitive_attribute: str, target_label: str, weights: np.array
) -> pd.DataFrame:

    W_DP, W_DN, W_FP, W_FN = weights

    df_ = df.copy()
    df_['weight'] = np.ones(len(df_))

    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 0) & (df_[target_label] == 1), W_DP, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 0) & (df_[target_label] == 0), W_DN, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 1) & (df_[target_label] == 1), W_FP, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 1) & (df_[target_label] == 0), W_FN, df_['weight']
    )

    return df_


def reweigh_df(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    sensitive_attribute: str,
    target_label: str,
    baseline_type: str,
    alpha: float = 0.0,
    forecasted_distributions: dict[str, float] = None,
) -> np.array:

    print('(1.1.1) reweighing dataframe.')
    train_df = X_train.copy()
    train_df[target_label] = y_train

    w_current = weight_estimation(train_df, sensitive_attribute, target_label)

    if baseline_type == 'abcinml' and forecasted_distributions is not None:
        print('(1.1.4) forecasting distributions')
        w_future = weight_estimation_anticipated(forecasted_distributions)
        w_new = alpha * w_current + (1 - alpha) * w_future
        print(
            f'Applied alpha blending: α={alpha}, '
            f'forecasted S0_Y1={forecasted_distributions["S0_Y1"]:.4f}, '
            f'S0_Y0={forecasted_distributions["S0_Y0"]:.4f}, '
            f'S1_Y1={forecasted_distributions["S1_Y1"]:.4f}, '
            f'S1_Y0={forecasted_distributions["S1_Y0"]:.4f}'
        )
    else:
        w_new = w_current

    reweighed_train_df = reweighing(train_df, sensitive_attribute, target_label, w_new)

    sample_weights = reweighed_train_df['weight'].values

    return sample_weights


def compute_batch_proportions(
    batch_df: pd.DataFrame,
    sensitive_attribute: str,
    target_label: str,
) -> dict[str, float]:
    n = len(batch_df)
    return {
        'S0_Y1': len(
            batch_df[
                (batch_df[sensitive_attribute] == 0) & (batch_df[target_label] == 1)
            ]
        )
        / n,
        'S0_Y0': len(
            batch_df[
                (batch_df[sensitive_attribute] == 0) & (batch_df[target_label] == 0)
            ]
        )
        / n,
        'S1_Y1': len(
            batch_df[
                (batch_df[sensitive_attribute] == 1) & (batch_df[target_label] == 1)
            ]
        )
        / n,
        'S1_Y0': len(
            batch_df[
                (batch_df[sensitive_attribute] == 1) & (batch_df[target_label] == 0)
            ]
        )
        / n,
    }


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


def ABCInML(
    df: pd.DataFrame,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    baseline_type: str,
    alpha: float,
    forecast_window: int = 3,
) -> None:

    batches = sorted(df[window_column].unique())

    historical_proportions: dict[str, list[float]] = {
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

    model = None

    print(f'(1) running with alpha: {alpha}')

    for i in range(len(batches)):
        print(f'(1.1) running for batch: {batches[i]}')

        current_batch_df = df[df[window_column] == batches[i]].copy()

        current_proportions = compute_batch_proportions(
            current_batch_df, sensitive_attribute, target_label
        )

        if i == 0:
            for key in historical_proportions:
                historical_proportions[key].append(current_proportions[key])

            if baseline_type in ['vanilla', 'static']:
                train_df = current_batch_df.drop(columns=[window_column])
                X_train = train_df.drop(columns=[target_label])
                y_train = train_df[target_label]

                sample_weights = None
                if baseline_type == 'static':
                    sample_weights = reweigh_df(
                        X_train,
                        y_train,
                        sensitive_attribute,
                        target_label,
                        baseline_type,
                    )

                print('(1.1.2) training logistic regression model')
                model = LogisticRegression(max_iter=150)
                model.fit(X_train, y_train, sample_weight=sample_weights)

            continue

        test_df = current_batch_df.drop(columns=[window_column])
        X_test = test_df.drop(columns=[target_label])
        y_test = test_df[target_label]

        if baseline_type in ['dynamic', 'abcinml']:
            train_df = df[df[window_column].isin(batches[0:i])].copy()
            train_df = train_df.drop(columns=[window_column])
            X_train = train_df.drop(columns=[target_label])
            y_train = train_df[target_label]

            forecasted_distributions = None
            if baseline_type == 'abcinml':
                forecasted_distributions = forecast_joint_distributions(
                    historical_proportions,
                    window=forecast_window,
                )

            sample_weights = reweigh_df(
                X_train,
                y_train,
                sensitive_attribute,
                target_label,
                baseline_type,
                alpha=alpha,
                forecasted_distributions=forecasted_distributions,
            )

            print('(1.1.2) training logistic regression model')
            model = LogisticRegression(max_iter=50)
            model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        sf_test = X_test[sensitive_attribute]

        print('(1.1.3) testing logistic regression model')
        acc, auc, sp, tpr, fpr = compute_snapshot_metrics(
            y_test, y_pred, y_pred_proba, sf_test
        )
        assessment_metrics.loc[len(assessment_metrics)] = [
            acc,
            auc,
            sp,
            tpr,
            fpr,
            batches[i],
            alpha,
        ]

        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])

    temporal_metrics = compute_temporal_metrics(assessment_metrics, alpha)

    print('\n')

    return assessment_metrics, temporal_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--baseline_type', type=str, default='abcinml')
    parser.add_argument('--path', type=str)
    parser.add_argument('--sensitive_attribute', type=str)
    parser.add_argument('--target_label', type=str)
    parser.add_argument('--window_column', type=str)

    args = parser.parse_args()

    alpha = args.alpha
    baseline_type = args.baseline_type
    path = args.path
    sensitive_attribute = args.sensitive_attribute
    target_label = args.target_label
    window_column = args.window_column

    if alpha == None:
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else:
        alphas = [alpha]

    return alphas, baseline_type, path, sensitive_attribute, target_label, window_column


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    alphas, baseline_type, path, sensitive_attribute, target_label, window_column  = parse_arguments()

    df = pre_process_df(
        df=pd.read_csv(path),
        sensitive_attribute=sensitive_attribute,
        sensitive_value=2,
        favored_value=1,
    )

    assessment_results_df = pd.DataFrame()
    temporal_results_df = pd.DataFrame()

    print(f'--- Running baseline: {baseline_type} ---\n')

    for alpha in alphas:

        assessment_metrics, temporal_metrics = ABCInML(
            df=df,
            window_column=window_column,
            sensitive_attribute=sensitive_attribute,
            target_label=target_label,
            baseline_type=baseline_type,
            alpha=alpha,
            forecast_window=3,
        )

        assessment_results_df = pd.concat([assessment_results_df, assessment_metrics])
        temporal_results_df = pd.concat([temporal_results_df, temporal_metrics])

    print('finished training.')


if __name__ == '__main__':
    main()
