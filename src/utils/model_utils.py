import pandas as pd
import numpy as np


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


def forecast_joint_distributions(
    historical_proportions: dict[str, list[float]],
    window: int,
) -> dict[str, float] | None:
    first_history = next(iter(historical_proportions.values()))
    if len(first_history) < window:
        return None

    forecasted = {}
    for key, history in historical_proportions.items():
        series = pd.Series(history)
        rolling_avg = series.rolling(window=window, min_periods=window).mean()
        forecasted[key] = float(rolling_avg.iloc[-1])

    return forecasted


def weight_estimation(
    df: pd.DataFrame,
    sensitive_attribute: str,
    target_label: str,
) -> np.array:

    n = df.shape[0]
    positve = df[df[target_label] == 1].shape[0]
    negative = df[df[target_label] == 0].shape[0]

    deprived = (df[df[sensitive_attribute] == 1]).shape[0]
    favored = (df[df[sensitive_attribute] == 0]).shape[0]

    DP = df[(df[sensitive_attribute] == 1) & (df[target_label] == 1)].shape[0]
    DN = df[(df[sensitive_attribute] == 1) & (df[target_label] == 0)].shape[0]
    FP = df[(df[sensitive_attribute] == 0) & (df[target_label] == 1)].shape[0]
    FN = df[(df[sensitive_attribute] == 0) & (df[target_label] == 0)].shape[0]

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

    deprived = p_S1_Y1 + p_S1_Y0
    favored = p_S0_Y1 + p_S0_Y0
    positive = p_S0_Y1 + p_S1_Y1
    negative = p_S0_Y0 + p_S1_Y0

    W_DP = (deprived * positive) / (n * p_S1_Y1)
    W_DN = (deprived * negative) / (n * p_S1_Y0)
    W_FP = (favored * positive) / (n * p_S0_Y1)
    W_FN = (favored * negative) / (n * p_S0_Y0)

    return np.array([W_DP, W_DN, W_FP, W_FN])


def reweighing(
    df: pd.DataFrame, sensitive_attribute: str, target_label: str, weights: np.array
) -> pd.DataFrame:

    W_DP, W_DN, W_FP, W_FN = weights

    df_ = df.copy()
    df_['weight'] = np.ones(len(df_))

    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 1) & (df_[target_label] == 1), W_DP, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 1) & (df_[target_label] == 0), W_DN, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 0) & (df_[target_label] == 1), W_FP, df_['weight']
    )
    df_['weight'] = np.where(
        (df_[sensitive_attribute] == 0) & (df_[target_label] == 0), W_FN, df_['weight']
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
        print('(1.1.2) forecasting distributions')
        w_future = weight_estimation_anticipated(forecasted_distributions)
        w_new = alpha * w_current + (1 - alpha) * w_future
    else:
        w_new = w_current

    reweighed_train_df = reweighing(train_df, sensitive_attribute, target_label, w_new)

    sample_weights = reweighed_train_df['weight'].values

    return sample_weights


def get_train_and_test(
    df: pd.DataFrame,
    baseline_type: str,
    window_column: str,
    target_label: str,
    batches: pd.Series,
    i: int,
):
    if baseline_type in ['dynamic', 'abcinml']:
        prior_batches = batches[0:i]
        if len(prior_batches) == 0:
            return None, None, None, None

        train_df = df[df[window_column].isin(prior_batches)]
        test_df = df[df[window_column] == batches[i]]

        X_train = train_df.drop(columns=[target_label, window_column])
        y_train = train_df[target_label]
        X_test = test_df.drop(columns=[target_label, window_column])
        y_test = test_df[target_label]

        return X_train, y_train, X_test, y_test

    # vanilla, static: batch 0 → only train; batch i≥1 → only test
    if i == 0:
        X_train = df.drop(columns=[target_label, window_column])
        y_train = df[target_label]
        return X_train, y_train, None, None

    X_test = df.drop(columns=[target_label, window_column])
    y_test = df[target_label]
    return None, None, X_test, y_test