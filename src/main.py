import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def run_tests(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_pred_proba: np.array,
    sensitive_features: str,
) -> None:
    print(f'ACC (Accuracy): {accuracy_score(y_test, y_pred)}')
    print(
        f'AUC (Area Under the ROC Curve): {roc_auc_score(y_test, y_pred_proba[:, 1])}'
    )
    print(
        f'S.P (Statistical Parity): {demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)}'
    )
    print(
        f'TPR (True Positive Rate): {true_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)}'
    )
    print(
        f'FPR (False Positive Rate): {false_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features)}'
    )


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
    # Forecasted joint cell values (these are proportions of total)
    p_S0_Y1 = forecasted['S0_Y1']  # P(S=0, Y=1)
    p_S0_Y0 = forecasted['S0_Y0']  # P(S=0, Y=0)
    p_S1_Y1 = forecasted['S1_Y1']  # P(S=1, Y=1)
    p_S1_Y0 = forecasted['S1_Y0']  # P(S=1, Y=0)

    # Derive marginals from the four joint cells
    n = p_S0_Y1 + p_S0_Y0 + p_S1_Y1 + p_S1_Y0  # total (sums to ~1.0 if proportions)

    deprived = p_S0_Y1 + p_S0_Y0   # P(S=0)
    favored = p_S1_Y1 + p_S1_Y0    # P(S=1)
    positive = p_S0_Y1 + p_S1_Y1   # P(Y=1)
    negative = p_S0_Y0 + p_S1_Y0   # P(Y=0)

    # Kamiran formula: W = P_expected / P_observed = [P(S) * P(Y)] / [n * P(S,Y)]
    W_DP = (deprived * positive) / (n * p_S0_Y1)   # W for S=0, Y=1
    W_DN = (deprived * negative) / (n * p_S0_Y0)   # W for S=0, Y=0
    W_FP = (favored * positive) / (n * p_S1_Y1)    # W for S=1, Y=1
    W_FN = (favored * negative) / (n * p_S1_Y0)    # W for S=1, Y=0

    return np.array([W_DP, W_DN, W_FP, W_FN])


def reweighing(
    df: pd.DataFrame, sensitive_attribute: str, target_label: str, weights: np.array
) -> pd.DataFrame:

    W_DP, W_DN, W_FP, W_FN = weights

    df_ = df.copy()
    df_['weight'] = np.ones(len(df_))

    # Update weights based on group/label combination
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

    print('started reweighing process.')
    train_df = X_train.copy()
    train_df[target_label] = y_train

    w_current = weight_estimation(train_df, sensitive_attribute, target_label)

    if baseline_type == 'abcinml' and forecasted_distributions is not None:
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

    reweighed_train_df = reweighing(
        train_df, sensitive_attribute, target_label, w_new
    )

    sample_weights = reweighed_train_df['weight'].values

    print('finished reweighing process.')

    return sample_weights


def compute_batch_proportions(
    batch_df: pd.DataFrame,
    sensitive_attribute: str,
    target_label: str,
) -> dict[str, float]:
    n = len(batch_df)
    return {
        'S0_Y1': len(batch_df[(batch_df[sensitive_attribute] == 0) & (batch_df[target_label] == 1)]) / n,
        'S0_Y0': len(batch_df[(batch_df[sensitive_attribute] == 0) & (batch_df[target_label] == 0)]) / n,
        'S1_Y1': len(batch_df[(batch_df[sensitive_attribute] == 1) & (batch_df[target_label] == 1)]) / n,
        'S1_Y0': len(batch_df[(batch_df[sensitive_attribute] == 1) & (batch_df[target_label] == 0)]) / n,
    }


def ABCInML(
    df: pd.DataFrame,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    baseline_type: str,
    alpha: float,
    forecast_window: int = 3,
) -> None:

    df = pre_process_df(
        df=df,
        sensitive_attribute=sensitive_attribute,
        sensitive_value=2,
        favored_value=1,
    )

    i = 0

    historical_proportions: dict[str, list[float]] = {
        'S0_Y1': [],
        'S0_Y0': [],
        'S1_Y1': [],
        'S1_Y0': [],
    }

    for batch in df[window_column].unique():
        print(f'BATCH: {batch}')

        sample_weights = None
        batch_df = df[df[window_column] == batch].copy()

        forecasted_distributions = None
        if baseline_type == 'abcinml' and i > 0:
            forecasted_distributions = forecast_joint_distributions(
                historical_proportions,
                window=forecast_window,
            )
            print(
                f'Forecasted distributions for batch {i}: '
                f'S0_Y1={forecasted_distributions["S0_Y1"]:.4f}, '
                f'S0_Y0={forecasted_distributions["S0_Y0"]:.4f}, '
                f'S1_Y1={forecasted_distributions["S1_Y1"]:.4f}, '
                f'S1_Y0={forecasted_distributions["S1_Y0"]:.4f}'
            )

        X = batch_df.drop(columns=[window_column])
        y = batch_df[target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if baseline_type in ['static', 'dynamic', 'abcinml']:
            if not (baseline_type == 'static' and i != 0):
                sample_weights = reweigh_df(
                    X_train,
                    y_train,
                    sensitive_attribute,
                    target_label,
                    baseline_type,
                    alpha=alpha,
                    forecasted_distributions=forecasted_distributions,
                )

        if (baseline_type in ['vanilla', 'static'] and i == 0) or (
            baseline_type in ['dynamic', 'abcinml']
        ):
            print('started training the model.')
            model = LogisticRegression(max_iter=150)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            print('finished trained the model.')

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        sf_test = X_test[sensitive_attribute]
        run_tests(y_test, y_pred, y_pred_proba, sf_test)

        # GAP 1 FIX: Record all 4 proportions for this batch
        current_proportions = compute_batch_proportions(
            batch_df, sensitive_attribute, target_label
        )
        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])

        print(f'Actual proportions for batch {i}: {current_proportions}')

        i += 1


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    path = r'src\data\csv\acs_income_TX.csv'
    df = pd.read_csv(path)
    ABCInML(
        df=df,
        window_column='Year',
        sensitive_attribute='SEX',
        target_label='PINCP',
        baseline_type='abcinml',
        alpha=0.20,
        forecast_window=3,
    )


if __name__ == '__main__':
    main()