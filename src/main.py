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
from statsmodels.tsa.ar_model import AutoReg


def get_next_ratio_forecast(history: list | np.ndarray) -> float:
    history = np.array(history) if isinstance(history, list) else history

    if len(history) == 0:
        return 0.5

    if len(history) == 1:
        return float(history[-1])

    if len(history) == 2:
        slope = history[1] - history[0]
        predicted_ratio = history[1] + slope
        return float(np.clip(predicted_ratio, 0.0, 1.0))

    model = AutoReg(history, lags=1, trend='c')
    model.fit()
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    predicted_ratio = np.clip(forecast, 0.0, 1.0)

    return float(predicted_ratio)


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
    predicted_ratio: float = None,
) -> np.array:
    n = df.shape[0]
    positve = df[df[target_label] == 1].shape[0]
    negative = df[df[target_label] == 0].shape[0]

    if predicted_ratio is None:
        deprived = (df[df[sensitive_attribute] == 0]).shape[0]
        favored = (df[df[sensitive_attribute] == 1]).shape[0]
    else:
        deprived = n * predicted_ratio
        favored = n * (1 - predicted_ratio)

    DP = df[(df[sensitive_attribute] == 0) & (df[target_label] == 1)].shape[0]
    DN = df[(df[sensitive_attribute] == 0) & (df[target_label] == 0)].shape[0]
    FP = df[(df[sensitive_attribute] == 1) & (df[target_label] == 1)].shape[0]
    FN = df[(df[sensitive_attribute] == 1) & (df[target_label] == 0)].shape[0]

    W_DP = (deprived * positve) / (n * DP)
    W_DN = (deprived * negative) / (n * DN)
    W_FP = (favored * positve) / (n * FP)
    W_FN = (favored * negative) / (n * FN)

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
    predicted_ratio: float = None,
) -> np.array:

    print('started reweighing process.')
    train_df = X_train.copy()
    train_df[target_label] = y_train

    w_past = weight_estimation(train_df, sensitive_attribute, target_label)

    if baseline_type == 'abcinml' and predicted_ratio is not None:
        w_future = weight_estimation(
            train_df, sensitive_attribute, target_label, predicted_ratio
        )
        w_final = (1 - alpha) * w_past + alpha * w_future
        print(
            f'Applied alpha blending: α={alpha}, predicted_ratio={predicted_ratio:.4f}'
        )
    else:
        w_final = w_past

    reweighed_train_df = reweighing(
        train_df, sensitive_attribute, target_label, w_final
    )

    sample_weights = reweighed_train_df['weight'].values

    print('finished reweighing process.')

    return sample_weights


def ABCInML(
    df: pd.DataFrame,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    baseline_type: str,
    alpha: float,
) -> None:

    df = pre_process_df(
        df=df,
        sensitive_attribute=sensitive_attribute,
        sensitive_value=2,
        favored_value=1,
    )

    i = 0
    historical_ratios = []

    for batch in df[window_column].unique():
        print(f'BATCH: {batch}')

        sample_weights = None
        batch_df = df[df[window_column] == batch].copy()

        current_ratio = len(batch_df[batch_df[sensitive_attribute] == 0]) / len(
            batch_df
        )

        predicted_ratio = None
        if baseline_type == 'abcinml' and i > 0:
            predicted_ratio = get_next_ratio_forecast(historical_ratios)
            print(f'Forecasted ratio for batch {i}: {predicted_ratio:.4f}')

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
                    predicted_ratio=predicted_ratio,
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

        historical_ratios.append(current_ratio)
        print(f'Actual ratio for batch {i}: {current_ratio:.4f}')

        i += 1


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    path = r'src\data\csv\acs_income_union.csv'
    df = pd.read_csv(path)
    ABCInML(
        df=df,
        window_column='Year',
        sensitive_attribute='SEX',
        target_label='PINCP',
        baseline_type='abcinml',
        alpha=0.20,
    )


if __name__ == '__main__':
    main()
