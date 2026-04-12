import pandas as pd
import numpy as np
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

    reweighed_train_df = reweighing(train_df, sensitive_attribute, target_label, w_new)

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


def compute_temporal_metrics(assessment_metrics: pd.DataFrame) -> pd.DataFrame:
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

    df = pre_process_df(
        df=df,
        sensitive_attribute=sensitive_attribute,
        sensitive_value=2,
        favored_value=1,
    )

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
        }
    )

    model = None

    for i in range(len(batches)):
        print(f'BATCH: {batches[i]}')

        current_batch_df = df[df[window_column] == batches[i]].copy()

        current_proportions = compute_batch_proportions(
            current_batch_df, sensitive_attribute, target_label
        )

        if i == 0:
            for key in historical_proportions:
                historical_proportions[key].append(current_proportions[key])
            print(f'Actual proportions for batch {i}: {current_proportions}')

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

                model = LogisticRegression(max_iter=150)
                model.fit(X_train, y_train, sample_weight=sample_weights)
                print('Trained initial model on batch 0.')

            continue

        train_batches = batches[0:i]
        train_df = df[df[window_column].isin(train_batches)].copy()
        train_df = train_df.drop(columns=[window_column])
        X_train = train_df.drop(columns=[target_label])
        y_train = train_df[target_label]

        test_df = current_batch_df.drop(columns=[window_column])
        X_test = test_df.drop(columns=[target_label])
        y_test = test_df[target_label]

        if baseline_type in ['dynamic', 'abcinml']:
            forecasted_distributions = None
            if baseline_type == 'abcinml':
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

            sample_weights = reweigh_df(
                X_train,
                y_train,
                sensitive_attribute,
                target_label,
                baseline_type,
                alpha=alpha,
                forecasted_distributions=forecasted_distributions,
            )

            model = LogisticRegression(max_iter=150)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            print(f'Trained model on batches {train_batches}.')

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        sf_test = X_test[sensitive_attribute]

        print(f'Testing on batch {batches[i]}:')
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
        ]

        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])
        print(f'Actual proportions for batch {i}: {current_proportions}')

    temporal_metrics = compute_temporal_metrics(assessment_metrics)
    print(assessment_metrics)
    print(temporal_metrics)

def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    path = r'src\data\ACSIncome\acs_income_TX.csv'
    df = pd.read_csv(path)
    ABCInML(
        df=df,
        window_column='Year',
        sensitive_attribute='SEX',
        target_label='PINCP',
        baseline_type='vanilla',
        alpha=0.20,
        forecast_window=3,
    )


if __name__ == '__main__':
    main()
