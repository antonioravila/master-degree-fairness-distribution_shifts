import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils import general_utils, metrics_utils, model_utils, data_utils


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
    sample_weights = None
    forecasted_distributions = None

    print(f'(1) running with alpha: {alpha}')

    for i in range(len(batches)):
        batch = batches[i]

        print(f'(1.1) running for batch: {batch}')

        current_batch_df = df[df[window_column] == batch].copy()

        current_proportions = model_utils.compute_batch_proportions(
            current_batch_df, sensitive_attribute, target_label
        )

        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])

        if baseline_type in ['vanilla', 'static']:
            X_train, y_train, X_test, y_test = model_utils.get_train_and_test(
                df=current_batch_df, 
                baseline_type=baseline_type, 
                window_column=window_column, 
                target_label=target_label, 
                batches=batches,
                i=i
            )
            
            if i == 0:
        
                if baseline_type == 'static':
                    sample_weights = model_utils.reweigh_df(
                        X_train,
                        y_train,
                        sensitive_attribute,
                        target_label,
                        baseline_type,
                    )

                print('(1.1.2) training logistic regression model')
                model = LogisticRegression(max_iter=150)
                model.fit(X_train, y_train, sample_weight=sample_weights)

        if baseline_type in ['dynamic', 'abcinml']:
            X_train, y_train, X_test, y_test = model_utils.get_train_and_test(
                df=df, 
                baseline_type=baseline_type, 
                window_column=window_column, 
                target_label=target_label, 
                batches=batches,
                i=i
            )

            if baseline_type == 'abcinml':
                forecasted_distributions = model_utils.forecast_joint_distributions(
                    historical_proportions,
                    window=forecast_window,
                )

            sample_weights = model_utils.reweigh_df(
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
        acc, auc, sp, tpr, fpr = metrics_utils.compute_snapshot_metrics(
            y_test, y_pred, y_pred_proba, sf_test
        )
        assessment_metrics.loc[len(assessment_metrics)] = [
            acc,
            auc,
            sp,
            tpr,
            fpr,
            batch,
            alpha,
        ]

    temporal_metrics = metrics_utils.compute_temporal_metrics(assessment_metrics, alpha)

    print('\n')

    return assessment_metrics, temporal_metrics


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    alphas, baseline_type, path, sensitive_attribute, target_label, window_column = (
        general_utils.parse_arguments()
    )

    df = data_utils.pre_process_df(
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

    metrics_utils.generate_charts()

    print('finished training.')


if __name__ == '__main__':
    main()
