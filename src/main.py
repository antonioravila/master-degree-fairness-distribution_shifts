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
    
    _MAX_ITER = 150

    (
        batches,
        historical_proportions,
        assessment_metrics,
        model,
        sample_weights,
        forecasted_distributions,
    ) = general_utils.initialize_variables(df, window_column)

    shap_records = []


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
                i=i,
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
                model = LogisticRegression(max_iter=_MAX_ITER)
                model.fit(X_train, y_train, sample_weight=sample_weights)

        if baseline_type in ['dynamic', 'abcinml']:
            X_train, y_train, X_test, y_test = model_utils.get_train_and_test(
                df=df,
                baseline_type=baseline_type,
                window_column=window_column,
                target_label=target_label,
                batches=batches,
                i=i,
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
            model = LogisticRegression(max_iter=_MAX_ITER)
            model.fit(X_train, y_train, sample_weight=sample_weights)

        if i != 0:
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

            shap_record = metrics_utils.compute_shap_values(model, X_test)
            shap_record['batch'] = str(batch)
            shap_record['alpha'] = alpha
            shap_records.append(shap_record)

    temporal_metrics = metrics_utils.compute_temporal_metrics(assessment_metrics, alpha)

    print('\n')

    return assessment_metrics, temporal_metrics, shap_records


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    alphas, baseline_type, sensitive_attribute, target_label, window_column = (
        general_utils.parse_arguments()
    )

    df = data_utils.pre_process_df(
        sensitive_attribute=sensitive_attribute,
        sensitive_value=2,
        favored_value=1,
    )

    assessment_results_df = pd.DataFrame()
    temporal_results_df = pd.DataFrame()
    shap_records = []

    print(f'--- Running baseline: {baseline_type} ---\n')

    for alpha in alphas:
        assessment_metrics, temporal_metrics, alpha_shap_records = ABCInML(
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
        shap_records.extend(alpha_shap_records)

    metrics_utils.generate_charts(assessment_results_df, temporal_results_df, baseline_type=baseline_type, save=True)
    metrics_utils.generate_shap_charts(shap_records, save=True)

    print('finished training.')


if __name__ == '__main__':
    main()
