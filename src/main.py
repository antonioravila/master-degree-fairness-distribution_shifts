import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils import general_utils, metrics_utils, model_utils, data_utils


def run_baseline(
    df: pd.DataFrame,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    baseline_type: str,
    alpha: float,
    forecast_window: int,
    model_name: str,
    forecasting_technique: str,
) -> None:

    _MAX_ITER = 1000

    (
        batches,
        historical_proportions,
        assessment_metrics,
        model,
        sample_weights,
        forecasted_distributions,
        shap_records,
        absolute_rates,
    ) = general_utils.initialize_variables(df, window_column)

    print(f'(1) running with alpha: {alpha}')

    accumulated_train_df = pd.DataFrame()

    for i in range(len(batches)):
        batch = batches[i]

        print(f'(1.1) running for batch: {batch}')

        current_batch_df = df[df[window_column] == batch].copy()

        current_proportions = model_utils.compute_batch_proportions(
            current_batch_df, sensitive_attribute, target_label
        )

        if baseline_type in ['vanilla', 'static']:
            X_train, y_train, X_test, y_test = model_utils.get_train_and_test(
                df=current_batch_df,
                baseline_type=baseline_type,
                window_column=window_column,
                target_label=target_label,
                batches=batches,
                i=i,
            )

            if X_train is not None:
                if baseline_type == 'static':
                    sample_weights = model_utils.reweigh_df(
                        X_train,
                        y_train,
                        sensitive_attribute,
                        target_label,
                        baseline_type,
                    )

                print(f'(1.1.3) training {model_name} model')

                model = model_utils.initialize_model(model_name, MAX_ITER=_MAX_ITER)

                model.fit(X_train, y_train, sample_weight=sample_weights)

        if baseline_type in ['dynamic', 'abcinml']:
            if i == 0:
                X_train, y_train, X_test, y_test = None, None, None, None
            else:
                X_train = accumulated_train_df.drop(columns=[target_label, window_column])
                y_train = accumulated_train_df[target_label]
                X_test = current_batch_df.drop(columns=[target_label, window_column])
                y_test = current_batch_df[target_label]

            if X_train is not None:
                if baseline_type == 'abcinml':
                    forecasted_distributions = model_utils.forecast_joint_distributions(
                        historical_proportions,
                        window=forecast_window,
                        forecasting_technique=forecasting_technique,
                    )
                    if forecasted_distributions is None:
                        for key in historical_proportions:
                            historical_proportions[key].append(current_proportions[key])
                        accumulated_train_df = pd.concat([accumulated_train_df, current_batch_df], ignore_index=True)
                        continue

                sample_weights = model_utils.reweigh_df(
                    X_train,
                    y_train,
                    sensitive_attribute,
                    target_label,
                    baseline_type,
                    alpha=alpha,
                    forecasted_distributions=forecasted_distributions,
                )

                print(f'(1.1.3) training {model_name} model')
                model = model_utils.initialize_model(model_name, MAX_ITER=_MAX_ITER)
                model.fit(X_train, y_train, sample_weight=sample_weights)

        should_test = X_test is not None and i >= forecast_window

        if should_test:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            sf_test = X_test[sensitive_attribute]

            print(f'(1.1.4) testing {model_name} model')
            acc, auc, sp, tpr_diff, fpr_diff, tpr_S0, fpr_S0, tpr_S1, fpr_S1 = (
                metrics_utils.compute_snapshot_metrics(
                    y_test, y_pred, y_pred_proba, sf_test
                )
            )

            assessment_metrics.loc[len(assessment_metrics)] = [
                acc,
                auc,
                sp,
                tpr_diff,
                fpr_diff,
                batch,
                alpha,
            ]

            absolute_rates.loc[len(absolute_rates)] = [
                batch,
                alpha,
                tpr_S0,
                fpr_S0,
                tpr_S1,
                fpr_S1,
            ]

            shap_record = metrics_utils.compute_shap_values(model_name, model, X_test)
            shap_record['batch'] = str(batch)
            shap_record['alpha'] = alpha
            shap_records.append(shap_record)

        accumulated_train_df = pd.concat([accumulated_train_df, current_batch_df], ignore_index=True)

        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])

    temporal_metrics = metrics_utils.compute_temporal_metrics(assessment_metrics, alpha)

    print('\n')

    return assessment_metrics, temporal_metrics, shap_records, absolute_rates


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    _ALL_BASELINES = ['vanilla', 'static', 'dynamic', 'abcinml']

    alphas, baseline_type, sensitive_attribute, target_label, model_name, test_run, forecasting_technique, forecast_window, batch_type = (
        general_utils.parse_arguments()
    )

    window_column = general_utils.get_window_column(batch_type)

    df = data_utils.pre_process_df(
        sensitive_attribute,
        window_column,
    )
    
    if test_run:
        df = df.iloc[:100000].copy()    

    baselines = _ALL_BASELINES if baseline_type == 'all' else [baseline_type]

    for alpha in alphas:
        all_assessment_df = pd.DataFrame()
        all_temporal_df = pd.DataFrame()
        all_absolute_df = pd.DataFrame()
        all_shap_records = []

        for bt in baselines:
            print(f'--- Running baseline: {bt}, alpha: {alpha} ---\n')

            assessment_metrics, temporal_metrics, alpha_shap_records, absolute_rates = (
                run_baseline(
                    df=df,
                    window_column=window_column,
                    sensitive_attribute=sensitive_attribute,
                    target_label=target_label,
                    baseline_type=bt,
                    alpha=alpha,
                    forecast_window=forecast_window,
                    model_name=model_name,
                    forecasting_technique=forecasting_technique,
                )
            )

            assessment_metrics['baseline'] = bt
            temporal_metrics['baseline'] = bt
            absolute_rates['baseline'] = bt

            for metrics_df in (assessment_metrics, temporal_metrics, absolute_rates):
                metrics_df['model_name'] = model_name
                metrics_df['forecast_window'] = forecast_window
                metrics_df['forecasting_technique'] = forecasting_technique
                metrics_df['batch_type'] = batch_type

            all_assessment_df = pd.concat([all_assessment_df, assessment_metrics], ignore_index=True)
            all_temporal_df = pd.concat([all_temporal_df, temporal_metrics], ignore_index=True)
            all_absolute_df = pd.concat([all_absolute_df, absolute_rates], ignore_index=True)
            all_shap_records.extend(alpha_shap_records)

        metrics_utils.generate_all_charts(
            baseline_type,
            baselines,
            alpha,
            sensitive_attribute,
            all_assessment_df,
            all_temporal_df,
            all_absolute_df,
            all_shap_records,
            model_name,
            forecast_window,
            forecasting_technique,
            batch_type,
        )

    print('finished training.')


if __name__ == '__main__':
    main()
