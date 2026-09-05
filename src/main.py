import os
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
    forecast_window_batches: int,
    model_name: str,
    forecasting_technique: str,
    alpha_ema: float,
    fairness_correction: str,
    apr_iterations: int = 20,
    apr_eta: float = 1.0,
    apr_subgroup_alpha: float = 100.0,
    apr_decision_boundary: float = 0.5,
    apr_fairness_measure: str = 'equalized_odds',
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
            X_train, y_train, X_test, y_test = model_utils._get_train_and_test(
                df=current_batch_df,
                baseline_type=baseline_type,
                window_column=window_column,
                target_label=target_label,
                batches=batches,
                i=i,
            )

            if X_train is not None:
                print(f'(1.1.3) training {model_name} model')

                model, sample_weights = model_utils.fit_fair_model(
                    X_train,
                    y_train,
                    sensitive_attribute,
                    target_label,
                    model_name,
                    baseline_type,
                    fairness_correction,
                    _MAX_ITER,
                    alpha=alpha,
                    forecasted_distributions=forecasted_distributions,
                    apr_iterations=apr_iterations,
                    apr_eta=apr_eta,
                    apr_subgroup_alpha=apr_subgroup_alpha,
                    apr_decision_boundary=apr_decision_boundary,
                    apr_fairness_measure=apr_fairness_measure,
                )

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
                    if forecasting_technique == 'oracle':
                        forecasted_distributions = model_utils.forecast_joint_distributions_oracle(
                            df=df,
                            batches=batches,
                            i=i,
                            window_column=window_column,
                            sensitive_attribute=sensitive_attribute,
                            target_label=target_label,
                            historical_proportions=historical_proportions,
                            window=forecast_window_batches,
                            alpha_ema=alpha_ema,
                        )
                    else:
                        forecasted_distributions = model_utils.forecast_joint_distributions(
                            historical_proportions,
                            window=forecast_window_batches,
                            forecasting_technique=forecasting_technique,
                            alpha_ema=alpha_ema,
                        )
                    if forecasted_distributions is None:
                        for key in historical_proportions:
                            historical_proportions[key].append(current_proportions[key])
                        accumulated_train_df = pd.concat([accumulated_train_df, current_batch_df], ignore_index=True)
                        continue

                print(f'(1.1.3) training {model_name} model')
                model, sample_weights = model_utils.fit_fair_model(
                    X_train,
                    y_train,
                    sensitive_attribute,
                    target_label,
                    model_name,
                    baseline_type,
                    fairness_correction,
                    _MAX_ITER,
                    alpha=alpha,
                    forecasted_distributions=forecasted_distributions,
                    apr_iterations=apr_iterations,
                    apr_eta=apr_eta,
                    apr_subgroup_alpha=apr_subgroup_alpha,
                    apr_decision_boundary=apr_decision_boundary,
                    apr_fairness_measure=apr_fairness_measure,
                )

        should_test = X_test is not None and i >= forecast_window_batches

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


def _attach_metadata(dfs, model_name, forecast_window, forecasting_technique, batch_type, alpha_ema, fairness_correction, apr_fairness_measure):
    for df in dfs:
        df['model_name'] = model_name
        df['forecast_window'] = forecast_window
        df['forecasting_technique'] = forecasting_technique
        df['batch_type'] = batch_type
        df['alpha_ema'] = alpha_ema
        df['fairness_correction'] = fairness_correction
        # APR-only knob — blank for non-APR runs so the column is unambiguous.
        df['apr_fairness_measure'] = apr_fairness_measure if fairness_correction == 'adaptative_rw' else ''


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    _ALL_BASELINES = ['vanilla', 'static', 'dynamic', 'abcinml']
    _NON_ABC_BASELINES = ['vanilla', 'static', 'dynamic']

    alphas, baseline_type, sensitive_attribute, target_label, model_name, test_run, forecasting_technique, forecast_window, batch_type, alpha_ema, fairness_correction, apr_iterations, apr_eta, apr_subgroup_alpha, apr_decision_boundary, apr_fairness_measure = (
        general_utils.parse_arguments()
    )

    window_column = general_utils.get_window_column(batch_type)

    # forecast_window is given in months; convert to the number of batches so the
    # forecasting horizon and warm-up are the same real duration for any batch_type.
    forecast_window_batches = general_utils.effective_forecast_window(forecast_window, batch_type)

    df = data_utils.pre_process_df(
        sensitive_attribute,
        window_column,
    )

    if test_run:
        df = df.iloc[:100000].copy()

    baselines = _ALL_BASELINES if baseline_type == 'all' else [baseline_type]
    non_abc_to_run = [b for b in baselines if b in _NON_ABC_BASELINES]
    run_abcinml = 'abcinml' in baselines

    # --- Run vanilla/static/dynamic once — results are alpha-independent ---
    non_abc_assessment_df = pd.DataFrame()
    non_abc_temporal_df = pd.DataFrame()
    non_abc_absolute_df = pd.DataFrame()
    non_abc_shap_records = []

    if non_abc_to_run:
        non_abc_folder = metrics_utils.baseline_non_abc_folder_name(model_name, batch_type, fairness_correction)
        non_abc_results_dir = os.path.join('src', 'results', sensitive_attribute, non_abc_folder)
        non_abc_assessment_path = os.path.join(non_abc_results_dir, 'assessment_all_baselines.csv')
        non_abc_subfolder = os.path.join(sensitive_attribute, non_abc_folder)

        if os.path.exists(non_abc_assessment_path):
            print(f'--- Loading cached non-ABC results from {non_abc_results_dir} ---\n')
            non_abc_assessment_df = pd.read_csv(non_abc_assessment_path)
            non_abc_temporal_df = pd.read_csv(os.path.join(non_abc_results_dir, 'temporal_results_df.csv'))
            non_abc_absolute_df = pd.read_csv(os.path.join(non_abc_results_dir, 'absolute_results_df.csv'))
        else:
            for bt in non_abc_to_run:
                print(f'--- Running baseline: {bt} (alpha-independent) ---\n')

                assessment_metrics, temporal_metrics, shap_records_bt, absolute_rates = run_baseline(
                    df=df,
                    window_column=window_column,
                    sensitive_attribute=sensitive_attribute,
                    target_label=target_label,
                    baseline_type=bt,
                    alpha=0,
                    forecast_window_batches=forecast_window_batches,
                    model_name=model_name,
                    forecasting_technique=forecasting_technique,
                    alpha_ema=alpha_ema,
                    fairness_correction=fairness_correction,
                    apr_iterations=apr_iterations,
                    apr_eta=apr_eta,
                    apr_subgroup_alpha=apr_subgroup_alpha,
                    apr_decision_boundary=apr_decision_boundary,
                    apr_fairness_measure=apr_fairness_measure,
                )
                assessment_metrics['baseline'] = bt
                temporal_metrics['baseline'] = bt
                absolute_rates['baseline'] = bt
                _attach_metadata(
                    (assessment_metrics, temporal_metrics, absolute_rates),
                    model_name, forecast_window, forecasting_technique, batch_type, alpha_ema,
                    fairness_correction, apr_fairness_measure,
                )

                non_abc_assessment_df = pd.concat([non_abc_assessment_df, assessment_metrics], ignore_index=True)
                non_abc_temporal_df = pd.concat([non_abc_temporal_df, temporal_metrics], ignore_index=True)
                non_abc_absolute_df = pd.concat([non_abc_absolute_df, absolute_rates], ignore_index=True)
                non_abc_shap_records.extend(shap_records_bt)

            metrics_utils.save_baseline_non_abc_results(
                sensitive_attribute, non_abc_assessment_df, non_abc_temporal_df, non_abc_absolute_df,
                model_name, batch_type, fairness_correction,
            )
            metrics_utils.generate_all_charts(
                baseline_type='all' if len(non_abc_to_run) > 1 else non_abc_to_run[0],
                baselines=non_abc_to_run,
                alpha=0,
                sensitive_attribute=sensitive_attribute,
                all_assessment_df=non_abc_assessment_df,
                all_temporal_df=non_abc_temporal_df,
                all_absolute_df=non_abc_absolute_df,
                all_shap_records=non_abc_shap_records,
                model_name=model_name,
                forecast_window=forecast_window,
                forecasting_technique=forecasting_technique,
                batch_type=batch_type,
                alpha_ema=alpha_ema,
                fairness_correction=fairness_correction,
                subfolder_override=non_abc_subfolder,
            )

    # --- Run abcinml once per alpha ---
    if run_abcinml:
        for alpha in alphas:
            print(f'--- Running baseline: abcinml, alpha: {alpha} ---\n')

            assessment_metrics, temporal_metrics, shap_records_alpha, absolute_rates = run_baseline(
                df=df,
                window_column=window_column,
                sensitive_attribute=sensitive_attribute,
                target_label=target_label,
                baseline_type='abcinml',
                alpha=alpha,
                forecast_window_batches=forecast_window_batches,
                model_name=model_name,
                forecasting_technique=forecasting_technique,
                alpha_ema=alpha_ema,
                fairness_correction=fairness_correction,
                apr_iterations=apr_iterations,
                apr_eta=apr_eta,
                apr_subgroup_alpha=apr_subgroup_alpha,
                apr_decision_boundary=apr_decision_boundary,
                apr_fairness_measure=apr_fairness_measure,
            )
            assessment_metrics['baseline'] = 'abcinml'
            temporal_metrics['baseline'] = 'abcinml'
            absolute_rates['baseline'] = 'abcinml'
            _attach_metadata(
                (assessment_metrics, temporal_metrics, absolute_rates),
                model_name, forecast_window, forecasting_technique, batch_type, alpha_ema,
                fairness_correction, apr_fairness_measure,
            )

            metrics_utils.save_results(
                sensitive_attribute,
                assessment_metrics,
                temporal_metrics,
                absolute_rates,
                model_name, alpha, forecast_window, forecasting_technique, batch_type, alpha_ema,
                fairness_correction,
            )

            # Combine with non-abc results so comparative charts include all baselines
            combined_assessment = pd.concat([non_abc_assessment_df, assessment_metrics], ignore_index=True)
            combined_temporal = pd.concat([non_abc_temporal_df, temporal_metrics], ignore_index=True)
            combined_absolute = pd.concat([non_abc_absolute_df, absolute_rates], ignore_index=True)
            combined_shap = non_abc_shap_records + shap_records_alpha

            metrics_utils.generate_all_charts(
                baseline_type=baseline_type,
                baselines=non_abc_to_run + ['abcinml'],
                alpha=alpha,
                sensitive_attribute=sensitive_attribute,
                all_assessment_df=combined_assessment,
                all_temporal_df=combined_temporal,
                all_absolute_df=combined_absolute,
                all_shap_records=combined_shap,
                model_name=model_name,
                forecast_window=forecast_window,
                forecasting_technique=forecasting_technique,
                batch_type=batch_type,
                alpha_ema=alpha_ema,
                fairness_correction=fairness_correction,
            )

    print('finished training.')


if __name__ == '__main__':
    main()
