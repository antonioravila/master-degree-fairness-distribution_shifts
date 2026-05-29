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
    forecast_window: int,
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
        absolute_rates
    ) = general_utils.initialize_variables(df, window_column)

    print(f'(1) running with alpha: {alpha}')

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

                print('(1.1.3) training logistic regression model')
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

            if X_train is not None:
                if baseline_type == 'abcinml':
                    forecasted_distributions = model_utils.forecast_joint_distributions(
                        historical_proportions,
                        window=forecast_window,
                    )
                    if forecasted_distributions is None:
                        for key in historical_proportions:
                            historical_proportions[key].append(current_proportions[key])
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

                print('(1.1.3) training logistic regression model')
                model = LogisticRegression(max_iter=_MAX_ITER)
                model.fit(X_train, y_train, sample_weight=sample_weights)

        if X_test is not None and i >= forecast_window:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            sf_test = X_test[sensitive_attribute]

            print('(1.1.4) testing logistic regression model')
            acc, auc, sp, tpr_diff, fpr_diff, tpr_S0, fpr_S0, tpr_S1, tpr_S1 = metrics_utils.compute_snapshot_metrics(
                y_test, y_pred, y_pred_proba, sf_test
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
                batch, alpha, tpr_S0, fpr_S0, tpr_S1, tpr_S1
            ]

            shap_record = metrics_utils.compute_shap_values(model, X_test)
            shap_record['batch'] = str(batch)
            shap_record['alpha'] = alpha
            shap_records.append(shap_record)

        for key in historical_proportions:
            historical_proportions[key].append(current_proportions[key])

    temporal_metrics = metrics_utils.compute_temporal_metrics(assessment_metrics, alpha)

    print('\n')

    return assessment_metrics, temporal_metrics, shap_records, absolute_rates


_ALL_BASELINES = ['vanilla', 'static', 'dynamic', 'abcinml']


def _run_baseline(df, baseline_type, alphas, sensitive_attribute, target_label, window_column):
    assessment_results_df = pd.DataFrame()
    temporal_results_df = pd.DataFrame()
    absolute_rates_df = pd.DataFrame()
    shap_records = []

    print(f'--- Running baseline: {baseline_type} ---\n')

    for alpha in alphas:
        assessment_metrics, temporal_metrics, alpha_shap_records, absolute_rates = ABCInML(
            df=df,
            window_column=window_column,
            sensitive_attribute=sensitive_attribute,
            target_label=target_label,
            baseline_type=baseline_type,
            alpha=alpha,
            forecast_window=10,
        )

        assessment_results_df = pd.concat([assessment_results_df, assessment_metrics])
        temporal_results_df = pd.concat([temporal_results_df, temporal_metrics])
        absolute_rates_df = pd.concat([absolute_rates_df, absolute_rates])
        shap_records.extend(alpha_shap_records)

    return assessment_results_df, temporal_results_df, shap_records, absolute_rates_df


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

    baselines = _ALL_BASELINES if baseline_type == 'all' else [baseline_type]

    all_assessment_df = pd.DataFrame()
    all_temporal_df = pd.DataFrame()
    all_absolute_df = pd.DataFrame()
    all_shap_records = []

    for bt in baselines:
        assessment_results_df, temporal_results_df, shap_records, absolute_rates_df = _run_baseline(
            df, bt, alphas, sensitive_attribute, target_label, window_column
        )
        assessment_results_df['baseline'] = bt
        temporal_results_df['baseline'] = bt
        absolute_rates_df['baseline'] = bt
        all_assessment_df = pd.concat([all_assessment_df, assessment_results_df])
        all_temporal_df = pd.concat([all_temporal_df, temporal_results_df])
        all_absolute_df = pd.concat([all_absolute_df, absolute_rates_df])
        all_shap_records.extend(shap_records)

    chart_label = 'all' if baseline_type == 'all' else baseline_type
    alpha_label = '_'.join(str(a) for a in alphas)
    import os
    subfolder = os.path.join(sensitive_attribute, alpha_label)
    metrics_utils.generate_charts(all_assessment_df, all_temporal_df, baseline_type=chart_label, save=True, subfolder=subfolder)
    metrics_utils.generate_shap_charts(all_shap_records, save=True, subfolder=subfolder)

    # Save consolidated results
    results_dir = os.path.join(os.path.dirname(__file__), 'results', sensitive_attribute, alpha_label)
    os.makedirs(results_dir, exist_ok=True)
    all_assessment_df.to_csv(os.path.join(results_dir, 'assessment_all_baselines.csv'), index=False)
    all_temporal_df.to_csv(os.path.join(results_dir, 'temporal_results_df.csv'), index=False)
    all_absolute_df.to_csv(os.path.join(results_dir, 'absolute_results_df.csv'), index=False)

    # Comparative charts (only when multiple baselines were run)
    if len(baselines) > 1:
        metrics_utils.generate_comparative_chart(all_assessment_df, moving_avg_window=5, save=True, subfolder=subfolder)
        metrics_utils.generate_fairness_comparison_chart(all_assessment_df, moving_avg_window=5, save=True, subfolder=subfolder)

    print('finished training.')


if __name__ == '__main__':
    main()
