import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

def initialize_model(model_name, MAX_ITER):
    models_mapping = {
        'logistic_regression': LogisticRegression(max_iter=MAX_ITER),
        'xgboost': xgb.XGBClassifier(device='cuda', tree_method='hist')
    }

    model = models_mapping.get(model_name)
    return model

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
    forecasting_technique: str,
    alpha_ema: float
) -> dict[str, float] | None:
    first_history = next(iter(historical_proportions.values()))
    if len(first_history) < window:
        return None

    forecasted = {}
    for key, history in historical_proportions.items():
        series = pd.Series(history)
        if forecasting_technique == 'ema':
            avg = series.ewm(alpha=alpha_ema, min_periods=window).mean()
        elif forecasting_technique == 'sma':
            avg = series.rolling(window=window, min_periods=window).mean()
        forecasted[key] = float(avg.iloc[-1])

    return forecasted


def forecast_joint_distributions_oracle(
    df: pd.DataFrame,
    batches: list,
    i: int,
    window_column: str,
    sensitive_attribute: str,
    target_label: str,
    historical_proportions: dict[str, list[float]],
    window: int,
    alpha_ema: float,
) -> dict[str, float] | None:
    """Oracle forecaster for upper-bound diagnostic analysis.

    Instead of estimating future joint distributions from historical trends,
    reads the actual next batch's proportions directly.

    Oracle is a diagnostic tool only. It requires knowledge of the future and
    cannot be used in production. Use only for upper-bound analysis to determine
    whether forecasting quality is the bottleneck in ABCInML, or whether the
    reweighing mechanism itself is the limiting factor.

    Falls back to SMA when i+1 >= len(batches) (no next batch available).
    """
    if i + 1 < len(batches):
        next_batch_df = df[df[window_column] == batches[i + 1]].copy()
        return compute_batch_proportions(next_batch_df, sensitive_attribute, target_label)
    return forecast_joint_distributions(
        historical_proportions, window=window, forecasting_technique='sma', alpha_ema=alpha_ema
    )


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

    def _safe_weight(numerator: float, count: int) -> float:
        if count == 0:
            return 1.0
        return numerator / (n * count)

    W_DP = _safe_weight(deprived * positve, DP)
    W_DN = _safe_weight(deprived * negative, DN)
    W_FP = _safe_weight(favored * positve, FP)
    W_FN = _safe_weight(favored * negative, FN)

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


def compute_margins(model, X: pd.DataFrame, decision_boundary: float = 0.5) -> np.array:
    """Margin between each prediction and the decision boundary (Eq. 19, Hu et al.).

    ``phi_i = |h_theta(x_i) - d|`` where ``h_theta(x_i)`` is the model's predicted
    probability for the positive class and ``d`` is the decision boundary value.
    Samples with a small margin sit close to the boundary and are prioritised
    (higher weight) by the per-sample update in Eq. 18.
    """
    proba = model.predict_proba(X)[:, 1]
    return np.abs(proba - decision_boundary)


def update_apr_subgroup_weights(
    prev_W: dict,
    y_pred: np.array,
    sensitive_attribute_values: np.array,
    y_true_values: np.array,
    fairness_measure: str,
    subgroup_alpha: float,
    alpha: float = None,
    forecasted_distributions: dict[str, float] = None,
) -> dict:
    """Multiplicative subgroup-weight update (Eq. 17 / Algorithms 1-3, Hu et al.).

    ``prev_W`` maps ``(y, a) -> W_{y,a}^(t-1)`` for the four (label, sensitive)
    subgroups; the returned dict holds ``W_{y,a}^(t)``.

    - ``demographic_parity`` (Algorithm 1, Eq. 17): numerator uses the raw
      predicted-label counts ``|{i: y_pred_i = y}| * |{i: a_i = a}|`` and the
      denominator scale is ``m``.
    - ``equalized_odds`` (Algorithm 2): counts are conditioned on a correct
      prediction, ``|{i: y_pred_i = y, y_i = y}| * |{i: y_i = y, a_i = a}|`` with
      denominator scale ``m_{y,*} = |{i: y_i = y}|``.
    - ``equal_opportunity`` (Algorithm 3): only the ``y = 1`` subgroups are
      updated (Algorithm 2 form restricted to ``y = 1``); the ``y = 0`` subgroups
      keep their initial ``W_{0,a}^(0) = 1``.

    ABCInML forecasting hook (used only when ``baseline_type == 'abcinml'``, i.e.
    both ``alpha`` and ``forecasted_distributions`` are given): the *numerator*
    count product is blended with an expected-count version derived from
    ``forecasted_distributions`` (the same dict consumed by
    :func:`weight_estimation_anticipated`), following the CLI ``alpha`` blend
    ``num = alpha * num_current + (1 - alpha) * num_forecast``. The
    ``+ subgroup_alpha`` smoothing and the denominator stay based on the current
    batch only, and Eq. 18 (per-sample margin term) is left untouched by the
    forecast, because ``forecasted_distributions`` only carries subgroup-level
    aggregates.
    """
    m = len(y_pred)
    W = dict(prev_W)

    if fairness_measure == 'equal_opportunity':
        pairs = [(1, 0), (1, 1)]
    else:
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    use_forecast = alpha is not None and forecasted_distributions is not None

    for y, a in pairs:
        a_mask = sensitive_attribute_values == a

        if fairness_measure == 'demographic_parity':
            pred_y_mask = y_pred == y
            count_f1 = float(np.sum(pred_y_mask))                 # |{y_pred = y}|
            count_f2 = float(np.sum(a_mask))                      # |{a = a}|
            count_denom = float(np.sum(pred_y_mask & a_mask))     # |{y_pred = y, a = a}|
            m_scale = float(m)
        else:  # equalized_odds / equal_opportunity
            true_y_mask = y_true_values == y
            correct_y_mask = (y_pred == y) & true_y_mask
            count_f1 = float(np.sum(correct_y_mask))              # |{y_pred = y, y = y}|
            count_f2 = float(np.sum(true_y_mask & a_mask))        # |{y = y, a = a}|
            count_denom = float(np.sum(correct_y_mask & a_mask))  # |{y_pred = y, y = y, a = a}|
            m_scale = float(np.sum(true_y_mask))                  # m_{y,*}

        numerator_current = count_f1 * count_f2

        if use_forecast:
            p_a_fc = (
                forecasted_distributions[f'S{a}_Y0']
                + forecasted_distributions[f'S{a}_Y1']
            )
            p_y_fc = (
                forecasted_distributions[f'S0_Y{y}']
                + forecasted_distributions[f'S1_Y{y}']
            )
            if fairness_measure == 'demographic_parity':
                f1_fc = m * p_y_fc                                # expected |{y_pred = y}|
                f2_fc = m * p_a_fc                                # expected |{a = a}|
            else:
                f1_fc = m * p_y_fc                                # expected |{y_pred = y, y = y}|
                f2_fc = m * forecasted_distributions[f'S{a}_Y{y}']  # expected |{y = y, a = a}|
            numerator_forecast = f1_fc * f2_fc
            numerator = alpha * numerator_current + (1.0 - alpha) * numerator_forecast
        else:
            numerator = numerator_current

        denominator = m_scale * count_denom + subgroup_alpha
        if denominator <= 0.0:
            # no samples for this subgroup this batch — leave the weight unchanged
            continue
        W[(y, a)] = prev_W[(y, a)] * (numerator + subgroup_alpha) / denominator

    return W


def update_apr_sample_weights(
    W: dict,
    p_ya: dict,
    margins: np.array,
    group_labels: tuple,
    eta: float,
    fairness_measure: str = 'equalized_odds',
) -> np.array:
    """Per-sample weight update (Eq. 18 / Algorithm 3 line 9, Hu et al.).

    ``w_i = (W_{y,a} / sum_{y',a'} W_{y',a'})
            * (p_{y,a} * exp(-eta * phi_i) / sum_{j in G_{y,a}} exp(-eta * phi_j))``

    for ``z_i`` in the *fixed* subgroup ``G_{y,a}`` — partitioned on the true label
    ``y`` and sensitive attribute ``a`` (Section 3.2), never re-derived from
    predictions. ``margins`` are the Eq. 19 values ``phi_i``; smaller margins
    (closer to the boundary) receive exponentially more weight.

    For ``equal_opportunity`` (Algorithm 3) the ``y = 0`` subgroups instead get
    the uniform weight ``1 / (m * sum W)``.

    ``group_labels`` is ``(y_true_values, sensitive_attribute_values)`` as aligned
    integer arrays; ``p_ya`` maps ``(y, a) -> p_{y,a}`` (fixed, from Section 3.2).

    The raw Eq. 18 weights are finally rescaled to mean 1 (``sum == m``). Hu et al.
    train with plain SGD, where only the *relative* per-sample weights matter; the
    sklearn / XGBoost estimators used here apply regularisation to the weighted
    loss, so the absolute scale is normalised to match the rest of the pipeline
    (Kamiran & Calders weights in :func:`reweighing` are O(1)). Relative
    priorities from Eq. 17-18 are preserved exactly.
    """
    y_true_values, sensitive_attribute_values = group_labels
    m = len(margins)
    w = np.zeros(m, dtype=float)

    W_sum = float(sum(W.values()))
    if W_sum <= 0.0:
        W_sum = 1.0

    for y in (0, 1):
        for a in (0, 1):
            mask = (y_true_values == y) & (sensitive_attribute_values == a)
            if not np.any(mask):
                continue

            if fairness_measure == 'equal_opportunity' and y == 0:
                w[mask] = 1.0 / (m * W_sum)
                continue

            phi = margins[mask]
            soft = np.exp(-eta * phi)
            soft_sum = float(soft.sum())
            if soft_sum <= 0.0:
                soft = np.ones_like(phi)
                soft_sum = float(len(phi))

            subgroup_share = W[(y, a)] / W_sum
            w[mask] = subgroup_share * (p_ya[(y, a)] * soft / soft_sum)

    total = float(w.sum())
    if total <= 0.0:
        return np.full(m, 1.0, dtype=float)
    return w * (m / total)


def adaptive_priority_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attribute: str,
    target_label: str,
    model_name: str,
    fairness_measure: str = 'equalized_odds',
    T: int = 20,
    eta: float = 1.0,
    subgroup_alpha: float = 100.0,
    decision_boundary: float = 0.5,
    max_iter: int = 1000,
    alpha: float = None,
    forecasted_distributions: dict[str, float] = None,
):
    """Adaptive Priority Reweighing (Hu et al., 2023/2024; arXiv:2309.08375).

    An *in-processing* fair-training procedure: unlike Kamiran & Calders
    Reweighing (weights computed once from the true labels — see
    :func:`weight_estimation`), APR alternates between fitting the classifier and
    re-deriving sample weights from the *current* model's predictions and margins,
    so it owns its own training loop rather than handing weights back to a single
    external ``fit`` (Algorithms 1-3).

    Steps per iteration ``t`` (Algorithm 1):

    1. Fit ``H`` on ``(X, y)`` with the current weights ``w^(t-1)``
       (``w^(0) = 1/m``, ``W_{y,a}^(0) = 1`` — line 3).
    2. Predicted labels ``y_pred^(t)`` and margins ``phi_i^(t)`` (Eq. 19).
    3. Subgroup weights ``W_{y,a}^(t)`` via :func:`update_apr_subgroup_weights`
       (Eq. 17 / Alg. 2 / Alg. 3).
    4. Per-sample weights ``w_i^(t)`` via :func:`update_apr_sample_weights`
       (Eq. 18), keyed on the fixed ``(y_true, a)`` partition and its fixed
       proportions ``p_{y,a} = m_{y,a} / m`` (Section 3.2, Eq. 14-16).

    ``fairness_measure`` selects Algorithm 1 (``demographic_parity``),
    Algorithm 2 (``equalized_odds``) or Algorithm 3 (``equal_opportunity``).

    ABCInML integration: when ``alpha`` and ``forecasted_distributions`` are
    supplied (only for ``baseline_type == 'abcinml'``), the Eq. 17 numerator is
    blended with the forecasted joint distribution — see
    :func:`update_apr_subgroup_weights`. Eq. 18 stays present-only.

    Returns ``(fitted_model, final_sample_weights)``.
    """
    X = X_train.reset_index(drop=True)
    y_true = np.asarray(y_train).astype(int)
    sensitive_values = np.asarray(X[sensitive_attribute]).astype(int)
    m = len(y_true)

    # Fixed subgroup proportions p_{y,a} = m_{y,a} / m (Section 3.2, Hu et al.).
    p_ya = {
        (y, a): float(np.sum((y_true == y) & (sensitive_values == a))) / m
        for y in (0, 1)
        for a in (0, 1)
    }

    # Initialisation (Algorithm 1, line 3): W_{y,a}^(0) = 1, w_i^(0) = 1/m —
    # rescaled to mean 1 here (uniform), for the scale reason documented in
    # update_apr_sample_weights.
    W = {(y, a): 1.0 for y in (0, 1) for a in (0, 1)}
    sample_weights = np.ones(m, dtype=float)

    fc_note = ' + ABCInML forecast blend' if (
        alpha is not None and forecasted_distributions is not None
    ) else ''
    print(
        f'(1.1.1) adaptive priority reweighing ({fairness_measure}, T={T}, '
        f'eta={eta}, subgroup_alpha={subgroup_alpha}){fc_note}'
    )

    model = initialize_model(model_name, MAX_ITER=max_iter)
    model.fit(X, y_true, sample_weight=sample_weights)

    for _ in tqdm(range(T)):
        y_pred = model.predict(X).astype(int)                     # y_pred^(t-1)
        margins = compute_margins(model, X, decision_boundary)    # phi_i (Eq. 19)

        W = update_apr_subgroup_weights(                          # Eq. 17 / Alg. 2 / Alg. 3
            W,
            y_pred,
            sensitive_values,
            y_true,
            fairness_measure,
            subgroup_alpha,
            alpha=alpha,
            forecasted_distributions=forecasted_distributions,
        )
        sample_weights = update_apr_sample_weights(               # Eq. 18
            W,
            p_ya,
            margins,
            (y_true, sensitive_values),
            eta,
            fairness_measure,
        )

        model = initialize_model(model_name, MAX_ITER=max_iter)
        model.fit(X, y_true, sample_weight=sample_weights)

    return model, sample_weights


def fit_fair_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attribute: str,
    target_label: str,
    model_name: str,
    baseline_type: str,
    fairness_correction: str,
    max_iter: int,
    alpha: float = 0.0,
    forecasted_distributions: dict[str, float] = None,
    apr_iterations: int = 20,
    apr_eta: float = 1.0,
    apr_subgroup_alpha: float = 100.0,
    apr_decision_boundary: float = 0.5,
    apr_fairness_measure: str = 'equalized_odds',
):
    """Train the (optionally fairness-corrected) classifier for one batch.

    Single dispatch point shared by every baseline in ``main.run_baseline`` so the
    reweigh-then-fit logic is not duplicated across the ``vanilla/static`` and
    ``dynamic/abcinml`` blocks:

    - ``vanilla``: plain fit, no reweighing, regardless of ``fairness_correction``.
    - ``fairness_correction == 'kamiran_rw'`` (default): static Reweighing of
      Kamiran & Calders via :func:`reweigh_df` (blended with the ABCInML forecast
      when ``baseline_type == 'abcinml'``), then one external ``fit``. Behaviour is
      byte-for-byte the pre-APR pipeline.
    - ``fairness_correction == 'adaptative_rw'``: Adaptive Priority Reweighing
      (Hu et al., Algorithms 1-3) via :func:`adaptive_priority_reweighing`, which
      owns its internal train/reweigh loop and returns an already-fitted model.
      ``alpha`` / ``forecasted_distributions`` are forwarded only for
      ``baseline_type == 'abcinml'``.

    Returns ``(fitted_model, sample_weights)``; ``sample_weights`` is ``None`` for
    ``vanilla``.
    """
    if baseline_type == 'vanilla':
        model = initialize_model(model_name, MAX_ITER=max_iter)
        model.fit(X_train, y_train, sample_weight=None)
        return model, None

    if fairness_correction == 'adaptative_rw':
        apr_alpha = alpha if baseline_type == 'abcinml' else None
        apr_forecast = forecasted_distributions if baseline_type == 'abcinml' else None
        model, sample_weights = adaptive_priority_reweighing(
            X_train,
            y_train,
            sensitive_attribute,
            target_label,
            model_name,
            fairness_measure=apr_fairness_measure,
            T=apr_iterations,
            eta=apr_eta,
            subgroup_alpha=apr_subgroup_alpha,
            decision_boundary=apr_decision_boundary,
            max_iter=max_iter,
            alpha=apr_alpha,
            forecasted_distributions=apr_forecast,
        )
        return model, sample_weights

    # kamiran_rw (default) — unchanged legacy path
    sample_weights = reweigh_df(
        X_train,
        y_train,
        sensitive_attribute,
        target_label,
        baseline_type,
        alpha=alpha,
        forecasted_distributions=forecasted_distributions,
    )
    model = initialize_model(model_name, MAX_ITER=max_iter)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model, sample_weights


def _get_train_and_test(
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
