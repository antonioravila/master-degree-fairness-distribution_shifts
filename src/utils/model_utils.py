import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def initialize_model(model_name, MAX_ITER):
    models_mapping = {
        'logistic_regression': LogisticRegression(max_iter=MAX_ITER),
        'xgboost': xgb.XGBClassifier(device='cuda', tree_method='hist'),
        # Hyperparameters are not in the result folder name (the cache key):
        # changing them requires a new model_name.
        # No class_weight on purpose: group balancing is the reweighing's job.
        # No min_samples_leaf: at 50 the ~7k-row vanilla/static batch never predicted
        # a positive; max_depth alone bounds tree size (memory is dominated by the
        # data copy, not the trees).
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
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


def _apr_eq17_factor(f1, f2, denom_count, m_scale, subgroup_alpha):
    """One complete Eq. 17 (Hu et al.) multiplicative factor for a single subgroup.

        factor = (f1 * f2 + a_sub) / (m_scale * denom_count + a_sub)

    ``W_{y,a}^(t) = W_{y,a}^(t-1) * factor``. Every argument MUST come from a
    single source — all current-batch counts, or all forecast-derived expected
    counts — never a mix, so the numerator (an independence-implied ``P_exp``) and
    the denominator (an observed ``P_obs``) stay probabilistically consistent.
    Returns ``None`` when the denominator collapses (no data), meaning "leave
    ``W_{y,a}`` unchanged".
    """
    denominator = m_scale * denom_count + subgroup_alpha
    if denominator <= 0.0:
        return None
    return (f1 * f2 + subgroup_alpha) / denominator


def _apr_current_counts(y, a, y_pred, sensitive_attribute_values, y_true_values, fairness_measure, m):
    """``(f1, f2, denom_count, m_scale)`` for Eq. 17 from the CURRENT batch only.

    - ``demographic_parity`` (Algorithm 1, Eq. 17): ``|{y_pred = y}| * |{a = a}|``
      over ``m * |{y_pred = y, a = a}|``.
    - ``equalized_odds`` / ``equal_opportunity`` (Algorithm 2/3): counts are
      conditioned on a correct prediction — ``|{y_pred = y, y = y}| *
      |{y = y, a = a}|`` over ``m_{y,*} * |{y_pred = y, y = y, a = a}|``.
    """
    a_mask = sensitive_attribute_values == a
    if fairness_measure == 'demographic_parity':
        pred_y_mask = y_pred == y
        f1 = float(np.sum(pred_y_mask))                   # |{y_pred = y}|
        f2 = float(np.sum(a_mask))                        # |{a = a}|
        denom_count = float(np.sum(pred_y_mask & a_mask))  # |{y_pred = y, a = a}|
        m_scale = float(m)
    else:  # equalized_odds / equal_opportunity
        true_y_mask = y_true_values == y
        correct_y_mask = (y_pred == y) & true_y_mask
        f1 = float(np.sum(correct_y_mask))               # |{y_pred = y, y = y}|
        f2 = float(np.sum(true_y_mask & a_mask))         # |{y = y, a = a}|
        denom_count = float(np.sum(correct_y_mask & a_mask))  # |{y_pred = y, y = y, a = a}|
        m_scale = float(np.sum(true_y_mask))             # m_{y,*}
    return f1, f2, denom_count, m_scale


def _apr_forecast_counts(y, a, forecasted_distributions, fairness_measure, m):
    """``(f1, f2, denom_count, m_scale)`` for Eq. 17 derived ENTIRELY from the four
    forecast joint proportions in ``forecasted_distributions`` (``S0_Y1``,
    ``S0_Y0``, ``S1_Y1``, ``S1_Y0``), by summing them into the needed marginals —
    the same pattern :func:`weight_estimation_anticipated` uses for K&C. Counts
    are placed on the current batch's ``m`` scale so the resulting Eq. 17 factor
    is directly comparable to the current-batch one before blending.

    For ``demographic_parity`` this reproduces exactly the K&C anticipated ratio
    ``P(Y=y) P(A=a) / P(Y=y, A=a)``. For ``equalized_odds`` / ``equal_opportunity``
    the forecast carries no information about *prediction correctness* (it only
    knows the (S, Y) base rates), so ``f1 == m_scale`` and ``f2 == denom_count``
    by construction and the factor is ~1 (exactly 1 as ``subgroup_alpha -> 0``):
    a joint (S, Y) forecast cannot anticipate an equalized-odds violation, only a
    demographic-parity-style base-rate shift.
    """
    p_y = forecasted_distributions[f'S0_Y{y}'] + forecasted_distributions[f'S1_Y{y}']
    p_a = forecasted_distributions[f'S{a}_Y0'] + forecasted_distributions[f'S{a}_Y1']
    p_ya = forecasted_distributions[f'S{a}_Y{y}']
    if fairness_measure == 'demographic_parity':
        f1 = m * p_y            # expected |{y_pred = y}|
        f2 = m * p_a            # expected |{a = a}|
        denom_count = m * p_ya  # expected |{y_pred = y, a = a}|
        m_scale = float(m)
    else:  # equalized_odds / equal_opportunity (no correctness signal in forecast)
        f1 = m * p_y            # expected |{y_pred = y, y = y}|  (predictions assumed correct)
        f2 = m * p_ya           # expected |{y = y, a = a}|
        denom_count = m * p_ya  # expected |{y_pred = y, y = y, a = a}|
        m_scale = m * p_y       # expected m_{y,*}
    return f1, f2, denom_count, m_scale


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
    subgroups; the returned dict holds ``W_{y,a}^(t)``. ``fairness_measure``
    selects Algorithm 1 (``demographic_parity``), Algorithm 2 (``equalized_odds``)
    or Algorithm 3 (``equal_opportunity`` — only the ``y = 1`` subgroups updated,
    ``y = 0`` keeps ``W_{0,a}^(0) = 1``). ``subgroup_alpha`` is the paper's
    ``alpha`` smoothing constant (``apr_subgroup_alpha`` on the CLI).

    ABCInML forecasting (only when ``baseline_type == 'abcinml'``, i.e. both
    ``alpha`` and ``forecasted_distributions`` are given): mirrors exactly how
    :func:`reweigh_df` blends K&C — two *complete* Eq. 17 updates are computed
    independently and blended only at the end::

        W_current  = W_prev * eq17(current-batch counts)          # all P_obs from now
        W_forecast = W_prev * eq17(forecast expected counts)      # all P_obs from forecast
        W_{y,a}    = alpha * W_current + (1 - alpha) * W_forecast

    where ``alpha`` is the CLI ABCInML blend (NOT ``subgroup_alpha``). Numerator
    and denominator of each Eq. 17 update come from a single source — never a
    present/forecast mix inside one fraction. Eq. 18 (per-sample margin term) is
    left untouched by the forecast: ``forecasted_distributions`` only carries
    subgroup-level aggregates.
    """
    m = len(y_pred)
    W = dict(prev_W)

    if fairness_measure == 'equal_opportunity':
        pairs = [(1, 0), (1, 1)]
    else:
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    use_forecast = alpha is not None and forecasted_distributions is not None

    for y, a in pairs:
        f1_c, f2_c, dc_c, ms_c = _apr_current_counts(
            y, a, y_pred, sensitive_attribute_values, y_true_values, fairness_measure, m
        )
        factor_current = _apr_eq17_factor(f1_c, f2_c, dc_c, ms_c, subgroup_alpha)

        if not use_forecast:
            if factor_current is not None:
                W[(y, a)] = prev_W[(y, a)] * factor_current
            continue

        # Two complete Eq. 17 updates, blended only on the finished results —
        # same pattern as reweigh_df's K&C w_current / w_future blend.
        f1_f, f2_f, dc_f, ms_f = _apr_forecast_counts(
            y, a, forecasted_distributions, fairness_measure, m
        )
        factor_forecast = _apr_eq17_factor(f1_f, f2_f, dc_f, ms_f, subgroup_alpha)

        w_current = prev_W[(y, a)] * factor_current if factor_current is not None else prev_W[(y, a)]
        w_forecast = prev_W[(y, a)] * factor_forecast if factor_forecast is not None else prev_W[(y, a)]
        W[(y, a)] = alpha * w_current + (1.0 - alpha) * w_forecast

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
    supplied (only for ``baseline_type == 'abcinml'``), each subgroup weight is a
    blend of two *complete* Eq. 17 updates (one from the current batch, one from
    the forecast) — ``alpha`` is the CLI ABCInML blend, see
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
      ``baseline_type == 'abcinml'`` (for ``static`` / ``dynamic`` they are
      ``None``, so the CLI ``alpha`` sweep never re-runs APR for those baselines).

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
