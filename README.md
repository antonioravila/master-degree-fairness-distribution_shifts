# Fairness Under Temporal Distribution Shifts

This project studies **algorithmic fairness under temporal distribution shifts**, using Brazilian SRAG (Severe Acute Respiratory Illness) surveillance data from 2019–2026.

The central research question is: as the statistical distribution of a population changes over time (e.g., across COVID-19 pandemic phases), do fairness-aware ML models trained on historical data remain fair? And can anticipating future distribution shifts lead to better fairness outcomes?

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data](#data)
  - [Source](#source)
  - [Feature Set](#feature-set)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Preprocessing Cache](#preprocessing-cache)
- [Baselines](#baselines)
  - [Vanilla](#vanilla)
  - [Static](#static)
  - [Dynamic](#dynamic)
  - [ABCINML (Anticipatory Bias Correction)](#abcinml-anticipatory-bias-correction)
- [Fairness Correction Methods](#fairness-correction-methods)
  - [Kamiran & Calders Reweighing](#kamiran--calders-reweighing-kamiran_rw)
  - [Adaptive Priority Reweighing](#adaptive-priority-reweighing-adaptative_rw)
- [Temporal Forecasting](#temporal-forecasting)
- [Train Window](#train-window)
- [Metrics](#metrics)
- [Outputs](#outputs)
- [Installation & Usage](#installation--usage)
- [CLI Reference](#cli-reference)
- [Parallel Grid Run](#parallel-grid-run)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Overview

The experiment runs a sliding-window evaluation over monthly or weekly batches of SRAG data. Each batch, a model is trained (or updated) and evaluated on the current batch's data. The four baselines differ in how they use historical data and how they apply fairness-aware reweighing; an independent axis (`--fairness_correction`) selects *how* that reweighing is computed (Kamiran & Calders vs. Adaptive Priority Reweighing).

The key contribution is the **ABCINML** baseline, which anticipates future demographic distribution shifts by forecasting the joint distribution of (sensitive attribute, label) and pre-emptively adjusting sample weights toward the expected future distribution.

---

## Architecture

```
src/
├── main.py                   # Entry point — orchestrates all baselines
├── data/
│   └── SRAG/
│       ├── raw/               # Raw parquet files (one per year, not committed)
│       └── cache/              # Cached preprocessed parquet (auto-invalidated, gitignored)
├── utils/
│   ├── data_utils.py          # Data ingestion, preprocessing, disk cache
│   ├── model_utils.py         # Model init, reweighing (kamiran_rw + APR), forecasting
│   ├── metrics_utils.py       # Snapshot/temporal metrics, all chart generation
│   └── general_utils.py       # Argument parsing, variable initialization
├── tools/
│   ├── regenerate_charts.py           # Rebuild figures from saved CSVs (no re-training)
│   ├── concatenate_results.py         # Merge all result CSVs into src/results/all/
│   ├── alpha_analysis.py              # Alpha-impact summary tables/figure
│   ├── analyze_forecast_quality.py    # SMA/EMA forecast accuracy diagnostics
│   ├── vanilla_shift_chart.py         # Standalone vanilla-baseline degradation chart
│   ├── migrate_legacy_schema.py       # One-off: migrate pre-APR result CSVs/folders
│   ├── migrate_train_window_schema.py # One-off: migrate pre-train_window result folders
│   ├── migrate_forecast_window_schema.py  # One-off: backfill _fw_{window} folder suffix
│   └── migrate_period_labels_en.py    # One-off: period labels in mae_by_period.csv → English
├── notebooks/                 # 01 EDA, 02 baseline comparison, 03 paper figures/tables, 04 results overview
├── figures/                   # Generated charts (gitignored)
└── results/                   # Generated CSVs (committed)
run_parallel.py                # Runs the full experiment grid across processes
```

### Data flow

```
data_utils.pre_process_df()
    └── (cache hit?)          → load cached parquet from src/data/SRAG/cache/, done
    └── _import_data()        → Read all parquet files, 42 columns
    └── _filter_region()      → No-op today (all regions kept); SG_UF_NOT dropped
    └── _fill_na()            → Drop rows with missing target/sensitive; fill NAs
    └── _convert_types()      → Encode binary/categorical features; create IDADE_BIN
    └── (post-processing)     → Drop leakage col, sort by window column, clip date range
    └── _downcast_dtypes()    → Shrink to int8/float32 to reduce memory footprint
    └── (cache write)         → Persist parquet keyed by a hash of the preprocessing source

main()                        → Run vanilla/static once (alpha- and train_window-independent):
    └── run_baseline() × {vanilla, static}
        └── cached under baseline_non_abc_{model}_{batch_type}_{fairness_correction}_fw_{window}/

                             → Run dynamic once (alpha-independent, train_window-dependent):
    └── run_baseline(dynamic)
        └── cached under baseline_dynamic_{model}_{batch_type}_{fairness_correction}_trainwindow_{tw}_fw_{window}/

                             → For each alpha, run abcinml:
    └── run_baseline(abcinml, alpha=α)
        └── save abcinml results to the per-alpha run folder
        └── combine with vanilla/static/dynamic → generate all charts (incl. comparative)

main.run_baseline()          → For each batch (month or week):
    ├── compute_batch_proportions()          → Joint (S, Y) distribution of current batch
    ├── forecast_joint_distributions[...]()  → SMA / EMA / oracle forecast (ABCINML only)
    ├── fit_fair_model()                     → dispatch to plain fit / kamiran_rw / APR
    ├── model.predict / predict_proba
    └── compute_snapshot_metrics()           → Per-batch fairness & performance
```

---

## Data

### Source

Brazilian SRAG (Síndrome Respiratória Aguda Grave) surveillance dataset published by the Ministry of Health (SIVEP-Gripe). Annual parquet files are stored in `src/data/SRAG/raw/`. Each file corresponds to one surveillance year.

The dataset spans **2019–2026** and covers the full COVID-19 pandemic arc: pre-pandemic, acute pandemic phases, transition, and endemic. The final date range kept is `2019-01` to `2026-02` (monthly) / `2019-01` to `2026-08` (weekly).

All Brazilian states/regions are currently included (region filtering is implemented in `data_utils._filter_region` but disabled — it can be restored to a subset of states by uncommenting one line).

### Feature Set

43 columns are loaded (42 model-relevant columns + `SG_UF_NOT`, which is read for the now-disabled region filter and dropped before modeling):

| Category | Columns |
|---|---|
| Target | `EVOLUCAO` (1=Cura/recovery, 2=Óbito/death) |
| Sensitive | `CS_RACA` (race), `NU_IDADE_N` (age in years) |
| Date | `DT_NOTIFIC` (notification date → `YEAR-MONTH` / `YEAR-WEEK`) |
| Symptoms | `FEBRE`, `TOSSE`, `GARGANTA`, `DISPNEIA`, `DESC_RESP`, `SATURACAO`, `DIARREIA`, `VOMITO`, `DOR_ABD`, `FADIGA`, `PERD_OLFT`, `PERD_PALA`, `OUTRO_SIN` |
| Comorbidities | `FATOR_RISC`, `PUERPERA`, `CARDIOPATI`, `HEMATOLOGI`, `SIND_DOWN`, `HEPATICA`, `ASMA`, `DIABETES`, `NEUROLOGIC`, `PNEUMOPATI`, `IMUNODEPRE`, `RENAL`, `OBESIDADE` |
| Exposure/flags | `NOSOCOMIAL`, `AVE_SUINO` |
| Treatment | `ANTIVIRAL`, `TP_ANTIVIR`, `UTI`, `SUPORT_VEN`, `VACINA_COV` |
| Exam results | `RAIOX_RES`, `TOMO_RES`, `CLASSI_FIN` |
| Demographics | `CS_GESTANT` |
| Geography (filter-only) | `SG_UF_NOT` — dropped before modeling, never a feature |

### Preprocessing Pipeline

Preprocessing is orchestrated by `data_utils.pre_process_df`.

#### Step 1: `_import_data`

Reads all parquet files under `src/data/SRAG/raw/` and concatenates them into a single DataFrame. All 43 columns are read; no filtering occurs here.

#### Step 2: `_filter_region`

Drops `SG_UF_NOT` after (optionally) filtering to a subset of states. Currently a no-op on the row filter — all regions are kept for the current all-regions experiments.

#### Step 3: `_fill_na`

1. Drops rows missing `EVOLUCAO` (target) or `CS_RACA` (sensitive attribute) — both are required.
2. Fills NAs using SRAG codebook sentinel values:
   - Most binary yes/no fields → `2` (= "Não")
   - `SUPORT_VEN` → `3`
   - `TOMO_RES`, `RAIOX_RES` → `6`
   - `CLASSI_FIN` → `4`
3. Replaces `9` ("Ignorado") with the same sentinels.
4. Drops rows where `CS_RACA == 9` or `CS_GESTANT == 9` (truly unknown race/pregnancy status cannot be imputed).

#### Step 4: `_convert_types`

Applied in this exact order (order matters), parameterized by `window_column` (`YEAR-MONTH` or `YEAR-WEEK`):

1. `DT_NOTIFIC` → `YEAR-MONTH` (pandas Period, monthly resolution) or `YEAR-WEEK` (`%Y-%U` string); original date column dropped.
2. **Binary 1/2 fields** (symptoms, comorbidities, flags): coerced to numeric, then `1 if value == 1 else 0`. Collapses "Não" (2) and all sentinels into 0.
3. **Multi-category fields** (`AVE_SUINO`, `CS_GESTANT`, `TP_ANTIVIR`, `SUPORT_VEN`, `RAIOX_RES`, `TOMO_RES`, `CLASSI_FIN`): cast to `int → str`, then one-hot encoded with `drop_first=True`.
4. `CS_RACA`: `1` (Branca/White) → `0` (privileged group); all other values → `1` (sensitive group).
5. `EVOLUCAO`: rows with values other than 1 or 2 are dropped; `1` (Cura) → `0`, `2` (Óbito/death) → `1` (positive class = death).
6. Rows with `NU_IDADE_N >= 120` are dropped as implausible ages.
7. **`IDADE_BIN` is derived here**, before normalization: `(NU_IDADE_N >= 65).astype(int)`. The threshold applies to raw years — order matters.
8. `NU_IDADE_N` is min-max normalized to `[0, 1]` using theoretical bounds `[0, 100]`.

#### Post-processing in `pre_process_df`

- DataFrame is sorted ascending by the window column.
- **Leakage prevention**:
  - `sensitive_attribute == 'IDADE_BIN'` → drop `NU_IDADE_N` (it would trivially reconstruct the sensitive attribute).
  - Otherwise → drop `IDADE_BIN` (not the target sensitive attribute; must never act as a predictor).
- Date range is clipped: `2019-01`–`2026-02` for `YEAR-MONTH`, `2019-01`–`2026-08` for `YEAR-WEEK` (replaces the older "drop 2018 rows" rule with an explicit bound on both ends).
- Dtypes are downcast (`int8`/`float32`) to reduce memory footprint; the window column is left untouched.

#### Sensitive attribute encoding

| Attribute | Value 0 | Value 1 |
|---|---|---|
| `CS_RACA` | Branca/White (privileged) | non-White (sensitive) |
| `IDADE_BIN` | Age < 65 (privileged) | Age >= 65 (sensitive) |

### Preprocessing Cache

`pre_process_df` caches its output as a parquet file under `src/data/SRAG/cache/`, named `{sensitive_attribute}_{window_column}_{fingerprint}.parquet`, where `fingerprint` is an MD5 hash of the combined source code of every preprocessing function (`_import_data`, `_filter_region`, `_fill_na`, `_convert_types`, `_downcast_dtypes`). Editing any of those functions automatically invalidates the cache — no manual version bump needed. Pass `use_cache=False` to force a fresh recompute.

---

## Baselines

The experiment iterates over batches (monthly or weekly, per `--batch_type`) in chronological order. Each baseline decides:
1. **Which training data to use.**
2. **How to compute sample weights** for fairness-aware training (delegated to whichever method `--fairness_correction` selects).

| Baseline | Training data | Reweighing |
|---|---|---|
| `vanilla` | First post-warm-up batch only, never retrained | None (uniform weights), regardless of `--fairness_correction` |
| `static` | Same single batch as `vanilla`, never retrained | Computed once from that batch, then frozen |
| `dynamic` | All prior batches (cumulative, or bounded by `--train_window`) | Recomputed from current training window each batch |
| `abcinml` | Same training data as `dynamic` | Blend of current + forecasted future: `w = α·w_current + (1−α)·w_future` |

All four baselines share the same warm-up: the first `forecast_window` batches (converted to batch units — see [Temporal Forecasting](#temporal-forecasting)) are skipped for every baseline, so all four are trained and evaluated over an identical batch range.

### Vanilla

A frozen baseline. The model is trained once, on the first batch after warm-up, with no sample weighting. It is never retrained. Serves as the lower bound — shows how a static, unaware model degrades in fairness and performance as the population distribution shifts over time.

### Static

Same frozen training approach as Vanilla (trained on the same single batch), but applies fairness reweighing computed from that batch. The weights are frozen alongside the model. Shows whether one-time reweighing at training time helps, and whether those weights remain effective as distributions shift.

### Dynamic

Retrained at every batch using accumulated prior data — unbounded by default, or bounded to the last `--train_window` months when set (see [Train Window](#train-window)). Sample weights are recomputed from the **current training window's** joint distribution of (sensitive attribute, label). This is the standard reactive fairness approach — it responds to current demographics but has no awareness of future shifts.

### ABCINML (Anticipatory Bias Correction)

The core contribution of this work. Extends the Dynamic approach by **anticipating future distribution shifts** rather than just reacting to the current one.

**Mechanism:**

At each batch `i`, it:
1. Forecasts the future joint distribution `p_hat(S, Y)` using one of the [temporal forecasting](#temporal-forecasting) techniques.
2. Computes current weights `w_current` from the current training window's distribution.
3. Derives anticipatory weights `w_future` from the forecasted distribution.
4. Blends the two: `w_final = alpha * w_current + (1 - alpha) * w_future`

**Alpha parameter:**
- `alpha = 0` → pure anticipation (fully weights toward the forecasted future distribution)
- `alpha = 1` → pure current reweighing (equivalent to Dynamic)
- `0 < alpha < 1` → interpolation between reactive and anticipatory

**Cold start:** Batches are skipped until enough history exists to forecast (same warm-up window as the other baselines). If a forecast still cannot be produced for a batch (e.g. `sma`/`ema` history not yet full), that batch is skipped too.

---

## Fairness Correction Methods

`--fairness_correction` selects *how* sample weights are computed for `static`, `dynamic`, and `abcinml` (`vanilla` never reweighs). Each writes to its own results/figures folder (folder names carry a `_kamiran_rw` or `_adaptative_rw` suffix), so the two methods never overwrite each other's runs.

### Kamiran & Calders Reweighing (`kamiran_rw`)

The default. A pre-processing method (Kamiran & Calders, 2012) that adjusts sample weights once per batch so that each (S, Y) group is represented as if S and Y were statistically independent — eliminating the correlation between the sensitive attribute and the label that drives classifier bias. Implemented in `model_utils.weight_estimation` / `reweighing` / `reweigh_df`.

For the four joint groups:

```
W(S=1, Y=1) = P(S=1) * P(Y=1) / P(S=1, Y=1)
W(S=1, Y=0) = P(S=1) * P(Y=0) / P(S=1, Y=0)
W(S=0, Y=1) = P(S=0) * P(Y=1) / P(S=0, Y=1)
W(S=0, Y=0) = P(S=0) * P(Y=0) / P(S=0, Y=0)
```

Where `S=0` is the privileged group (White / age < 65) and `S=1` is the sensitive group.

For ABCINML, the same formula is applied to the **forecasted** proportions `p_hat(S, Y)` (`weight_estimation_anticipated`) to produce anticipatory weights, which are then blended with current weights via `alpha`.

### Adaptive Priority Reweighing (`adaptative_rw`)

An *in-processing* alternative (Hu et al., 2023/2024; arXiv:2309.08375), implemented in `model_utils.adaptive_priority_reweighing`. Unlike `kamiran_rw` (weights computed once from the true labels, then handed to a single external `fit`), APR alternates between fitting the classifier and re-deriving sample weights from the *current* model's predictions and decision-boundary margins, over `--apr_iterations` rounds — it owns its own training loop.

Each iteration:
1. Fit the model with the current sample weights.
2. Compute predictions and per-sample margins to the decision boundary (`--apr_decision_boundary`), via `compute_margins` (Eq. 19 of the paper).
3. Update per-subgroup weights `W_{y,a}` (`update_apr_subgroup_weights`, Eq. 17 / Algorithms 1–3), smoothed by `--apr_subgroup_alpha`.
4. Derive new per-sample weights from `W_{y,a}` and the margins (`update_apr_sample_weights`, Eq. 18), sharpened by `--apr_eta`.

`--apr_fairness_measure` selects which algorithm variant runs: `demographic_parity` (Algorithm 1), `equalized_odds` (Algorithm 2), or `equal_opportunity` (Algorithm 3, which only updates the `y=1` subgroups).

**ABCInML integration:** when `baseline_type == 'abcinml'`, each subgroup-weight update blends two complete Eq. 17 updates — one from the current batch, one from the forecasted joint distribution — using the same `alpha` blend as `kamiran_rw`'s `w_current`/`w_future` split. For `equalized_odds`/`equal_opportunity`, the forecast carries no prediction-correctness signal, so its contribution is effectively neutral (factor ≈ 1) and only the demographic-parity-style base-rate shift is anticipated.

APR is far more expensive than `kamiran_rw` (it retrains the model `apr_iterations + 1` times per batch instead of once) — see [Parallel Grid Run](#parallel-grid-run) for how the grid runner throttles it.

---

## Temporal Forecasting

`model_utils.forecast_joint_distributions` (and its oracle counterpart) maintains/consults a rolling history of `p(S=s, Y=y)` for each of the four joint groups across all past batches. Three forecasting techniques are supported (`--forecasting_technique`):

| Technique | Description |
|---|---|
| `sma` | Simple Moving Average over the last `forecast_window` batches |
| `ema` | Exponential Moving Average with smoothing factor `--alpha_ema` (`min_periods=forecast_window`) |
| `oracle` | Diagnostic-only: reads the **actual next batch's** proportions directly (falls back to `sma` for the final batch). Requires knowledge of the future and is never a deployable method — use it only to determine whether forecast quality or the reweighing mechanism itself is the bottleneck in ABCInML. |

`sma`/`ema` return `None` if fewer than `forecast_window` batches of history are available, causing ABCINML to skip that batch.

`--forecast_window` is specified **in months** on the CLI and converted internally (`general_utils.effective_forecast_window`) to the equivalent number of batches for the chosen `--batch_type` (`round(months * 52/12)` for weekly, unchanged for monthly), so the forecasting horizon and warm-up period represent the same real-world duration regardless of batching granularity.

---

## Train Window

`--train_window` (months, same monthly→batch conversion as `forecast_window`) bounds how much history `dynamic`/`abcinml` train on:

- `None` (default): unbounded — train on every batch seen so far (original behavior).
- An integer: train only on the last `train_window` batches (a rolling window), re-sliced from the full DataFrame each batch instead of growing an unbounded accumulator.

`vanilla`/`static` ignore this entirely — they always train on a single batch. Because `train_window` changes `dynamic`'s output, it gets its own cache/result folder (`baseline_dynamic_*`), separate from the alpha- and train_window-independent `baseline_non_abc_*` folder shared by `vanilla`/`static`.

---

## Metrics

### Snapshot metrics (per batch)

Computed by `metrics_utils.compute_snapshot_metrics` at each evaluation batch:

| Metric | Column | Description |
|---|---|---|
| Accuracy | `acc` | Overall accuracy |
| AUC-ROC | `auc` | Area under the ROC curve |
| Demographic Parity Difference | `dp` | `P(Y_hat=1\|S=1) - P(Y_hat=1\|S=0)` — lower absolute value is fairer |
| TPR Difference | `tpr_diff` | `TPR(S=1) - TPR(S=0)` — equalized odds component |
| FPR Difference | `fpr_diff` | `FPR(S=1) - FPR(S=0)` — equalized odds component |
| Equalized Odds Difference | `eq_odds_diff` | `max(tpr_diff, fpr_diff)` — the worse of the two equalized odds components |

Absolute group rates (`tpr_S0`, `fpr_S0`, `tpr_S1`, `fpr_S1`) are also tracked separately (`absolute_results_df.csv`).

### Temporal metrics (summary across all batches)

Computed by `metrics_utils.compute_temporal_metrics`:

| Metric | Description |
|---|---|
| Maximum Bias | `max(\|dp\|)` across all batches — worst-case bias |
| Temporal Stability | `mean(\|delta_dp\|)` — average batch-to-batch change in bias |
| Maximum Bias Difference | `max(\|delta_dp\|)` — largest single-batch jump in bias |

---

## Outputs

Folder names double as cache keys — `main.py` loads vanilla/static and dynamic results instead of re-running them when the folder's `assessment_all_baselines.csv` already exists (`abcinml` has no such check and always runs; skipping already-complete alphas is done by `run_parallel.py`) — so every parameter that changes a baseline's output is encoded in the folder name. Model hyperparameters (fixed in `model_utils.initialize_model`) are the exception: they are encoded only through `model_name`, so a different configuration must be registered under a new `model_name` (e.g. `random_forest_deep`) rather than edited in place.

**Vanilla/Static** (`vanilla`, `static`) — run once, independent of alpha, `fairness_correction`'s specific measure and `train_window`:
```
src/results/{sensitive_attribute}/baseline_non_abc_{model_name}_{batch_type}_{fairness_correction}_fw_{forecast_window}/
src/figures/{sensitive_attribute}/baseline_non_abc_{model_name}_{batch_type}_{fairness_correction}_fw_{forecast_window}/
```

**Dynamic** — run once per `train_window` value, alpha-independent:
```
src/results/{sensitive_attribute}/baseline_dynamic_{model_name}_{batch_type}_{fairness_correction}_trainwindow_{train_window}_fw_{forecast_window}/
src/figures/{sensitive_attribute}/baseline_dynamic_{model_name}_{batch_type}_{fairness_correction}_trainwindow_{train_window}_fw_{forecast_window}/
```

**ABCINML** — one folder per alpha value:
```
src/results/{sensitive_attribute}/{model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}_{batch_type}[_ema_{alpha_ema}]_{fairness_correction}_trainwindow_{train_window}/
src/figures/{sensitive_attribute}/{model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}_{batch_type}[_ema_{alpha_ema}]_{fairness_correction}_trainwindow_{train_window}/
```

`train_window` and `forecast_window` values are `'none'`/`{value}` literals in the folder name (never omitted), and `_ema_{alpha_ema}` only appears when `--forecasting_technique ema`. The per-alpha CSV files contain **only ABCINML data**. The per-alpha chart folder contains all charts including comparative ones (which overlay vanilla/static/dynamic/abcinml by merging their DataFrames at chart-generation time).

An additional `baseline_comparison_*` folder is generated whenever 2+ of `{vanilla, static, dynamic}` ran in the same invocation, holding just the 3-baseline comparison charts (no ABCINML).

### Charts

| File | Description | Generated when |
|---|---|---|
| `model_performance_metrics.png` | Grid: one row per baseline, one column per metric. Raw values + moving average. Period spans overlaid. | Always |
| `model_performance_comparison.png` | Overlay of all baselines per metric, moving average only, common date range. | 2+ baselines in the same chart call |
| `absolute_rates_by_group.png` | TPR and FPR per sensitive group (S0/S1) over time, one row per baseline. | 2+ baselines in the same chart call |

Charts annotate five historical periods with colored spans (dates fixed regardless of `batch_type`). The labels are defined in `metrics_utils._PERIOD_SPANS` (`pre-pandemic`, `acute pandemic 1`, `acute pandemic 2`, `transition`, `endemic`):

| Period | Date range |
|---|---|
| Pre-pandemic | Feb 2019 – Mar 2020 |
| Acute pandemic 1 | Mar 2020 – Jan 2022 |
| Acute pandemic 2 | Jan 2022 – Jan 2023 |
| Transition | Jan 2023 – Jan 2024 |
| Endemic | Jan 2024 – Mar 2026 |

### CSVs

| File | Contents |
|---|---|
| `assessment_all_baselines.csv` | Per-batch snapshot metrics, tagged with `baseline`, `model_name`, `forecast_window`, `forecasting_technique`, `batch_type`, `alpha_ema`, `fairness_correction`, `apr_fairness_measure`, `train_window` |
| `temporal_results_df.csv` | Temporal summary metrics per baseline, same metadata columns |
| `absolute_results_df.csv` | Per-batch absolute TPR/FPR rates per group, same metadata columns |

### Regenerating charts without re-training

```bash
# Regenerate charts for a single run folder
uv run python src/tools/regenerate_charts.py --results_dir src/results/IDADE_BIN/logistic_regression_alpha_0.5_sma_window_5_monthly_kamiran_rw_trainwindow_none

# Regenerate all run folders under a sensitive-attribute directory
uv run python src/tools/regenerate_charts.py --results_dir src/results/IDADE_BIN
```

---

## Installation & Usage

**Requirements:** Python >= 3.12, [uv](https://github.com/astral-sh/uv), and an NVIDIA GPU with CUDA for `xgboost` (it is created with `device='cuda'`; `logistic_regression`, `random_forest` and `lightgbm` run on CPU only). `random_forest` uses every core (`n_jobs=-1`) and peaks at ~4.6 GB RAM when `dynamic` trains on the full, unbounded history. `lightgbm` also uses every core (`n_jobs=-1`); its PyPI wheel has no GPU support.

All commands must be run from the repository root — result, figure and data paths are relative (`src/results/...`).

```bash
# Install dependencies
uv sync

# Run a single baseline
uv run python src/main.py \
  --baseline_type abcinml \
  --sensitive_attribute CS_RACA \
  --model_name logistic_regression

# Run all four baselines at once
uv run python src/main.py \
  --baseline_type all \
  --sensitive_attribute CS_RACA \
  --model_name logistic_regression

# Adaptive Priority Reweighing instead of Kamiran & Calders
uv run python src/main.py \
  --baseline_type all \
  --sensitive_attribute IDADE_BIN \
  --model_name logistic_regression \
  --fairness_correction adaptative_rw \
  --apr_fairness_measure equalized_odds

# Bounded training history for dynamic/abcinml (last 6 months)
uv run python src/main.py \
  --baseline_type dynamic \
  --sensitive_attribute IDADE_BIN \
  --model_name xgboost \
  --train_window 6

# Weekly batches with EMA forecasting
uv run python src/main.py \
  --baseline_type abcinml \
  --sensitive_attribute IDADE_BIN \
  --model_name logistic_regression \
  --batch_type weekly \
  --forecasting_technique ema \
  --alpha_ema 0.3

# Quick iteration: single alpha value and capped dataset
uv run python src/main.py \
  --baseline_type abcinml \
  --sensitive_attribute IDADE_BIN \
  --model_name logistic_regression \
  --alpha 0.5 \
  --test_run True

# Lint / format
uv run ruff check src/
uv run ruff format src/
```

Note: `--window_column` no longer exists — batching granularity is selected via `--batch_type` (`monthly`/`weekly`), which derives the underlying window column (`YEAR-MONTH`/`YEAR-WEEK`) internally.

---

## CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--baseline_type` | str | `all` | `vanilla`, `static`, `dynamic`, `abcinml`, or `all` |
| `--sensitive_attribute` | str | `IDADE_BIN` | `CS_RACA` (race) or `IDADE_BIN` (age >= 65) |
| `--target_label` | str | `EVOLUCAO` | Target column |
| `--model_name` | str | — (required) | `logistic_regression`, `xgboost`, `random_forest`, or `lightgbm` |
| `--alpha` | float | `None` | Blend parameter for ABCINML. If omitted, runs all `[0.0, 0.1, ..., 1.0]` |
| `--batch_type` | str | `monthly` | `monthly` (→ `YEAR-MONTH`) or `weekly` (→ `YEAR-WEEK`) |
| `--forecasting_technique` | str | `sma` | `sma`, `ema`, or `oracle` (diagnostic only) |
| `--forecast_window` | int | `5` | Forecast/warm-up horizon, in months (converted to batches internally) |
| `--alpha_ema` | float | `0.5` | Smoothing factor for `ema` forecasting |
| `--train_window` | int | `None` | Bound (in months) on how much history `dynamic`/`abcinml` train on. `None` = unbounded. Ignored by `vanilla`/`static` |
| `--fairness_correction` | str | `kamiran_rw` | `kamiran_rw` (Kamiran & Calders) or `adaptative_rw` (Adaptive Priority Reweighing) |
| `--apr_iterations` | int | `20` | APR training rounds (`T` in the paper) |
| `--apr_eta` | float | `1.0` | APR per-sample weight sharpness (Eq. 18) |
| `--apr_subgroup_alpha` | float | `100.0` | APR subgroup-weight smoothing constant (Eq. 17) |
| `--apr_decision_boundary` | float | `0.5` | APR decision-boundary value used to compute margins (Eq. 19) |
| `--apr_fairness_measure` | str | `demographic_parity` | `demographic_parity`, `equalized_odds`, or `equal_opportunity` |
| `--test_run` | str (`'True'`/`'False'`) | `'False'` | If `'True'`, caps the preprocessed dataset at 100 000 rows for fast iteration |

---

## Parallel Grid Run

`run_parallel.py` runs the full experiment grid in parallel — separate thread pools for logistic regression (CPU), XGBoost (GPU) and a heavy-CPU pool shared by random forest and LightGBM (`--max_rf_workers`, default 1 since each RF/LightGBM job already uses every core; LightGBM jobs still run alongside XGBoost, which is on the GPU), plus a semaphore gate limiting how many `adaptative_rw` (APR) jobs run at once, since APR retrains the model `apr_iterations` times per fit and is far more expensive than `kamiran_rw`. Jobs are routed to pools by a model→pool map (`POOL_BY_MODEL`); an unlisted model falls back to the CPU pool, and an assertion guarantees no job is left unsubmitted.

```bash
# Run full grid (resumes automatically — skips already-complete scenarios)
uv run python run_parallel.py

# Custom worker counts (e.g. GPU with more VRAM headroom)
uv run python run_parallel.py --max_xgb_workers 2 --max_lr_workers 4 --max_rf_workers 1 --max_apr_workers 2

# Run only Kamiran & Calders, skipping the (expensive) APR sweep
uv run python run_parallel.py --fairness_corrections kamiran_rw

# Narrow the train_window sweep
uv run python run_parallel.py --train_windows none 6

# Run only some models (default: all four)
uv run python run_parallel.py --models lightgbm
```

Each run uses the default `sensitive_attribute=IDADE_BIN`. A scenario is considered **complete** when all 11 alpha folders (α = 0.0, 0.1, …, 1.0) exist for it under `src/results/IDADE_BIN/`. Incomplete scenarios are re-run only for their missing alphas; complete ones are skipped.

### Default scenario grid

The base grid (model × technique/window × batch_type) is:

| Model | Technique | Window (months) | Batch type |
|-------|-----------|--------|-------|
| `logistic_regression`, `xgboost`, `random_forest`, `lightgbm` | `sma` | 3, 5, 10 | `monthly`, `weekly` |
| `logistic_regression`, `xgboost`, `random_forest`, `lightgbm` | `ema` | 5 (default, no `--forecast_window` passed) | `monthly`, `weekly` |
| `logistic_regression`, `xgboost`, `random_forest`, `lightgbm` | `oracle` | 5 (default, no `--forecast_window` passed) | `monthly`, `weekly` |

That base grid (40 scenarios) is then multiplied by:
- `--fairness_corrections` (default: both `kamiran_rw` and `adaptative_rw`)
- `--train_windows` (default: `none`, `3`, `6`, `12` months)

giving 320 scenarios by default (each producing 11 alpha result folders plus one `baseline_non_abc_*` / `baseline_dynamic_*` pair). Logs for each command are written to `logs/` in the project root.

---

## Tools

All scripts live in `src/tools/` and are run with `uv run python src/tools/<script>.py`.

| Script | Purpose |
|---|---|
| `regenerate_charts.py` | Rebuild all charts for one or all run folders from their saved CSVs, without re-training. |
| `concatenate_results.py` | Merge every result CSV under `src/results/IDADE_BIN/` into `src/results/all/`, handling both the `baseline_non_abc_*`/`baseline_dynamic_*` and per-alpha folder naming conventions. |
| `alpha_analysis.py` | Generates max/min/mean statistical tables (per model) summarizing the impact of `alpha` on ABCInML's metrics. |
| `analyze_forecast_quality.py` | Compares SMA/EMA forecasted joint-distribution proportions against observed values per batch, to diagnose forecast accuracy. |
| `vanilla_shift_chart.py` | Standalone chart of the `vanilla` baseline's fairness degradation over time, comparing monthly vs. weekly batching. |
| `migrate_legacy_schema.py` | One-off migration: renames pre-APR-refactor CSV columns/folders (`sp`→`dp`, `tpr`→`tpr_diff`, `fpr`→`fpr_diff`) to the current schema. |
| `migrate_train_window_schema.py` | One-off migration: splits legacy `baseline_non_abc_*` folders into separate `vanilla`/`static` vs. `dynamic` folders and appends the `_trainwindow_none` suffix everywhere, for results produced before `--train_window` existed. |
| `migrate_forecast_window_schema.py` | One-off migration: backfills the `_fw_{forecast_window}` suffix onto existing `baseline_non_abc_*`/`baseline_dynamic_*` folders. |
| `migrate_add_eq_odds_diff.py` | Adds/recomputes `eq_odds_diff = max(tpr_diff, fpr_diff)` in every `assessment_all_baselines.csv` under `src/results/` (including `all/`), without retraining. Idempotent — rerun it on results produced by older code. `--dry-run` lists the files without writing. |
| `migrate_period_labels_en.py` | One-off migration: translates the epidemiological period labels stored in every `mae_by_period.csv` under `src/results/forecast_quality/` from Portuguese to English, matching `metrics_utils._PERIOD_SPANS`. Idempotent; `--dry-run` lists the files without writing. |

### Paper figures and tables

`src/notebooks/03_paper_figures_tables.ipynb` generates **only** the figures and tables that appear in the paper, into `src/notebooks/figures/paper/`. It reads `src/results/all/` (filtered to `kamiran_rw`, `train_window = none`, a single `FORECAST_WINDOW = 5` for every item), `src/results/forecast_quality/IDADE_BIN/` and, for Figs. 1–2 only, the raw SRAG parquets (`RUN_EDA = False` skips them). Run `concatenate_results.py` first so `src/results/all/` is up to date. A single forecast window is used because vanilla/static are trained once on the first post-warm-up batch, which depends on the window.

| Paper | File | Content |
|---|---|---|
| Fig. 1 | `fig01_cfr_temporal.png` | Weekly case fatality rate |
| Fig. 2 | `fig02b_age_period_local_scale.png` | Cases / deaths / fatality by age and period; top part shares the y-axis per row, gray bottom part has a free local y-axis |
| Fig. 3 | `fig03_vanilla_lr.png` | Vanilla LR under distribution shift |
| Fig. 4 | `fig04_baselines_lr.png` | The 4 baselines (LR) |
| Fig. 5 | `fig05c_frontier_dp_tpr_fpr_combined.png` | Accuracy × DP / TPR / FPR across α, LR × XGBoost × RF (monthly and weekly averaged) |
| Fig. 6 | `fig09b_forecast_vs_real_combined.png` | EMA forecast × real proportions |
| Tab. 3 | `tab03b_absolute_rates_lr_combined.tex` | Absolute TPR / FPR by baseline and group (LR) |
| Tab. 4 | `tab04_alpha_lr.tex` | α sweep (LR, monthly) |
| Tab. 5 | `tab05b_temporal_lr_combined.tex` | Temporal metrics (LR), monthly and weekly side by side |
| Tab. 6 | `tab06c_baselines_models.tex` | 4 baselines × 3 models, DP reduction vs. vanilla, ABCInML − Dynamic |
| Tab. 7 | `tab05c_temporal_baselines_models.tex` | Temporal metrics, 4 baselines × 3 models |
| Tab. 8 | `tab08b_mae_by_period.tex` | Forecast MAE of SMA and EMA by period |
| Tab. 9 | `tab10f_forecast_error_vs_fairness.tex` | Forecast MAE and ABCInML fairness for SMA / EMA / Oracle (LR and XGBoost) |

Section 5 of the notebook (*Dataset analytics*, for the Dataset Review) also writes `tabds01`–`tabds07` (overview, preprocessing funnel, profile by period, missing values, feature shift, selection by year, proxy correlations) and `figds01_records_per_batch.png`; the raw-data part runs only with `RUN_EDA = True`, the rest reads the preprocessing cache.

Tabs. 1–2 of the paper are written by hand.

---

## Project Structure

```
.
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency versions (uv)
├── .python-version          # Python version pinned for uv
├── README.md
├── run_parallel.py          # Full-grid parallel runner
├── logs/                    # run_parallel.py per-job logs (gitignored)
└── src/
    ├── main.py              # Entry point; run_baseline() loop
    ├── data/
    │   └── SRAG/
    │       ├── raw/          # Raw parquet files (not committed)
    │       └── cache/         # Cached preprocessed parquet (gitignored)
    ├── figures/              # Generated charts (gitignored)
    │   └── {sensitive_attribute}/
    │       ├── baseline_non_abc_{model}_{batch_type}_{fairness_correction}_fw_{window}/
    │       ├── baseline_dynamic_{model}_{batch_type}_{fairness_correction}_trainwindow_{tw}_fw_{window}/
    │       ├── baseline_comparison_{...}/
    │       └── {model}_alpha_{alpha}_{technique}_window_{window}_{batch_type}[_ema_{alpha_ema}]_{fairness_correction}_trainwindow_{tw}/
    ├── results/              # Generated CSVs (committed), mirrors figures/ layout, plus:
    │   ├── all/               # concatenate_results.py output
    │   └── forecast_quality/  # analyze_forecast_quality.py output
    ├── notebooks/            # Exploratory analysis notebooks
    ├── tools/                # Standalone analysis & migration scripts (see Tools)
    └── utils/
        ├── data_utils.py    # Data ingestion, preprocessing, disk cache
        ├── model_utils.py   # Model creation, kamiran_rw + APR reweighing, forecasting
        ├── metrics_utils.py # Metrics computation and chart generation
        └── general_utils.py # Argument parsing and state initialization
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and time-series handling |
| `scikit-learn` | Logistic Regression, Random Forest, accuracy, AUC, confusion matrix |
| `xgboost` | Gradient boosted tree classifier (GPU, `device='cuda'`) |
| `lightgbm` | Leaf-wise gradient boosted tree classifier (CPU, all cores) |
| `fairlearn` | Demographic parity, TPR diff, FPR diff |
| `matplotlib` | All chart generation |
| `seaborn` | Color palette used for chart period-span colors |
| `pyarrow` | Parquet file reading/writing (raw data and preprocessing cache) |
| `folktables` | Reference fairness-dataset utilities used in exploratory notebooks |
| `tqdm` | Progress bars (APR training loop) |
| `requests` | HTTP utilities used in exploratory/data-fetch notebooks |
| `ruff` | Linting and formatting |

Development group (`[dependency-groups] dev`, installed by `uv sync` by default):

| Package | Purpose |
|---|---|
| `jupyter`, `ipykernel` | Running the exploratory notebooks in `src/notebooks/` |
| `nbconvert` | Exporting notebooks |
