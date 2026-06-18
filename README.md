# Fairness Under Temporal Distribution Shifts

Master's thesis research project studying **algorithmic fairness under temporal distribution shifts**, using Brazilian SRAG (Severe Acute Respiratory Illness) surveillance data from 2019–2026.

The central research question is: as the statistical distribution of a population changes over time (e.g., across COVID-19 pandemic phases), do fairness-aware ML models trained on historical data remain fair? And can anticipating future distribution shifts lead to better fairness outcomes?

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data](#data)
  - [Source](#source)
  - [Feature Set](#feature-set)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
- [Baselines](#baselines)
  - [Vanilla](#vanilla)
  - [Static](#static)
  - [Dynamic](#dynamic)
  - [ABCINML (Anticipatory Bias Correction)](#abcinml-anticipatory-bias-correction)
- [Reweighing Algorithm](#reweighing-algorithm)
- [Temporal Forecasting](#temporal-forecasting)
- [Metrics](#metrics)
- [Outputs](#outputs)
- [Installation & Usage](#installation--usage)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)

---

## Overview

The experiment runs a sliding-window evaluation over monthly batches of SRAG data. Each month, a model is trained (or updated) and evaluated on the current month's data. The four baselines differ in how they use historical data and how they apply fairness-aware reweighing.

The key innovation is the **ABCINML** baseline, which anticipates future demographic distribution shifts by forecasting the joint distribution of (sensitive attribute, label) and pre-emptively adjusting sample weights toward the expected future distribution.

---

## Architecture

```
src/
├── main.py                   # Entry point — orchestrates all baselines
├── data/
│   └── SRAG/raw/             # Raw parquet files (one per year)
├── utils/
│   ├── data_utils.py         # Data ingestion and preprocessing
│   ├── model_utils.py        # Model initialization, reweighing, forecasting
│   ├── metrics_utils.py      # Snapshot/temporal metrics, all chart generation
│   └── general_utils.py      # Argument parsing, variable initialization
├── figures/                  # Generated charts (gitignored)
│   └── {sensitive_attribute}/{model}_alpha_{alpha}_{technique}_window_{window}/
└── results/                  # Generated CSVs (gitignored)
    └── {sensitive_attribute}/{model}_alpha_{alpha}_{technique}_window_{window}/
```

### Data flow

```
data_utils.pre_process_df()
    └── _import_data()       → Read all parquet files, 42 columns
    └── _fill_na()           → Drop rows with missing target/sensitive; fill NAs
    └── _convert_types()     → Encode binary/categorical features; create IDADE_BIN
    └── (post-processing)    → Drop 2018 rows, sort by YEAR-MONTH, drop leakage cols

main.run_baseline()          → For each batch (month):
    ├── compute_batch_proportions()    → Joint (S, Y) distribution of current batch
    ├── forecast_joint_distributions() → Rolling average forecast (ABCINML only)
    ├── reweigh_df()                   → Compute sample weights
    ├── model.fit(X_train, y_train, sample_weight=...)
    ├── model.predict / predict_proba
    ├── compute_snapshot_metrics()     → Per-batch fairness & performance
    └── compute_shap_values()          → Feature importance per batch
```

---

## Data

### Source

Brazilian SRAG (Síndrome Respiratória Aguda Grave) surveillance dataset published by the Ministry of Health (SIVEP-Gripe). Annual parquet files are stored in `src/data/SRAG/raw/`. Each file corresponds to one surveillance year.

The dataset spans **2019–2026** and covers the full COVID-19 pandemic arc: pre-pandemic, acute pandemic phases, transition, and endemic. Rows with year 2018 are discarded (incomplete surveillance year).

### Feature Set

42 columns are loaded:

| Category | Columns |
|---|---|
| Target | `EVOLUCAO` (1=Cura/recovery, 2=Óbito/death) |
| Sensitive | `CS_RACA` (race), `NU_IDADE_N` (age in years) |
| Date | `DT_NOTIFIC` (notification date → `YEAR-MONTH`) |
| Symptoms | `FEBRE`, `TOSSE`, `GARGANTA`, `DISPNEIA`, `DESC_RESP`, `SATURACAO`, `DIARREIA`, `VOMITO`, `DOR_ABD`, `FADIGA`, `PERD_OLFT`, `PERD_PALA`, `OUTRO_SIN` |
| Comorbidities | `FATOR_RISC`, `PUERPERA`, `CARDIOPATI`, `HEMATOLOGI`, `SIND_DOWN`, `HEPATICA`, `ASMA`, `DIABETES`, `NEUROLOGIC`, `PNEUMOPATI`, `IMUNODEPRE`, `RENAL`, `OBESIDADE` |
| Exposure/flags | `NOSOCOMIAL`, `AVE_SUINO` |
| Treatment | `ANTIVIRAL`, `TP_ANTIVIR`, `UTI`, `SUPORT_VEN`, `VACINA_COV` |
| Exam results | `RAIOX_RES`, `TOMO_RES`, `CLASSI_FIN` |
| Demographics | `CS_GESTANT` |

### Preprocessing Pipeline

Preprocessing is orchestrated by `data_utils.pre_process_df` in three private steps.

#### Step 1: `_import_data`

Reads all parquet files under `src/data/SRAG/raw/` and concatenates them into a single DataFrame. All 42 columns are read; no filtering occurs here.

#### Step 2: `_fill_na`

1. Drops rows missing `EVOLUCAO` (target) or `CS_RACA` (sensitive attribute) — both are required.
2. Fills NAs using SRAG codebook sentinel values:
   - Most binary yes/no fields → `2` (= "Não")
   - `SUPORT_VEN` → `3`
   - `TOMO_RES`, `RAIOX_RES` → `6`
   - `CLASSI_FIN` → `4`
3. Replaces `9` ("Ignorado") with the same sentinels.
4. Drops rows where `CS_RACA == 9` or `CS_GESTANT == 9` (truly unknown race/pregnancy status cannot be imputed).

#### Step 3: `_convert_types`

Applied in this exact order (order matters):

1. `DT_NOTIFIC` → `YEAR-MONTH` (pandas Period, monthly resolution); original date column dropped.
2. **Binary 1/2 fields** (symptoms, comorbidities, flags): coerced to numeric, then `1 if value == 1 else 0`. Collapses "Não" (2) and all sentinels into 0.
3. **Multi-category fields** (`AVE_SUINO`, `CS_GESTANT`, `TP_ANTIVIR`, `SUPORT_VEN`, `RAIOX_RES`, `TOMO_RES`, `CLASSI_FIN`): cast to `int → str`, then one-hot encoded with `drop_first=True`.
4. `CS_RACA`: `1` (Branca/White) → `0` (privileged group); all other values → `1` (sensitive group).
5. `EVOLUCAO`: rows with values other than 1 or 2 are dropped; `1` (Cura) → `0`, `2` (Óbito/death) → `1` (positive class = death).
6. Rows with `NU_IDADE_N >= 120` are dropped as implausible ages.
7. **`IDADE_BIN` is derived here**, before normalization: `(NU_IDADE_N >= 65).astype(int)`. The threshold applies to raw years — order matters.
8. `NU_IDADE_N` is min-max normalized to `[0, 1]` using theoretical bounds `[0, 100]`.

#### Post-processing in `pre_process_df`

- Rows from 2018 are dropped (incomplete surveillance year).
- DataFrame is sorted ascending by `YEAR-MONTH`.
- **Leakage prevention**:
  - `sensitive_attribute == 'IDADE_BIN'` → drop `NU_IDADE_N` (it would trivially reconstruct the sensitive attribute).
  - Otherwise → drop `IDADE_BIN` (not the target sensitive attribute; must never act as a predictor).

#### Sensitive attribute encoding

| Attribute | Value 0 | Value 1 |
|---|---|---|
| `CS_RACA` | Branca/White (privileged) | non-White (sensitive) |
| `IDADE_BIN` | Age < 65 (privileged) | Age >= 65 (sensitive) |

---

## Baselines

The experiment iterates over monthly batches in chronological order. Each baseline decides:
1. **Which training data to use.**
2. **How to compute sample weights** for fairness-aware training.

| Baseline | Training data | Reweighing |
|---|---|---|
| `vanilla` | Batch 0 only, never retrained | None (uniform weights) |
| `static` | Batch 0 only, never retrained | Batch 0 proportions (computed once, frozen) |
| `dynamic` | All prior batches (cumulative) | Current batch proportions |
| `abcinml` | All prior batches (cumulative) | Blend: `alpha * w_current + (1 - alpha) * w_future` |

### Vanilla

A frozen baseline. The model is trained once on the first monthly batch with no sample weighting. It is never retrained. Serves as the lower bound — shows how a static, unaware model degrades in fairness and performance as the population distribution shifts over time.

### Static

Same frozen training approach as Vanilla (trained on batch 0 only), but applies fairness reweighing computed from batch 0's proportions. The weights are frozen alongside the model. Shows whether one-time reweighing at training time helps, and whether those weights remain effective as distributions shift.

### Dynamic

Retrained at every batch using all accumulated prior data. Sample weights are recomputed from the **current batch's** joint distribution of (sensitive attribute, label). This is the standard reactive fairness approach — it responds to current demographics but has no awareness of future shifts.

### ABCINML (Anticipatory Bias Correction)

The core contribution of this thesis. Extends the Dynamic approach by **anticipating future distribution shifts** rather than just reacting to the current one.

**Mechanism:**

At each batch `i`, it:
1. Computes current weights `w_current` from the current batch's distribution.
2. Forecasts the future joint distribution `p_hat(S, Y)` using a rolling window over the last `forecast_window` batches (SMA or EMA).
3. Derives anticipatory weights `w_future` from the forecasted distribution.
4. Blends the two: `w_final = alpha * w_current + (1 - alpha) * w_future`

**Alpha parameter:**
- `alpha = 0` → pure anticipation (fully weights toward the forecasted future distribution)
- `alpha = 1` → pure current reweighing (equivalent to Dynamic)
- `0 < alpha < 1` → interpolation between reactive and anticipatory

**Cold start:** The first `forecast_window` batches are skipped (insufficient history). The model starts producing evaluations only after a full window of history is available.

---

## Reweighing Algorithm

The reweighing algorithm (Kamiran & Calders, 2012) adjusts sample weights so that each (S, Y) group is represented as if S and Y were statistically independent — eliminating the correlation between the sensitive attribute and the label that drives classifier bias.

For the four joint groups:

```
W(S=1, Y=1) = P(S=1) * P(Y=1) / P(S=1, Y=1)
W(S=1, Y=0) = P(S=1) * P(Y=0) / P(S=1, Y=0)
W(S=0, Y=1) = P(S=0) * P(Y=1) / P(S=0, Y=1)
W(S=0, Y=0) = P(S=0) * P(Y=0) / P(S=0, Y=0)
```

Where `S=0` is the privileged group (White / age < 65) and `S=1` is the sensitive group.

For ABCINML, the same formula is applied to the **forecasted** proportions `p_hat(S, Y)` to produce anticipatory weights, which are then blended with current weights via `alpha`.

---

## Temporal Forecasting

`model_utils.forecast_joint_distributions` maintains a rolling history of `p(S=s, Y=y)` for each of the four joint groups across all past batches. Two forecasting techniques are supported:

| Technique | Description |
|---|---|
| `sma` | Simple Moving Average over the last `forecast_window` batches |
| `ema` | Exponential Moving Average with `span=forecast_window` |

Returns `None` if fewer than `forecast_window` batches of history are available (causing ABCINML to skip that batch).

---

## Metrics

### Snapshot metrics (per batch)

Computed by `metrics_utils.compute_snapshot_metrics` at each evaluation batch:

| Metric | Column | Description |
|---|---|---|
| Accuracy | `acc` | Overall accuracy |
| AUC-ROC | `auc` | Area under the ROC curve |
| Demographic Parity Difference | `sp` | `P(Y_hat=1|S=1) - P(Y_hat=1|S=0)` — lower absolute value is fairer |
| TPR Difference | `tpr` | `TPR(S=1) - TPR(S=0)` — equalized odds component |
| FPR Difference | `fpr` | `FPR(S=1) - FPR(S=0)` — equalized odds component |

Absolute group rates (`tpr_S0`, `fpr_S0`, `tpr_S1`, `fpr_S1`) are also tracked separately.

### Temporal metrics (summary across all batches)

Computed by `metrics_utils.compute_temporal_metrics`:

| Metric | Description |
|---|---|
| Maximum Bias | `max(|sp|)` across all batches — worst-case bias |
| Temporal Stability | `mean(|delta_sp|)` — average batch-to-batch change in bias |
| Maximum Bias Difference | `max(|delta_sp|)` — largest single-batch jump in bias |

### SHAP values

`metrics_utils.compute_shap_values` uses SHAP's `LinearExplainer` (logistic regression) or `TreeExplainer` (XGBoost) to compute per-sample feature attributions at each batch. These are aggregated into global importance (mean |SHAP|) and a temporal heatmap showing how feature importance evolves over pandemic phases.

---

## Outputs

All outputs are placed under `src/figures/` (charts) and `src/results/` (CSVs), organized by:
```
{sensitive_attribute}/{model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}/
```

### Charts

| File | Description | When generated |
|---|---|---|
| `model_performance_metrics.png` | Grid: one row per baseline, one column per metric. Raw values + 10-batch moving average. Period spans overlaid. | Always |
| `model_performance_comparison.png` | Overlay of all baselines per metric, moving average only, common date range. | `--baseline_type all` only |
| `absolute_rates_by_group.png` | TPR and FPR per sensitive group (S0/S1) over time, one row per baseline. | `--baseline_type all` only |
| `shap_summary.png` | Global feature importance bar + temporal SHAP heatmap. | Always (logistic regression only) |

Charts annotate five historical periods with colored spans:

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
| `assessment_all_baselines.csv` | Per-batch snapshot metrics for all baselines and alphas |
| `temporal_results_df.csv` | Temporal summary metrics per baseline |
| `absolute_results_df.csv` | Per-batch absolute TPR/FPR rates per group |

---

## Installation & Usage

**Requirements:** Python >= 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run a single baseline
uv run python src/main.py \
  --baseline_type abcinml \
  --sensitive_attribute CS_RACA \
  --target_label EVOLUCAO \
  --window_column YEAR-MONTH \
  --model_name logistic_regression

# Run all four baselines at once
uv run python src/main.py \
  --baseline_type all \
  --sensitive_attribute CS_RACA \
  --target_label EVOLUCAO \
  --window_column YEAR-MONTH \
  --model_name logistic_regression

# Quick iteration: single alpha value and capped dataset
uv run python src/main.py \
  --baseline_type abcinml \
  --sensitive_attribute IDADE_BIN \
  --target_label EVOLUCAO \
  --window_column YEAR-MONTH \
  --model_name logistic_regression \
  --alpha 0.5 \
  --test_run True

# Lint / format
uv run ruff check src/
uv run ruff format src/
```

---

## CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--baseline_type` | str | `abcinml` | `vanilla`, `static`, `dynamic`, `abcinml`, or `all` |
| `--sensitive_attribute` | str | — | `CS_RACA` (race) or `IDADE_BIN` (age >= 65) |
| `--target_label` | str | — | `EVOLUCAO` |
| `--window_column` | str | — | `YEAR-MONTH` |
| `--model_name` | str | — | `logistic_regression` or `xgboost` |
| `--alpha` | float | `None` | Blend parameter for ABCINML. If omitted, runs all `[0.0, 0.1, ..., 1.0]` |
| `--forecasting_technique` | str | `sma` | `sma` (simple moving average) or `ema` (exponential moving average) |
| `--forecast_window` | int | `5` | Number of past batches used for forecasting |
| `--test_run` | bool | `False` | If `True`, caps the dataset at 100 000 rows for fast iteration |

---

## Project Structure

```
.
├── pyproject.toml           # Project metadata and dependencies
├── CLAUDE.md                # Codebase instructions for Claude Code
├── README.md
└── src/
    ├── main.py              # Entry point; run_baseline() loop
    ├── data/
    │   └── SRAG/raw/        # Raw parquet files (not committed)
    ├── figures/             # Generated charts (gitignored)
    ├── results/             # Generated CSVs (gitignored)
    └── utils/
        ├── data_utils.py    # Data ingestion and preprocessing
        ├── model_utils.py   # Model creation, reweighing, forecasting
        ├── metrics_utils.py # Metrics computation and chart generation
        └── general_utils.py # Argument parsing and state initialization
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and time-series handling |
| `scikit-learn` | Logistic Regression, accuracy, AUC, confusion matrix |
| `xgboost` | Gradient boosted tree classifier |
| `fairlearn` | Demographic parity, TPR diff, FPR diff |
| `shap` | Model explainability (SHAP values) |
| `matplotlib` | All chart generation |
| `pyarrow` | Parquet file reading |
| `ruff` | Linting and formatting |
