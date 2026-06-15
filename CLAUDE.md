# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the experiment
uv run python src/main.py --baseline_type abcinml --sensitive_attribute CS_RACA --target_label EVOLUCAO --window_column YEAR-MONTH --model_name logistic_regression

# Run all baselines at once
uv run python src/main.py --baseline_type all --sensitive_attribute CS_RACA --target_label EVOLUCAO --window_column YEAR-MONTH --model_name logistic_regression

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

**CLI arguments:**
- `--baseline_type`: `vanilla`, `static`, `dynamic`, `abcinml`, or `all`
- `--sensitive_attribute`: `CS_RACA` (race) or `IDADE_BIN` (age)
- `--target_label`: `EVOLUCAO`
- `--window_column`: `YEAR-MONTH`
- `--model_name`: `logistic_regression` or `xgboost`
- `--alpha`: float 0–1 (default: runs all [0, 0.1, ..., 1.0]); pass a single value (e.g. `--alpha 0.5`) for faster iteration during development
- `--test_run`: `True` to cap the dataset at 800 000 rows for quick iteration

## Architecture

This is a master's thesis research project studying **algorithmic fairness under temporal distribution shifts**, using Brazilian SRAG (Severe Acute Respiratory Illness) surveillance data from 2019–2026.

### Data flow

`data_utils.pre_process_df` orchestrates three private steps then applies final filters.

**`_import_data`** — reads a fixed set of 42 columns from all parquet files under `src/data/SRAG/raw/` and concatenates them into a single DataFrame. No filtering at this stage.

**`_fill_na`** — drops rows missing `EVOLUCAO` or `CS_RACA` (target and sensitive attribute — both required). Remaining NAs are filled with sentinel values matching the SRAG codebook: `2` (= "Não") for most binary yes/no fields, `3` for `SUPORT_VEN`, `6` for `TOMO_RES`/`RAIOX_RES`, `4` for `CLASSI_FIN`. `9` (= "Ignorado") is replaced with the same sentinel. Rows where `CS_RACA == 9` or `CS_GESTANT == 9` are dropped entirely.

**`_convert_types`** — applies the following transformations in order:
1. `DT_NOTIFIC` → `YEAR-MONTH` (period M); original column dropped.
2. Binary 1/2 fields (symptoms, comorbidities, flags): coerced to numeric, then remapped to `1 if value == 1 else 0`. This collapses "Não" (2) and any sentinel into 0.
3. Multi-category fields (`AVE_SUINO`, `CS_GESTANT`, `TP_ANTIVIR`, `SUPORT_VEN`, `RAIOX_RES`, `TOMO_RES`, `CLASSI_FIN`): cast to `int → str`, then one-hot encoded with `drop_first=True`.
4. `CS_RACA`: `1` (Branca) → `0` (privileged); all other values → `1` (sensitive).
5. `EVOLUCAO`: rows with values other than 1 (Cura) or 2 (Óbito) are dropped; `1 → 0`, `2 → 1` (death = positive class).
6. Rows with `NU_IDADE_N >= 120` are dropped as implausible.
7. **`IDADE_BIN` is derived here**, before normalization, as `(NU_IDADE_N >= 65).astype(int)`. Order matters — the threshold applies to raw years.
8. `NU_IDADE_N` is min-max normalized to `[0, 1]` using theoretical bounds `[0, 100]`.

**Post-processing in `pre_process_df`** — rows from 2018 are dropped (incomplete surveillance year); the DataFrame is sorted ascending by `YEAR-MONTH`. Then, to prevent leakage:
- `sensitive_attribute == 'IDADE_BIN'`: drop `NU_IDADE_N` (would trivially reconstruct the sensitive attribute)
- otherwise: drop `IDADE_BIN` (not the target sensitive attribute; must not act as a predictor)

The sensitive attribute encoding is:
- `CS_RACA`: 0 = Branca/White (privileged), 1 = non-White (sensitive)
- `IDADE_BIN`: 0 = age < 65 (privileged), 1 = age ≥ 65 (sensitive)

### The four baselines (`src/main.py` → `run_baseline`)

| Baseline | Training data | Reweighing |
|----------|--------------|------------|
| `vanilla` | Batch 0 only, never retrained | None |
| `static` | Batch 0 only, never retrained | Batch 0 proportions (computed once, frozen) |
| `dynamic` | All prior batches (cumulative) | Current batch proportions |
| `abcinml` | All prior batches (cumulative) | Blend of current + forecasted future: `w = α·w_current + (1−α)·w_future` |

The `abcinml` baseline implements the **Anticipatory Bias Correction in ML** approach: it forecasts future joint distributions of (S, Y) using a rolling window average over the last `forecast_window=10` batches, then pre-emptively adjusts sample weights toward the expected future distribution. `alpha=0` means full anticipation; `alpha=1` means pure current reweighing (equivalent to `dynamic`).

The first 10 batches are skipped for `abcinml` (insufficient history to forecast).

`dynamic` and `abcinml` maintain an `accumulated_train_df` inside `run_baseline` that grows by one batch per iteration, avoiding O(N²) re-filtering of the full DataFrame. `vanilla` and `static` still go through `model_utils.get_train_and_test`.

### Key utility modules

- **`model_utils.py`**: `weight_estimation` computes reweighing factors from current batch; `weight_estimation_anticipated` does the same from forecasted proportions; `reweigh_df` blends them. `get_train_and_test` handles train/test splits for `vanilla` and `static` only. `initialize_model` creates models with `n_jobs=-1` (all cores); XGBoost uses `tree_method='hist'` for histogram-based training.
- **`metrics_utils.py`**: Snapshot metrics per batch (accuracy, AUC, demographic parity difference, TPR diff, FPR diff via `fairlearn`). Temporal metrics (maximum bias, temporal stability, max bias difference) summarize across all batches. All chart functions compute y-axis limits dynamically from the data — no hardcoded ranges. Chart generation saves to `src/figures/{sensitive_attribute}/{model_name}_alpha_{alpha_label}_window_{forecast_window}/`.
- **`general_utils.py`**: Argument parsing and state initialization for a run.

### Outputs

- Charts: `src/figures/{sensitive_attribute}/{model_name}_alpha_{alpha_label}_window_{forecast_window}/` — `model_performance_metrics.png`, `model_performance_comparison.png`, `fairness_comparison_only.png`, `shap_summary.png`, `absolute_rates_by_group.png`
- CSVs: `src/utils/results/{sensitive_attribute}/{model_name}_alpha_{alpha_label}_window_{forecast_window}/` — `assessment_all_baselines.csv`, `temporal_results_df.csv`, `absolute_results_df.csv`

Both output directories are created automatically on first run. Comparative charts (`model_performance_comparison.png`, `fairness_comparison_only.png`, `absolute_rates_by_group.png`) are only generated when `--baseline_type all` (multiple baselines).

## Code style

Ruff is configured with single-quote style (`quote-style = "single"`).
