"""
Concatenates CSV results across all subfolders of src/results/IDADE_BIN/
and writes the merged files to src/results/all/.

Handles two folder naming conventions:
  baseline_non_abc_{model_name}_{batch_type}
  {model_name}_alpha_{alpha}_{technique}_window_{forecast_window}_{batch_type}[_ema_{alpha_ema}]
"""

import re
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / 'results'
SOURCE_DIR = RESULTS_ROOT / 'IDADE_BIN'
OUTPUT_DIR = RESULTS_ROOT / 'all'

CSV_FILES = [
    'absolute_results_df.csv',
    'assessment_all_baselines.csv',
    'temporal_results_df.csv',
]

_ABC_RE = re.compile(
    r'^(?P<model>.+)_alpha_(?P<alpha>[\d.]+)_(?P<technique>.+)_window_(?P<window>\d+)_(?P<batch>monthly|weekly)(?:_ema_(?P<alpha_ema>[\d.]+))?(?:_(?P<fairness_correction>kamiran_rw|adaptative_rw))?(?:_trainwindow_(?P<train_window>\w+))?$'
)
# vanilla/static (never train_window-dependent) and dynamic (own folder, always
# train_window-dependent) share this
# shape except for the prefix and the optional trainwindow suffix.
_BASELINE_RE = re.compile(
    # train_window is [^_]+, not \w+: \w matches underscores, so it would greedily
    # swallow the _fw_{n} suffix that follows it ('none_fw_5') and leave the
    # forecast_window group empty. Values are 3/6/12/none — never underscored.
    # forecast_window is required: legacy folders without _fw_ are skipped, not
    # merged, so a restored pre-migration folder can't duplicate rows here.
    r'^baseline_(?:non_abc|dynamic)_(?P<model>.+)_(?P<batch>monthly|weekly)(?:_(?P<fairness_correction>kamiran_rw|adaptative_rw))?(?:_trainwindow_(?P<train_window>[^_]+))?_fw_(?P<forecast_window>\d+)$'
)


def _parse_folder(name: str) -> dict | None:
    m = _ABC_RE.match(name)
    if m:
        return {
            'model_name': m.group('model'),
            'alpha': m.group('alpha'),
            'forecasting_technique': m.group('technique'),
            'forecast_window': m.group('window'),
            'batch_type': m.group('batch'),
            'alpha_ema': m.group('alpha_ema'),
            'fairness_correction': m.group('fairness_correction'),
            'train_window': m.group('train_window'),
        }
    m = _BASELINE_RE.match(name)
    if m:
        return {
            'model_name': m.group('model'),
            'alpha': None,
            'forecasting_technique': None,
            'forecast_window': m.group('forecast_window'),
            'batch_type': m.group('batch'),
            'alpha_ema': None,
            'fairness_correction': m.group('fairness_correction'),
            'train_window': m.group('train_window'),
        }
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in CSV_FILES}
    skipped = []

    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        meta = _parse_folder(folder.name)
        if meta is None:
            skipped.append(folder.name)
            continue
        for csv_name in CSV_FILES:
            csv_path = folder / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for key, val in meta.items():
                    if val is not None:
                        df[key] = val
                buckets[csv_name].append(df)
            else:
                print(f'Warning: {csv_path} not found, skipping.')

    if skipped:
        print(f'Skipped {len(skipped)} unrecognized folder(s): {skipped}')

    for csv_name, frames in buckets.items():
        if not frames:
            print(f'No data collected for {csv_name}, skipping output.')
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = OUTPUT_DIR / csv_name
        combined.to_csv(out_path, index=False)
        print(f'Saved {len(combined)} rows -> {out_path}')


if __name__ == '__main__':
    main()
