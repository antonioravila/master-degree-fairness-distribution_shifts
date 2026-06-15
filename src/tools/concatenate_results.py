"""
Concatenates CSV results across all subfolders of src/results/IDADE_BIN/
and writes the merged files to src/results/all/.

Folder name convention: {model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}
Example: xgboost_alpha_0.1_ema_window_10
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

# Matches: {model_name}_alpha_{alpha}_{forecasting_technique}_window_{forecast_window}
_FOLDER_RE = re.compile(r'^(.+)_alpha_[\d.]+_(.+)_window_(\d+)$')


def _parse_folder(name: str) -> dict[str, str]:
    m = _FOLDER_RE.match(name)
    if not m:
        raise ValueError(f'Folder name does not match expected pattern: {name!r}')
    return {
        'model_name': m.group(1),
        'forecasting_technique': m.group(2),
        'forecast_window': m.group(3),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in CSV_FILES}

    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        meta = _parse_folder(folder.name)
        for csv_name in CSV_FILES:
            csv_path = folder / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['model_name'] = meta['model_name']
                df['forecasting_technique'] = meta['forecasting_technique']
                df['forecast_window'] = meta['forecast_window']
                buckets[csv_name].append(df)
            else:
                print(f'Warning: {csv_path} not found, skipping.')

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
