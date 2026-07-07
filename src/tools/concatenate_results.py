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
    r'^(?P<model>.+)_alpha_(?P<alpha>[\d.]+)_(?P<technique>.+)_window_(?P<window>\d+)_(?P<batch>monthly|weekly)(?:_ema_(?P<alpha_ema>[\d.]+))?$'
)
_BASELINE_RE = re.compile(
    r'^baseline_non_abc_(?P<model>.+)_(?P<batch>monthly|weekly)$'
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
        }
    m = _BASELINE_RE.match(name)
    if m:
        return {
            'model_name': m.group('model'),
            'alpha': None,
            'forecasting_technique': None,
            'forecast_window': None,
            'batch_type': m.group('batch'),
            'alpha_ema': None,
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
