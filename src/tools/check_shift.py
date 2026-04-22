from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_kdeplots(csv_path: str, columns: list[str], window_column: str) -> None:
    df = pd.read_csv(csv_path)

    csv_name = Path(csv_path).stem
    output_dir = Path('src') / 'charts' / 'distribution_shift' / csv_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in columns:
        if col not in df.columns:
            print(f'Column "{col}" not found in dataset, skipping.')
            continue

        plt.figure()
        sns.kdeplot(
            data=df,
            x=col,
            hue=window_column,
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=0,
        )
        plt.title(f'KDE Plot - {col}')

        filepath = output_dir / f'{col}.png'
        plt.savefig(filepath)
        print(f'Saved: {filepath}')


def main():
    csv_path = r'C:\master-degree-fairness-distribution_shifts\src\data\ACSEmployment\acs_employment_CA.csv'
    columns = [
        
    ]
    window_column = 'Year'

    generate_kdeplots(csv_path, columns, window_column)


if __name__ == '__main__':
    main()
