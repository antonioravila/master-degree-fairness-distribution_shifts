from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_kdeplots(csv_path: str, columns: list[str]) -> None:
    df = pd.read_csv(csv_path)

    csv_name = Path(csv_path).stem
    output_dir = Path('src') / 'charts' / csv_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in columns:
        if col not in df.columns:
            print(f'Column "{col}" not found in dataset, skipping.')
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df, x=col, ax=ax, fill=True)
        ax.set_title(f'KDE Plot - {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')

        filepath = output_dir / f'{col}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {filepath}')


def main():
    csv_path = r'src/data/ACSIncome/acs_income_NE.csv'
    columns = ['AGEP', 'SEX', 'PINCP']

    generate_kdeplots(csv_path, columns)


if __name__ == '__main__':
    main()
