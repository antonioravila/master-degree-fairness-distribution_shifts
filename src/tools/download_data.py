import os
import pandas as pd
from folktables import ACSDataSource, ACSIncome


def extract_and_save_acs_data(
    years: list[int], state: str, target_path: str = 'src/data/csv'
):
    os.makedirs(target_path, exist_ok=True)

    all_years_data = []

    for year in years:
        print(f'--- Processing Year: {year} ---')
        try:
            data_source = ACSDataSource(
                survey_year=year, horizon='1-Year', survey='person'
            )
            acs_data = data_source.get_data(states=[state], download=True)

            features, label, _ = ACSIncome.df_to_pandas(acs_data)

            year_df = pd.concat([features, label], axis=1)

            year_df['Year'] = year

            all_years_data.append(year_df)
            print(f'Successfully processed {len(year_df)} records for {year}.')

        except Exception as e:
            print(f'Error downloading/processing year {year}: {e}')

    try:
        final_df = pd.concat(all_years_data, axis=0, ignore_index=True)

        filename = f'acs_income_{state}.csv'
        file_path = os.path.join(target_path, filename)

        final_df.to_csv(file_path, index=False)
        print('-' * 30)
        print(f'Success! Saved total of {len(final_df)} records to: {file_path}')
    except Exception as e:
        print(f'Error saving csv data: {e}')


def main():
    extract_and_save_acs_data(years=[2014, 2015, 2016, 2017, 2018], state='NE')


if __name__ == '__main__':
    main()
