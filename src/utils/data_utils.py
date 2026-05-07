import pandas as pd
from pathlib import Path


def _import_data():
    df = pd.DataFrame()
    columns = [
                'DT_NOTIFIC', 'EVOLUCAO', 'CS_RACA', 
                'NU_IDADE_N',
                'CS_GESTANT',
                'NOSOCOMIAL', 'AVE_SUINO',
                'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 
                'DESC_RESP', 'SATURACAO', 'DIARREIA', 
                'VOMITO', 'DOR_ABD', 'FADIGA', 'PERD_OLFT', 
                'PERD_PALA', 'OUTRO_SIN', 'FATOR_RISC', 
                'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 
                'SIND_DOWN', 'HEPATICA', 'ASMA', 'DIABETES', 
                'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 
                'OBESIDADE',
                'VACINA_COV', 'ANTIVIRAL', 'TP_ANTIVIR',
                'UTI', 'SUPORT_VEN', 'RAIOX_RES', 
                'TOMO_RES', 'CLASSI_FIN']


    for path in Path(r'src\data\SRAG\raw').iterdir():
        df_raw = pd.read_parquet(path, columns = columns)
        df = pd.concat([df, df_raw])

    return df


def _fill_na(df):
    df = df.dropna(subset=['EVOLUCAO', 'CS_RACA'])

    cols_fill2 = [
        'ANTIVIRAL', 'ASMA', 'AVE_SUINO', 'CARDIOPATI', 'DESC_RESP', 
        'DIABETES', 'DIARREIA', 'DISPNEIA', 'DOR_ABD', 'FADIGA', 'FATOR_RISC', 'FEBRE', 
        'GARGANTA', 'HEMATOLOGI', 'HEPATICA', 'IMUNODEPRE', 'NEUROLOGIC', 'NOSOCOMIAL', 
        'OBESIDADE', 'OUTRO_SIN', 'PERD_OLFT', 'PERD_PALA', 'PNEUMOPATI', 'PUERPERA', 
        'RENAL', 'SATURACAO', 'SIND_DOWN', 'TOSSE', 'TP_ANTIVIR', 'UTI', 'VACINA_COV', 'VOMITO'
        ]

    df[cols_fill2] = df[cols_fill2].fillna(2).replace(9, 2).replace('9', 2)
    df['SUPORT_VEN'] = df['SUPORT_VEN'].fillna(3).replace(9, 3).replace('9', 3)
    df[['TOMO_RES', 'RAIOX_RES']] = df[['TOMO_RES', 'RAIOX_RES']].fillna(6).replace(9, 6).replace('9', 6)
    df['CLASSI_FIN'] = df['CLASSI_FIN'].fillna(4)

    df = df[df['CS_RACA'] != 9]
    df = df[df['CS_GESTANT'] != 9]

    return df
    

def _convert_types(df):
    # DT_NOTIFIC: extract year, drop original
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
    df['YEAR-MONTH'] = df['DT_NOTIFIC'].dt.to_period('M')    
    df.drop(columns=['DT_NOTIFIC'], inplace=True)

    # NU_IDADE_N: numeric, fill NaN with 0
    df['NU_IDADE_N'] = pd.to_numeric(df['NU_IDADE_N'], errors='coerce')

    # Binary 1/2 columns: 1→1, 2→0, NaN→0
    binary_cols = [
        'ANTIVIRAL', 'ASMA', 'CARDIOPATI', 'DESC_RESP', 'DIABETES', 'DIARREIA', 
        'DISPNEIA', 'DOR_ABD', 'FADIGA', 'FATOR_RISC', 'FEBRE', 'GARGANTA', 
        'HEMATOLOGI', 'HEPATICA', 'IMUNODEPRE', 'NEUROLOGIC', 'NOSOCOMIAL', 
        'OBESIDADE', 'OUTRO_SIN', 'PERD_OLFT', 'PERD_PALA', 'PNEUMOPATI', 
        'PUERPERA', 'RENAL', 'SATURACAO', 'SIND_DOWN', 'TOSSE', 'UTI', 
        'VACINA_COV', 'VOMITO'
        ]

    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = (df[col] == 1).astype(int)

    # Multi-category columns: cast to string, one-hot encode
    multi_cat_cols = [
        'AVE_SUINO', 'CS_GESTANT', 'TP_ANTIVIR',
        'SUPORT_VEN', 'RAIOX_RES', 'TOMO_RES', 'CLASSI_FIN',
    ]
    for col in multi_cat_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)

    # CS_RACA: binarize 1 (Branca)→0, others→1
    df['CS_RACA'] = pd.to_numeric(df['CS_RACA'], errors='coerce')
    df['CS_RACA'] = df['CS_RACA'].apply(lambda x: 0 if x == 1 else 1).astype(int)

    # EVOLUCAO: keep only 1 (Cura) and 2 (Óbito)
    df['EVOLUCAO'] = pd.to_numeric(df['EVOLUCAO'], errors='coerce')
    df = df[df['EVOLUCAO'].isin([1, 2])]
    df['EVOLUCAO'] = df['EVOLUCAO'].map({1: 0, 2: 1}).astype(int)

    df = df[df['NU_IDADE_N'] < 120]
    IDADE_MIN_TEORICA = 0
    IDADE_MAX_TEORICA = 120
    df['NU_IDADE_N'] = (df['NU_IDADE_N'] - IDADE_MIN_TEORICA) / (IDADE_MAX_TEORICA - IDADE_MIN_TEORICA)

    return df


def pre_process_df(
    sensitive_attribute: str,
    sensitive_value: str | int,
    favored_value: str | int,
) -> pd.DataFrame:

    print('0) starting data pre-process')

    df = _import_data()
    df = _fill_na(df)
    df = _convert_types(df)
    
    df = df[df['YEAR-MONTH'].dt.year != 2018]
    df.sort_values(by='YEAR-MONTH', ascending=True, inplace=True)

    return df
