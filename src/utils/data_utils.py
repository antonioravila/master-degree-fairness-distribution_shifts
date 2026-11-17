import hashlib
import inspect
import pandas as pd
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent / 'data' / 'SRAG' / 'cache'


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
                'TOMO_RES', 'CLASSI_FIN', 'SG_UF_NOT']


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
    

def _convert_types(df: pd.DataFrame, window_column: str):
    # DT_NOTIFIC: extract year, drop original
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
    if window_column == 'YEAR-MONTH':
        df[window_column] = df['DT_NOTIFIC'].dt.to_period('M')    
        df.drop(columns=['DT_NOTIFIC'], inplace=True)
    elif window_column == 'YEAR-WEEK':
        df[window_column] = df['DT_NOTIFIC'].dt.strftime("%Y-%U")    
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
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int).astype(str)
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)

    # CS_RACA: 0 = Branca (privileged), 1 = não-Branca (sensitive)
    df['CS_RACA'] = pd.to_numeric(df['CS_RACA'], errors='coerce')
    df['CS_RACA'] = df['CS_RACA'].apply(lambda x: 0 if x == 1 else 1).astype(int)

    # EVOLUCAO: keep only 1 (Cura) and 2 (Óbito)
    df['EVOLUCAO'] = pd.to_numeric(df['EVOLUCAO'], errors='coerce')
    df = df[df['EVOLUCAO'].isin([1, 2])]
    df['EVOLUCAO'] = df['EVOLUCAO'].map({1: 0, 2: 1}).astype(int)

    df = df[df['NU_IDADE_N'] < 120]

    # IDADE_BIN must be derived before normalization (threshold is 65 real years)
    # IDADE_BIN: 0 = não-idoso (<65, privileged), 1 = idoso (≥65, sensitive)
    df['IDADE_BIN'] = (df['NU_IDADE_N'] >= 65).astype(int)

    IDADE_MIN_TEORICA = 0
    IDADE_MAX_TEORICA = 100
    df['NU_IDADE_N'] = (df['NU_IDADE_N'] - IDADE_MIN_TEORICA) / (IDADE_MAX_TEORICA - IDADE_MIN_TEORICA)

    return df


def _downcast_dtypes(df: pd.DataFrame, window_column: str) -> pd.DataFrame:
    """Shrink dtypes in place: every engineered column here is either a 0/1
    flag (binary_cols, the one-hot dummies, EVOLUCAO/CS_RACA/IDADE_BIN) or a
    normalized-to-[0,1] float (NU_IDADE_N, when it isn't dropped). int8/float32
    represent both exactly — this is a memory optimization only, not a
    behavior change (sklearn/xgboost upcast internally as needed regardless
    of the input dtype). window_column (period[M] or the YEAR-WEEK string) is
    left untouched.
    """
    for col in df.columns:
        if col == window_column:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('int8')
    return df


def _filter_region(df, ):
    # Disabled for the all-regions experiments. To restrict to a subset of
    # states, add: df = df[df['SG_UF_NOT'].isin([...])]. SG_UF_NOT is still
    # dropped below regardless — it isn't one-hot encoded in _convert_types
    # and isn't meant to be a model feature.
    df.pop('SG_UF_NOT')
    return df


def _preprocessing_fingerprint() -> str:
    """Hash of the source of every function pre_process_df calls, so the disk
    cache below is invalidated automatically whenever preprocessing logic
    changes (e.g. enabling the region filter in _filter_region) — no manual cache
    version bump needed.
    """
    src = ''.join(
        inspect.getsource(fn)
        for fn in (_import_data, _filter_region, _fill_na, _convert_types, _downcast_dtypes)
    )
    return hashlib.md5(src.encode()).hexdigest()[:10]


def pre_process_df(
    sensitive_attribute: str,
    window_column: str,
    use_cache: bool = True,
) -> pd.DataFrame:

    print('0) starting data pre-process')

    cache_path = _CACHE_DIR / f'{sensitive_attribute}_{window_column.replace("-", "")}_{_preprocessing_fingerprint()}.parquet'
    if use_cache and cache_path.exists():
        print(f'\t loading cached preprocessed data from {cache_path.name}')
        df = pd.read_parquet(cache_path)
        print('\t finished data pre-process (cache hit)')
        return df

    print('\t importing data...')
    df = _import_data()

    print('\t filtering region-wise')
    df = _filter_region(df)

    print('\t filling NAs...')
    df = _fill_na(df)

    print('\t converting types...')
    df = _convert_types(df, window_column)

    df.sort_values(by=window_column, ascending=True, inplace=True)

    # Drop correlated column to prevent leakage or redundancy
    if sensitive_attribute == 'IDADE_BIN':
        # NU_IDADE_N would trivially reconstruct IDADE_BIN — drop it
        df = df.drop(columns=['NU_IDADE_N'])
    else:
        # IDADE_BIN is not the sensitive attribute; drop it so it's never a predictor
        df = df.drop(columns=['IDADE_BIN'])

    if window_column == 'YEAR-WEEK':
        df = df[df[window_column].between('2019-01', '2026-08')]
    elif window_column == 'YEAR-MONTH':
        df = df[
            df[window_column].between(
                pd.Period('2019-01', freq='M'),
                pd.Period('2026-02', freq='M')
            )
        ]

    df = _downcast_dtypes(df, window_column)

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f'\t cached preprocessed data to {cache_path.name}')

    print('\t finished data pre-process')

    return df
