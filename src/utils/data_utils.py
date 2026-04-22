import pandas as pd

def pre_process_df(
    df: pd.DataFrame,
    sensitive_attribute: str,
    sensitive_value: str | int,
    favored_value: str | int,
) -> pd.DataFrame:
    df[sensitive_attribute] = df[sensitive_attribute].replace(
        {sensitive_value: 1, favored_value: 0}
    )
    return df