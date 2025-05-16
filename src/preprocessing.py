import pandas as pd

def sort_and_diff(df: pd.DataFrame, time_col: str, group_cols: list) -> pd.DataFrame:
    df = df.sort_values(group_cols + [time_col])
    df['time_diff_prev'] = df.groupby(group_cols)[time_col].diff().dt.total_seconds().fillna(0)
    return df