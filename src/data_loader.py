import pandas as pd

def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)