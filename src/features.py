import numpy as np
import pandas as pd

def rolling_counts(df: pd.DataFrame, time_col: str, group_cols: list, windows: list) -> pd.DataFrame:
    # создаём признак count_{w} — число событий за последние w минут
    df = df.set_index(time_col)
    for w in windows:
        col = f'count_{w}'
        df[col] = df.groupby(group_cols)['transactionstatus'].rolling(f'{w}T').count().reset_index(level=group_cols, drop=True)
    return df.reset_index()

def geo_features(df: pd.DataFrame, lat_col: str, lon_col: str, group_cols: list) -> pd.DataFrame:
    # флаг наличия координат
    df['has_coords'] = df[lat_col].notna() & df[lon_col].notna()
    # вычисляем расстояние между точками (м)
    coords = df[[lat_col, lon_col]].fillna(0).to_numpy()
    prev = df.groupby(group_cols)[[lat_col, lon_col]].shift().fillna(method='bfill').fillna(0).to_numpy()
    df['distance_prev'] = np.linalg.norm(coords - prev, axis=1)
    # скорость м/с -> км/ч
    df['speed_kmh'] = df['distance_prev'] / df['time_diff_prev'].replace(0, np.nan) * 3.6
    df['speed_kmh'] = df['speed_kmh'].fillna(0)
    return df