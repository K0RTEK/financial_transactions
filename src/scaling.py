from sklearn.preprocessing import RobustScaler

def scale_features(df, feature_cols, params):
    scaler = RobustScaler(**params)
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler