import hdbscan

def cluster_geo(df, lat, lon, cfg):
    coords = df[[lat, lon]].fillna(0).to_numpy()
    cl = hdbscan.HDBSCAN(min_cluster_size=cfg['min_cluster_size'], min_samples=cfg['min_samples'])
    df['geo_cluster'] = cl.fit_predict(coords)
    return df