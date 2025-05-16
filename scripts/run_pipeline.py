import argparse, logging, json, os
from src.data_loader import load_csv
from src.preprocessing import sort_and_diff
from src.features import rolling_counts, geo_features
from src.scaling import scale_features
from src.model import AnomalyAutoencoder
from src.clustering import cluster_geo
from src.visualization import save_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--map', default='map.html')
    parser.add_argument('--model', default='processed/model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = json.load(open(args.config))

    df = load_csv(args.input, parse_dates=[cfg['time_col']])
    df = sort_and_diff(df, cfg['time_col'], cfg['group_cols'])
    df = rolling_counts(df, cfg['time_col'], cfg['group_cols'], cfg['rolling_windows'])
    df = geo_features(df, cfg['lat_col'], cfg['lon_col'], cfg['group_cols'])
    df = cluster_geo(df, cfg['lat_col'], cfg['lon_col'], cfg)

    features = cfg['feature_cols']
    X, scaler = scale_features(df, features, cfg.get('scaler_params', {}))
    ae = AnomalyAutoencoder(X.shape[1], cfg['encoding_dim'], cfg['autoencoder'])
    ae.train(X, cfg['validation_split'])

    # сохраняем модель
    model_dir = args.model
    os.makedirs(model_dir, exist_ok=True)
    ae.save(os.path.join(model_dir, 'autoencoder'))

    # детектируем и сохраняем аномалии
    mse, anoms = ae.detect(X, cfg['mse_threshold'])
    df['mse'], df['is_anomaly'] = mse, anoms
    anomalies = df[df['is_anomaly']]
    anomalies.to_csv(os.path.join('data/processed', 'anomalies.csv'), index=False)

    # визуализация
    save_map(df, cfg['lat_col'], cfg['lon_col'], 'geo_cluster', args.map)

    # сохраняем полный набор
    df.to_csv(args.output, index=False)
    logging.info("Pipeline completed. Model and anomalies saved.")

if __name__ == '__main__':
    main()