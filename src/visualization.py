import folium

def save_map(df, lat, lon, cluster_col, out_path):
    m = folium.Map(location=df[[lat, lon]].mean().values.tolist(), zoom_start=6)
    for _, r in df.iterrows():
        folium.CircleMarker(location=(r[lat], r[lon]), radius=3,
                            popup=f"Cluster: {r[cluster_col]}", fill=True).add_to(m)
    m.save(out_path)