import os
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\ramya\Downloads\DVA-final"
PROCESSED_DIR = os.path.join(BASE_DIR, "taxi_processed")
ZONES_DIR = os.path.join(BASE_DIR, "taxi_zones")

GROUPED_PATH = os.path.join(PROCESSED_DIR, "grouped_23_25.parquet")
ZONES_SHP = os.path.join(ZONES_DIR, "taxi_zones.shp")

OUTPUT_CLUSTERED = os.path.join(PROCESSED_DIR, "grouped_clustered_23_25.parquet")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------- HELPERS ----------------

def load_zone_centroids() -> pd.DataFrame:
    """Always return centroids in WGS84 lat/lon, never projected coordinates."""
    print(f"Reading zones from: {ZONES_SHP}")
    zones = gpd.read_file(ZONES_SHP)

    # Force CRS if missing
    if zones.crs is None:
        print("Shapefile missing CRS, forcing EPSG:4326 (WGS84)")
        zones = zones.set_crs(4326)

    # ---- STEP 1: Project to 3857 ONLY TO COMPUTE CENTROID ----
    zones_proj = zones.to_crs(3857)
    zones_proj["centroid"] = zones_proj.geometry.centroid

    # ---- STEP 2: Convert centroid BACK TO WGS84 ----
    centroid_wgs = zones_proj.set_geometry("centroid").to_crs(4326)

    # Extract final lat/lon in degrees
    centroid_wgs["latitude"] = centroid_wgs.geometry.y
    centroid_wgs["longitude"] = centroid_wgs.geometry.x

    # Keep only required columns
    out = centroid_wgs[["LocationID", "latitude", "longitude"]].dropna()

    # Make sure LocationID is int
    out["LocationID"] = out["LocationID"].astype(int)

    print("Sample centroid output:")
    print(out.head())

    return out


def compute_kmeans_clusters(
    vol_df: pd.DataFrame,
    id_col: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    vol_col: str = "volume",
    n_clusters: int = 15,
    label_name: str = "cluster"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KMeans on zone centroids (weighted by volume).

    Returns:
      - assignments: [id_col, label_name]
      - centers: [label_name, center_lat, center_lon]
    """
    df = vol_df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        raise RuntimeError("No valid coordinates for KMeans clustering.")

    X = df[[lat_col, lon_col]].to_numpy()
    weights = df[vol_col].to_numpy()

    # Make sure we don't ask for more clusters than points
    k = min(n_clusters, len(df))
    if k < 2:
        k = 2
    print(f"  KMeans: {label_name}, points={len(df)}, n_clusters={k}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs, sample_weight=weights)

    # Assign cluster labels per zone id
    df_assign = df[[id_col]].copy()
    df_assign[label_name] = labels.astype(int)

    # Compute centroids back in ORIGINAL lat/lon space (not scaled)
    centers_scaled = km.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)

    centers = pd.DataFrame(centers_unscaled, columns=["center_lat", "center_lon"])
    centers[label_name] = range(len(centers))

    return df_assign, centers


def compute_dbscan_clusters(
    vol_df: pd.DataFrame,
    id_col: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    vol_col: str = "volume",
    eps: float = 0.08,
    min_samples: int = 10,
    label_name: str = "cluster"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DBSCAN on zone centroids.

    SAFE VERSION: 
    - Never raises error.
    - If DBSCAN finds only noise, falls back to a single cluster 0 using mean lat/lon.
    """
    df = vol_df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        # Fallback: one global cluster if no coords
        df[label_name] = 0
        centers = pd.DataFrame([{
            label_name: 0,
            "center_lat": df[lat_col].mean(),
            "center_lon": df[lon_col].mean()
        }])
        assignments = df[[id_col, label_name]].copy()
        return assignments, centers

    X = df[[lat_col, lon_col]].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Xs)

    df[label_name] = labels.astype(int)

    # If all noise → fallback to single cluster
    if (df[label_name] == -1).all():
        print("  DBSCAN: all points were noise → fallback to one cluster")
        df[label_name] = 0
        centers = pd.DataFrame([{
            label_name: 0,
            "center_lat": df[lat_col].mean(),
            "center_lon": df[lon_col].mean()
        }])
        assignments = df[[id_col, label_name]].copy()
        return assignments, centers

    # Keep only non-noise clusters
    df_clusters = df[df[label_name] >= 0]

    # Weighted cluster centers in ORIGINAL (unscaled) lat/lon
    centers = (
        df_clusters
        .groupby(label_name)
        .apply(lambda g: pd.Series({
            "center_lat": np.average(g[lat_col], weights=g[vol_col]),
            "center_lon": np.average(g[lon_col], weights=g[vol_col]),
        }))
        .reset_index()
    )

    assignments = df_clusters[[id_col, label_name]].copy()
    return assignments, centers


def build_clustered_flows(
    grouped: pd.DataFrame,
    pickup_assign: pd.DataFrame,
    dropoff_assign: pd.DataFrame,
    pickup_centers: pd.DataFrame,
    dropoff_centers: pd.DataFrame,
    pickup_zone_col: str,
    dropoff_zone_col: str,
    pickup_cluster_col: str,
    dropoff_cluster_col: str,
    clustering_name: str
) -> pd.DataFrame:
    """
    Aggregate grouped data to cluster→cluster flows and attach cluster center coordinates.
    """
    print(f"\nBuilding clustered flows for: {clustering_name}")

    # Join cluster labels to each OD row
    df = (
        grouped
        .merge(
            pickup_assign.rename(columns={pickup_zone_col: pickup_zone_col}),
            on=pickup_zone_col,
            how="left",
        )
        .merge(
            dropoff_assign.rename(columns={dropoff_zone_col: dropoff_zone_col}),
            on=dropoff_zone_col,
            how="left",
        )
    )

    # Drop rows without cluster assignments
    df = df.dropna(subset=[pickup_cluster_col, dropoff_cluster_col])
    df[pickup_cluster_col] = df[pickup_cluster_col].astype(int)
    df[dropoff_cluster_col] = df[dropoff_cluster_col].astype(int)

    print(f"  Rows with both clusters: {len(df)}")

    # Aggregate by date, hour, pickup cluster, dropoff cluster, holiday flag
    grouped_clusters = (
        df.groupby(
            ["trip_date", "pickup_hour", pickup_cluster_col,
             dropoff_cluster_col, "is_public_holiday"],
            as_index=False
        )["count"]
        .sum()
    )

    # Attach pickup cluster centers
    pickup_centers_ren = pickup_centers.rename(columns={
        "center_lat": "pickup_latitude",
        "center_lon": "pickup_longitude",
        pickup_cluster_col: pickup_cluster_col,
    })
    grouped_clusters = grouped_clusters.merge(
        pickup_centers_ren,
        on=pickup_cluster_col,
        how="left",
    )

    # Attach dropoff cluster centers
    dropoff_centers_ren = dropoff_centers.rename(columns={
        "center_lat": "dropoff_latitude",
        "center_lon": "dropoff_longitude",
        dropoff_cluster_col: dropoff_cluster_col,
    })
    grouped_clusters = grouped_clusters.merge(
        dropoff_centers_ren,
        on=dropoff_cluster_col,
        how="left",
    )

    # Final column names
    grouped_clusters = grouped_clusters.rename(columns={
        pickup_cluster_col: "pickup",
        dropoff_cluster_col: "dropoff",
    })
    grouped_clusters["Clustering"] = clustering_name

    cols = [
        "trip_date",
        "pickup_hour",
        "pickup",
        "dropoff",
        "is_public_holiday",
        "count",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "Clustering",
    ]
    grouped_clusters = grouped_clusters[cols]

    print(f"  Clustered rows for {clustering_name}: {len(grouped_clusters)}")
    return grouped_clusters


# ---------------- MAIN ----------------

def main():
    print(f"Reading grouped data from: {GROUPED_PATH}")
    grouped = pd.read_parquet(GROUPED_PATH)
    print(f"Rows in grouped data: {len(grouped)}")

    print("Loading taxi zones and computing centroids...")
    zones = load_zone_centroids()

    # ---------------- Aggregate volume per zone ----------------
    pickup_vol = (
        grouped.groupby("PULocationID")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "volume"})
    )
    dropoff_vol = (
        grouped.groupby("DOLocationID")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "volume"})
    )

    # Attach lat/lon to volume tables
    pickup_vol = pickup_vol.merge(
        zones.rename(columns={"LocationID": "PULocationID"}),
        on="PULocationID",
        how="left",
    )
    dropoff_vol = dropoff_vol.merge(
        zones.rename(columns={"LocationID": "DOLocationID"}),
        on="DOLocationID",
        how="left",
    )

    # ---------------- KMEANS (pickup & dropoff) ----------------
    print("\nRunning KMeans clustering for pickup zones...")
    pick_assign_km, pick_centers_km = compute_kmeans_clusters(
        pickup_vol,
        id_col="PULocationID",
        lat_col="latitude",
        lon_col="longitude",
        vol_col="volume",
        label_name="kmeans_cluster_pickup",
    )

    print("Running KMeans clustering for dropoff zones...")
    drop_assign_km, drop_centers_km = compute_kmeans_clusters(
        dropoff_vol,
        id_col="DOLocationID",
        lat_col="latitude",
        lon_col="longitude",
        vol_col="volume",
        label_name="kmeans_cluster_dropoff",
    )

    # ---------------- DBSCAN (pickup & dropoff) ----------------
    print("\nRunning DBSCAN clustering for pickup zones...")
    pick_assign_db, pick_centers_db = compute_dbscan_clusters(
        pickup_vol,
        id_col="PULocationID",
        lat_col="latitude",
        lon_col="longitude",
        vol_col="volume",
        label_name="dbscan_cluster_pickup",
    )

    print("Running DBSCAN clustering for dropoff zones...")
    drop_assign_db, drop_centers_db = compute_dbscan_clusters(
        dropoff_vol,
        id_col="DOLocationID",
        lat_col="latitude",
        lon_col="longitude",
        vol_col="volume",
        label_name="dbscan_cluster_dropoff",
    )

    # ---------------- Build final grouped_clustered ----------------
    grouped_kmeans = build_clustered_flows(
        grouped=grouped,
        pickup_assign=pick_assign_km,
        dropoff_assign=drop_assign_km,
        pickup_centers=pick_centers_km,
        dropoff_centers=drop_centers_km,
        pickup_zone_col="PULocationID",
        dropoff_zone_col="DOLocationID",
        pickup_cluster_col="kmeans_cluster_pickup",
        dropoff_cluster_col="kmeans_cluster_dropoff",
        clustering_name="kmeans",
    )

    grouped_dbscan = build_clustered_flows(
        grouped=grouped,
        pickup_assign=pick_assign_db,
        dropoff_assign=drop_assign_db,
        pickup_centers=pick_centers_db,
        dropoff_centers=drop_centers_db,
        pickup_zone_col="PULocationID",
        dropoff_zone_col="DOLocationID",
        pickup_cluster_col="dbscan_cluster_pickup",
        dropoff_cluster_col="dbscan_cluster_dropoff",
        clustering_name="dbscan",
    )

    grouped_clustered = pd.concat([grouped_kmeans, grouped_dbscan], ignore_index=True)
    print(f"\nTotal rows in grouped_clustered: {len(grouped_clustered)}")

    print(f"Writing clustered flows to: {OUTPUT_CLUSTERED}")
    grouped_clustered.to_parquet(OUTPUT_CLUSTERED, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
