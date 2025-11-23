import os
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\ramya\Downloads\DVA-final"
PROCESSED_DIR = os.path.join(BASE_DIR, "taxi_processed")
ZONES_DIR = os.path.join(BASE_DIR, "taxi_zones")

# Input: grouped data from step 01
GROUPED_PATH = os.path.join(PROCESSED_DIR, "grouped_23_25.parquet")
ZONES_SHP = os.path.join(ZONES_DIR, "taxi_zones.shp")

# Outputs
HUB_OUT = os.path.join(PROCESSED_DIR, "hub_strength_23_25.parquet")
SPATIAL_CLUSTERS_OUT = os.path.join(PROCESSED_DIR, "spatial_clusters_23_25.parquet")
HUB_CENTERS_OUT = os.path.join(PROCESSED_DIR, "hub_centers_23_25.parquet")  

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_zones():
    """Load taxi_zones shapefile and compute centroids lat/lon."""
    zones = gpd.read_file(ZONES_SHP)

    if zones.crs is None:
        zones = zones.set_crs(4326)

    # centroid in projected CRS, then back to WGS84
    zones = zones.to_crs(3857)
    zones["centroid"] = zones.geometry.centroid
    zones = zones.to_crs(4326)

    zones["latitude"] = zones["centroid"].apply(lambda x: x.y)
    zones["longitude"] = zones["centroid"].apply(lambda x: x.x)

    return zones[["LocationID", "latitude", "longitude"]]


def main():
    print("Loading grouped OD data...")
    df = pd.read_parquet(GROUPED_PATH)
    print("Rows in grouped data:", len(df))

    print("Loading taxi zones / centroids...")
    zones = load_zones()

    # total pickups per (time_period, PULocationID)
    origins = (
        df.groupby(["time_period", "PULocationID"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total_pickups"})
    )

    # attach lat/lon for each pickup zone
    origins = origins.merge(
        zones,
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )

    time_periods = origins["time_period"].dropna().unique().tolist()

    all_assignments = []  # per-zone cluster membership
    all_centers = []      # cluster centers per time_period

    for tp in time_periods:
        print(f"\n=== Clustering time_period = {tp} ===")
        subset = origins[origins["time_period"] == tp].dropna(subset=["latitude", "longitude"])

        if len(subset) < 3:
            print(f"  Skipping {tp}: not enough points ({len(subset)})")
            continue

        X = subset[["latitude", "longitude"]]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # choose k based on data size
        n_clusters = max(2, min(10, len(subset) // 4))
        print(f"  Using n_clusters = {n_clusters}, points = {len(subset)}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xs)
        subset = subset.copy()
        subset["cluster_id"] = labels

        centers_scaled = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)

        # store centers
        for cid, (lat, lon) in enumerate(centers):
            all_centers.append(
                {
                    "time_period": tp,
                    "cluster_id": cid,
                    "center_lat": lat,
                    "center_lon": lon,
                }
            )

        # store assignments (per zone)
        all_assignments.append(
            subset[["time_period", "PULocationID", "total_pickups", "cluster_id"]]
        )

    if not all_assignments:
        print("No time_period had enough points to cluster. Exiting.")
        return

    assignments = pd.concat(all_assignments, ignore_index=True)
    centers = pd.DataFrame(all_centers)

    # ---------- 1) Hub strength (same as before) ----------
    hub_strength = (
        assignments.groupby(["time_period", "cluster_id"])["total_pickups"]
        .sum()
        .reset_index()
    )

    hub_strength = hub_strength.merge(
        centers,
        on=["time_period", "cluster_id"],
        how="left"
    )

    print("\nFinal hub_strength rows:", len(hub_strength))
    print("Saving hubs to:", HUB_OUT)
    hub_strength.to_parquet(HUB_OUT, index=False)

    # ---------- 1b) Save centers separately ----------
    print("Saving hub centers to:", HUB_CENTERS_OUT)
    centers.to_parquet(HUB_CENTERS_OUT, index=False)

    # ---------- 2) Spatial clusters file (per-zone membership + center coords) ----------
    spatial_clusters = assignments.merge(
        centers,
        on=["time_period", "cluster_id"],
        how="left"
    )

    # optional: sort nicely
    spatial_clusters = spatial_clusters.sort_values(
        by=["time_period", "cluster_id", "total_pickups"], ascending=[True, True, False]
    )

    print("Final spatial_clusters rows:", len(spatial_clusters))
    print("Saving spatial clusters to:", SPATIAL_CLUSTERS_OUT)
    spatial_clusters.to_parquet(SPATIAL_CLUSTERS_OUT, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
