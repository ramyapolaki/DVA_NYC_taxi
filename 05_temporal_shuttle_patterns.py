import os
import pandas as pd
import numpy as np

# --------- PATHS (LOCAL, NO SPARK) ---------
BASE_DIR = r"C:\Users\ramya\Downloads\DVA-final"
PROCESSED_DIR = os.path.join(BASE_DIR, "taxi_processed")

# Inputs
GROUPED_PATH        = os.path.join(PROCESSED_DIR, "grouped_23_25.parquet")
SPATIAL_CLUSTERS    = os.path.join(PROCESSED_DIR, "spatial_clusters_23_25.parquet")
HUB_STATS_PATH      = os.path.join(PROCESSED_DIR, "hub_strength_23_25.parquet")

# Outputs (mirroring notebook intent + Streamlit needs)
ROUTES_PATH         = os.path.join(PROCESSED_DIR, "routes_by_cluster_23_25.parquet")
METAMORPHIC_PATH    = os.path.join(PROCESSED_DIR, "metamorphic_routes_23_25.parquet")
ROUTES_FOR_MAP_PATH = os.path.join(PROCESSED_DIR, "shuttle_routes_for_map_23_25.parquet")


def main():
    # ---------------- LOAD DATA ----------------
    print("Loading grouped OD data...")
    grouped = pd.read_parquet(GROUPED_PATH)

    print("Loading spatial clusters (per time_period, PULocationID)...")
    clusters = pd.read_parquet(SPATIAL_CLUSTERS)

    print("Loading hub strength (cluster centers)...")
    hub_stats = pd.read_parquet(HUB_STATS_PATH)

    # We only need time_period, PULocationID, cluster_id from clusters
    cluster_assign = clusters[["time_period", "PULocationID", "cluster_id"]].drop_duplicates()

    # ---------------- ASSIGN CLUSTER TO EACH TRIP ----------------
    print("Assigning cluster_id to each OD pair based on pickup zone + time_period...")

    df = grouped.merge(
        cluster_assign,
        on=["time_period", "PULocationID"],
        how="left"
    )

    # Drop trips with no assigned cluster (if any)
    before = len(df)
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    print(f"  Dropped {before - len(df)} trips with no cluster_id.")

    # ---------------- BUILD CLUSTER→DESTINATION ROUTES ----------------
    print("Aggregating cluster→destination flows...")

    routes = (
        df.groupby(["time_period", "cluster_id", "DOLocationID"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "num_trips"})
    )

    print("Cluster→destination routes:", len(routes))
    print("Saving routes_by_cluster to:", ROUTES_PATH)
    routes.to_parquet(ROUTES_PATH, index=False)

    # ---------------- METAMORPHIC CORRIDORS (PIVOT + RATIOS) ----------------
    print("Computing metamorphic corridors (volume ratios across time periods)...")

    # Pivot: rows = (cluster_id, DOLocationID), columns = time_period, values = num_trips
    pivot = routes.pivot_table(
        index=["cluster_id", "DOLocationID"],
        columns="time_period",
        values="num_trips",
        aggfunc="sum",
        fill_value=0.0,
    )

    volumes = pivot.values
    max_vol = volumes.max(axis=1)
    min_nonzero = np.where(volumes > 0, volumes, np.nan).min(axis=1)
    periods_with_flow = (volumes > 0).sum(axis=1)

    # Avoid divide-by-zero issues
    ratio = np.where(min_nonzero > 0, max_vol / min_nonzero, np.nan)

    meta_df = pivot.copy()
    meta_df["max_volume"]         = max_vol
    meta_df["min_nonzero_volume"] = min_nonzero
    meta_df["volume_ratio"]       = ratio
    meta_df["periods_with_flow"]  = periods_with_flow

    # Same thresholds as notebook
    METAMORPHIC_MIN_PERIODS = 2
    METAMORPHIC_MIN_RATIO   = 3.0
    METAMORPHIC_MIN_MAXVOL  = 5000

    mask = (
        (meta_df["periods_with_flow"] >= METAMORPHIC_MIN_PERIODS) &
        (meta_df["volume_ratio"]      >= METAMORPHIC_MIN_RATIO) &
        (meta_df["max_volume"]        >= METAMORPHIC_MIN_MAXVOL)
    )

    metamorphic = meta_df[mask].copy()
    print("Metamorphic routes count:", len(metamorphic))

    # reset index so cluster_id, DOLocationID become columns
    metamorphic = metamorphic.reset_index()
    print("Saving metamorphic routes to:", METAMORPHIC_PATH)
    metamorphic.to_parquet(METAMORPHIC_PATH, index=False)

    # ---------------- BUILD STREAMLIT MAP ROUTES ----------------
    print("Preparing routes_for_map (for Streamlit)...")

    # Hub centers: one row per (time_period, cluster_id)
    hub_pdf = hub_stats[["time_period", "cluster_id", "center_lat", "center_lon"]].drop_duplicates()

    # Use all routes (not only metamorphic) for map, but we can filter on volume
    routes_map = routes.merge(
        hub_pdf,
        on=["time_period", "cluster_id"],
        how="left"
    )

    # Rename/augment for Streamlit expectations
    routes_map = routes_map.rename(columns={"num_trips": "total_trips"})
    routes_map["origin_cluster"] = routes_map["cluster_id"]

    # Rank routes within each time_period by total_trips
    routes_map["rank"] = (
        routes_map
        .groupby("time_period")["total_trips"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Optional threshold: only keep reasonably strong flows for visualization
    MIN_FLOW_FOR_MAP = 5000
    routes_map_filt = routes_map[routes_map["total_trips"] >= MIN_FLOW_FOR_MAP].copy()
    print("Routes in map output (after volume filter):", len(routes_map_filt))

    print("Saving routes_for_map to:", ROUTES_FOR_MAP_PATH)
    routes_map_filt.to_parquet(ROUTES_FOR_MAP_PATH, index=False)

    print("Done – routes_by_cluster, metamorphic_routes, and shuttle_routes_for_map have been generated.")


if __name__ == "__main__":
    main()
