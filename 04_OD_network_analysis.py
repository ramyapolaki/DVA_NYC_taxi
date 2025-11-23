# 03_OD_network_analysis.py

import os
import pandas as pd

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "taxi_processed")

GROUPED_CLUSTERED = os.path.join(PROCESSED_DIR, "grouped_clustered_23_25.parquet")
OD_OUT = os.path.join(PROCESSED_DIR, "od_network_23_25.parquet")
OD_BY_HOUR_OUT = os.path.join(PROCESSED_DIR, "od_by_hour_ready.parquet")


def hour_bin_coarse(h: int) -> str:
    """Coarse bins used for OD network (morning/midday/evening/night)."""
    if 6 <= h < 10:
        return "morning"
    if 10 <= h < 16:
        return "midday"
    if 16 <= h < 20:
        return "evening"
    return "night"


def hour_bin_00_24(h: int) -> str:
    """Broader 6-hour bins used for time-profile page."""
    if 0 <= h < 6:
        return "00-06"
    if 6 <= h < 12:
        return "06-12"
    if 12 <= h < 18:
        return "12-18"
    return "18-24"


def main():
    print("Loading clustered grouped data from:", GROUPED_CLUSTERED)
    df = pd.read_parquet(GROUPED_CLUSTERED)

    # Expected columns in grouped_clustered_23_25:
    # trip_date, pickup_hour, pickup, dropoff, is_public_holiday,
    # count, pickup_latitude, pickup_longitude,
    # dropoff_latitude, dropoff_longitude, Clustering

    print("Raw rows in clustered data:", len(df))

    # Use only one clustering algorithm (avoid double-counting)
    if "Clustering" in df.columns:
        df = df[df["Clustering"] == "kmeans"].copy()
        print("Rows after filtering to kmeans:", len(df))

    # Make sure types are clean
    df["pickup_hour"] = df["pickup_hour"].astype(int)
    df["pickup"] = df["pickup"].astype(int)
    df["dropoff"] = df["dropoff"].astype(int)

    # ---------------- OD NETWORK (morning/midday/evening/night) ----------------
    print("Computing OD network (coarse time-of-day bins)...")
    df["hour_bin"] = df["pickup_hour"].apply(hour_bin_coarse)

    od = (
        df.groupby(["pickup", "dropoff", "hour_bin"], as_index=False)["count"]
        .sum()
    )

    od_pivot = od.pivot_table(
        index=["pickup", "dropoff"],
        columns="hour_bin",
        values="count",
        fill_value=0,
    ).reset_index()

    # Ensure all expected columns exist
    for col in ["morning", "midday", "evening", "night"]:
        if col not in od_pivot.columns:
            od_pivot[col] = 0

    od_pivot["total_trips"] = (
        od_pivot["morning"]
        + od_pivot["midday"]
        + od_pivot["evening"]
        + od_pivot["night"]
    )

    print("OD network rows:", len(od_pivot))
    print("Saving OD network to:", OD_OUT)
    od_pivot.to_parquet(OD_OUT, index=False)

    # ---------------- OD BY HOUR (for time-profile page) ----------------
    print("Computing OD-by-hour profiles (00-06 / 06-12 / 12-18 / 18-24)...")
    df["hour_bin_6h"] = df["pickup_hour"].apply(hour_bin_00_24)

    od_hour = (
        df.groupby(["pickup", "dropoff", "hour_bin_6h"], as_index=False)["count"]
        .sum()
        .rename(columns={"hour_bin_6h": "hour_bin", "count": "trip_count"})
    )

    print("OD-by-hour rows:", len(od_hour))
    print("Saving OD-by-hour profiles to:", OD_BY_HOUR_OUT)
    od_hour.to_parquet(OD_BY_HOUR_OUT, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
