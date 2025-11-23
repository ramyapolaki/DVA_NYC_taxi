import os
import glob
import holidays
import pandas as pd
import numpy as np

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = r"C:\Users\ramya\Downloads\DVA-final"
RAW_DIR = os.path.join(BASE_DIR, "taxi_raw", "taxt_data_23_25")
PROCESSED_DIR = os.path.join(BASE_DIR, "taxi_processed")
GROUPED_PATH = os.path.join(PROCESSED_DIR, "grouped_23_25.parquet")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# US holidays for 2023–2025
us_holidays = holidays.US(years=[2023, 2024, 2025])

def is_holiday_series(dates: pd.Series) -> pd.Series:
    """Vectorized holiday flag for a pandas Series of dates."""
    d = pd.to_datetime(dates).dt.date
    return d.map(lambda x: x in us_holidays)

def classify_time_period(hour_series: pd.Series) -> pd.Series:
    h = hour_series
    return np.select(
        [
            (h >= 5) & (h < 9),
            (h >= 9) & (h < 12),
            (h >= 12) & (h < 14),
            (h >= 14) & (h < 17),
            (h >= 17) & (h < 20),
            (h >= 20) & (h < 23),
        ],
        [
            "morning_rush",
            "late_morning",
            "lunch",
            "afternoon",
            "evening_rush",
            "evening",
        ],
        default="night",
    )

def process_one_file(path: str) -> pd.DataFrame:
    print(f"\n=== Reading: {path} ===")
    df = pd.read_parquet(path)

    required_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "trip_distance",
        "passenger_count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    mask = (
        (df["fare_amount"] > 0) &
        (df["trip_distance"] > 0) &
        (df["passenger_count"] > 0) &
        (df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"])
    )
    df = df.loc[mask].copy()

    df["trip_date"] = df["tpep_pickup_datetime"].dt.date

    # ------------------------------------------------------------------
    #  🚨 NEW RULE — DROP IF YEAR NOT IN 2023 / 2024 / 2025
    # ------------------------------------------------------------------
    valid_years = {2023, 2024, 2025}
    df = df[df["trip_date"].map(lambda d: d.year in valid_years)]
    # ------------------------------------------------------------------

    # Time features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day"] = df["tpep_pickup_datetime"].dt.dayofweek

    df["is_public_holiday"] = is_holiday_series(df["trip_date"])

    df["day_type"] = np.select(
        [
            df["is_public_holiday"],
            df["pickup_day"].isin([5, 6]),
        ],
        [
            "holiday",
            "weekend",
        ],
        default="weekday",
    )

    df["time_period"] = classify_time_period(df["pickup_hour"])

    grouped = (
        df.groupby(
            [
                "trip_date",
                "pickup_hour",
                "PULocationID",
                "DOLocationID",
                "pickup_day",
                "is_public_holiday",
                "day_type",
                "time_period",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
    )

    return grouped

def main():
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not all_files:
        print(f"No parquet files found in {RAW_DIR}")
        return

    print("Found raw files:")
    for f in all_files:
        print("  ", f)

    grouped_list = []

    for path in all_files:
        g = process_one_file(path)
        print(f"  -> grouped rows: {len(g)}")
        grouped_list.append(g)

    print("\nCombining all months...")
    all_grouped = pd.concat(grouped_list, ignore_index=True)

    print("Aggregating across all months...")
    final = (
        all_grouped
        .groupby(
            [
                "trip_date",
                "pickup_hour",
                "PULocationID",
                "DOLocationID",
                "pickup_day",
                "is_public_holiday",
                "day_type",
                "time_period",
            ],
            as_index=False,
        )["count"]
        .sum()
    )

    print("\nFinal row count:", len(final))
    print("Writing to:", GROUPED_PATH)
    final.to_parquet(GROUPED_PATH, engine="pyarrow", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
