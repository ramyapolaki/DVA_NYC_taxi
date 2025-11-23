from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "taxi_processed"


@st.cache_data(show_spinner=False)
def load_metamorphic_routes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "metamorphic_routes_23_25.parquet")


def app(colors=None):
    st.header("Shuttle patterns")

    df = load_metamorphic_routes()

    if df.empty:
        st.info("No routes found in the processed data.")
        return

    st.markdown("**Top shuttle patterns by volume ratio**")

    top_n = st.slider("Show top N", 5, 100, 20)
    df_top = df.sort_values("volume_ratio", ascending=False).head(top_n)

    fig_ratio = px.bar(
        df_top,
        x="volume_ratio",
        y=df_top.index.astype(str),
        orientation="h",
        hover_data=[
            "cluster_id",
            "DOLocationID",
            "max_volume",
            "min_nonzero_volume",
            "periods_with_flow",
        ],
        labels={"y": "route index", "volume_ratio": "max_volume / min_nonzero_volume"},
        title="routes ranked by volume ratio",
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("### Inspect a single metamorphic route")
    route_idx = st.selectbox("Pick route index", df.index)
    row = df.loc[route_idx]

    # Display route details in a formatted way with visible text
    route_data = {
        "cluster_id": int(row["cluster_id"]),
        "DOLocationID": int(row["DOLocationID"]),
        "max_volume": int(row["max_volume"]),
        "min_nonzero_volume": float(row["min_nonzero_volume"]),
        "volume_ratio": float(row["volume_ratio"]),
        "periods_with_flow": int(row["periods_with_flow"]),
    }
    
    # Use columns for better visibility and formatting
    st.markdown("#### 📊 Route Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cluster ID", route_data["cluster_id"])
        st.metric("Destination Zone", route_data["DOLocationID"])
    
    with col2:
        st.metric("Max Volume", f"{route_data['max_volume']:,}")
        st.metric("Min Volume", f"{route_data['min_nonzero_volume']:.1f}")
    
    with col3:
        st.metric("Volume Ratio", f"{route_data['volume_ratio']:.1f}x")
        st.metric("Active Periods", route_data["periods_with_flow"])

    static_cols = {
        "cluster_id",
        "DOLocationID",
        "max_volume",
        "min_nonzero_volume",
        "volume_ratio",
        "periods_with_flow",
    }
    tp_cols = [c for c in df.columns if c not in static_cols]

    vol_series = row[tp_cols].astype(float)

    st.markdown("#### 📈 Volume Distribution by Time Period")
    fig_tp = px.bar(
        x=tp_cols,
        y=vol_series.values,
        labels={"x": "time period", "y": "trip volume"},
        title="Volume per time period for selected route",
        color=vol_series.values,
        color_continuous_scale="Blues",
    )
    fig_tp.update_layout(showlegend=False)
    st.plotly_chart(fig_tp, use_container_width=True)
    
    # Optional: Show detailed data table
    with st.expander("📋 View Detailed Period Volumes"):
        period_df = pd.DataFrame({
            'Time Period': tp_cols,
            'Volume': vol_series.values
        }).sort_values('Volume', ascending=False)
        st.dataframe(period_df, use_container_width=True, hide_index=True)