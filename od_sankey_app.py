from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "taxi_processed"


@st.cache_data(show_spinner=False)
def load_od_network() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "od_network_23_25.parquet")


def app(colors=None):
    st.header("OD Network — Sankey View")

    df = load_od_network()

    if df.empty:
        st.info("OD network file is empty.")
        return

    st.markdown("**Visualize origin-destination flows by time of day**")
    
    time_col = st.selectbox("Time of day", ["morning", "midday", "evening", "night"])
    top_n = st.slider("Top routes by volume", 10, 200, 50)

    df["flow"] = df[time_col]
    df_top = df.sort_values("flow", ascending=False).head(top_n)

    if df_top["flow"].sum() == 0:
        st.info("No flows for this time-of-day selection.")
        return

    # Use 'pickup' and 'dropoff' columns (from clustered data)
    pulocs = df_top["pickup"].unique().tolist()
    dolocs = df_top["dropoff"].unique().tolist()

    # Create node labels
    nodes = [f"Origin {i}" for i in pulocs] + [f"Dest {i}" for i in dolocs]
    node_index = {label: idx for idx, label in enumerate(nodes)}

    # Map source and target indices
    sources = [node_index[f"Origin {i}"] for i in df_top["pickup"]]
    targets = [node_index[f"Dest {j}"] for j in df_top["dropoff"]]
    values = df_top["flow"].tolist()

    # Create color scheme for better visibility
    node_colors = ['rgba(31, 119, 180, 0.8)'] * len(pulocs) + ['rgba(255, 127, 14, 0.8)'] * len(dolocs)
    
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    label=nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color='rgba(100, 100, 100, 0.3)',
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Top {top_n} OD flows during {time_col}",
        font_size=12,
        height=600,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary statistics
    st.markdown("### 📊 Flow Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trips", f"{int(df_top['flow'].sum()):,}")
    
    with col2:
        st.metric("Origin Clusters", len(pulocs))
    
    with col3:
        st.metric("Destination Clusters", len(dolocs))
    
    # Optional: Show top routes table
    with st.expander("📋 View Top Routes Details"):
        display_df = df_top[["pickup", "dropoff", "flow"]].copy()
        display_df.columns = ["Origin Cluster", "Destination Cluster", "Trip Count"]
        display_df = display_df.sort_values("Trip Count", ascending=False).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=False)