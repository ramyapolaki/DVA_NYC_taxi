import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pydeck as pdk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="NYC Taxi Mobility – Team 13",
    layout="wide",
    page_icon="🗽",
)


def _theme_palette(dark_mode: bool) -> Dict[str, object]:
    if dark_mode:
        return {
            "bg": "#0b1220",
            "bg_overlay": "#0f172a",
            "panel": "#111827",
            "text": "#e2e8f0",
            "muted": "#a0aec0",
            "border": "rgba(148, 163, 184, 0.25)",
            "primary": "#7dd3fc",
            "secondary": "#a5b4fc",
            "accent": "#fb7185",
            "discrete": ["#7dd3fc", "#f6d860", "#fb7185", "#a5b4fc", "#34d399", "#f97210"],
            "continuous": "Plasma",
            "heat_scale": "Magma",
            "hour_scale": "Turbo",
        }
    return {
        "bg": "#f8fafc",
        "bg_overlay": "#ffffff",
        "panel": "#edf2f7",
        "text": "#0f172a",
        "muted": "#475569",
        "border": "rgba(148, 163, 184, 0.35)",
        "primary": "#1d6996",
        "secondary": "#e17c05",
        "accent": "#ef476f",
        "discrete": px.colors.qualitative.Set2,
        "continuous": "Viridis",
        "heat_scale": "Viridis",
        "hour_scale": "Turbo",
    }


def apply_visual_theme(dark_mode: bool) -> Dict[str, object]:
    palette = _theme_palette(dark_mode)
    px.defaults.template = "plotly_dark" if dark_mode else "plotly_white"
    px.defaults.color_discrete_sequence = palette["discrete"]
    px.defaults.color_continuous_scale = palette["continuous"]
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-color: {palette["bg"]};
            --panel-color: {palette["panel"]};
            --bg-overlay: {palette["bg_overlay"]};
            --text-color: {palette["text"]};
            --muted-color: {palette["muted"]};
            --border-color: {palette["border"]};
            --accent-color: {palette["primary"]};
            --font-stack: 'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
        }}
        html, body, .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: var(--font-stack);
        }}
        [data-testid="stSidebar"] {{
            background-color: var(--panel-color);
            color: var(--text-color);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text-color) !important;
            font-family: var(--font-stack);
        }}

        /* Sidebar selects & inputs with dark background + white text */
        [data-testid="stSidebar"] [data-baseweb="select"] {{
            background-color: #1a1a1a;
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: #1a1a1a;
            color: #ffffff !important;
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] input {{
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] .stDateInput > div > div > input {{
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border-radius: 8px;
            max-width: 200px;
        }}
        [data-testid="stSidebar"] .stDateInput {{
            max-width: 200px;
        }}
        [data-testid="stSidebar"] .stDateInput > div {{
            width: 100%;
            max-width: 200px;
        }}
        [data-testid="stSidebar"] .stSelectbox {{
            max-width: 200px;
        }}
        [data-testid="stSidebar"] .stSelectbox > div {{
            max-width: 200px;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: #ffffff !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] div[role="button"] {{
            color: #ffffff !important;
        }}
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {{
            color: #ffffff !important;
        }}

        /* 🔥 SPECIFIC FIX for the 'keyboard_double_arrow_left' span */
        span.st-emotion-cache-ujm5ma[data-testid="stIconMaterial"] {{
            font-size: 0 !important;              /* hide the raw text */
        }}
        span.st-emotion-cache-ujm5ma[data-testid="stIconMaterial"]::before {{
            content: "«" !important;              /* actual arrow */
            font-size: 22px !important;
            color: #000000 !important;
            display: inline-block !important;
            font-weight: bold !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: var(--text-color);
            font-family: var(--font-stack);
        }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {{
            color: var(--text-color);
        }}
        div[data-testid="stDeckGlChart"] {{
            background: var(--bg-color);
        }}
        button[kind="secondary"] {{
            border: 1px solid var(--border-color);
        }}
        .pill-toggle button {{
            border-radius: 999px !important;
            font-weight: 700;
        }}
        textarea, textarea:focus {{
            font-size: 15px !important;
            line-height: 1.5 !important;
        }}
        .cluster-selector [role="radiogroup"] {{
            flex-direction: row !important;
            gap: 0.5rem;
        }}
        .cluster-selector [role="radiogroup"] label {{
            border: 1px solid var(--border-color);
            border-radius: 999px;
            padding: 4px 12px;
            background: var(--panel-color);
            color: var(--text-color);
        }}
        .cluster-selector [role="radiogroup"] label:hover {{
            border-color: var(--accent-color);
        }}
        .cluster-selector [role="radiogroup"] input:checked + div {{
            background: var(--accent-color);
            color: #0b1220;
            padding: 4px 12px;
            border-radius: 999px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return palette

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "taxi_processed" / "grouped_clustered_23_25.parquet"


@st.cache_data(show_spinner=False)
def load_flow_data() -> pd.DataFrame:
    flow = pd.read_parquet(DATA_FILE)
    flow["trip_date"] = pd.to_datetime(flow["trip_date"])
    flow['pickup_day'] = flow['trip_date'].dt.day_name()
    return flow


def kpi_card(label: str, value: str, help_text: str = ""):
    st.metric(label, value, help=help_text if help_text else None)


def render_explanation(title: str, details: Dict[str, str]):
    return None


def _to_date(value, fallback: date) -> date:
    if isinstance(value, (list, tuple)) and value:
        return _to_date(value[0], fallback)
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return fallback


def _hex_to_rgb_tuple(hex_color: str) -> Tuple[int, int, int]:
    if hex_color.startswith("rgb"):
        nums = hex_color.strip("rgba() ").split(",")
        return tuple(int(float(n)) for n in nums[:3])
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def apply_map_height_style(height: int = 650):
    st.markdown(
        f"""
        <style>
        div[data-testid="stDeckGlChart"] {{
            min-height: {height}px !important;
            height: {height}px !important;
        }}
        iframe[data-testid="stDeckGlChart"] {{
            min-height: {height}px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_zone_lookup() -> Optional[pd.DataFrame]:
    lookup_path = BASE_DIR / "taxi_zone_lookup.csv"
    if not lookup_path.exists():
        return None
    df = pd.read_csv(lookup_path)
    loc_col = next((c for c in df.columns if c.lower() in {"locationid", "location_id"}), None)
    borough_col = next((c for c in df.columns if c.lower() == "borough"), None)
    lat_col = next((c for c in df.columns if c.lower() in {"lat", "latitude"}), None)
    lon_col = next((c for c in df.columns if c.lower() in {"lon", "longitude", "lng"}), None)
    if not loc_col:
        return None
    df = df.rename(columns={loc_col: "LocationID"})
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
    else:
        if not borough_col:
            return None
        borough_centers = {
            "Manhattan": (40.7831, -73.9712),
            "Brooklyn": (40.6782, -73.9442),
            "Queens": (40.7282, -73.7949),
            "Bronx": (40.8448, -73.8648),
            "Staten Island": (40.5795, -74.1502),
            "EWR": (40.6895, -74.1745),
        }
        df["lat"] = df[borough_col].map(lambda b: borough_centers.get(b, (None, None))[0])
        df["lon"] = df[borough_col].map(lambda b: borough_centers.get(b, (None, None))[1])
    df = df.dropna(subset=["lat", "lon"])
    cols = ["LocationID", "lat", "lon"]
    if borough_col:
        df = df.rename(columns={borough_col: "Borough"})
        cols.append("Borough")
    return df[cols]


def render_flow_map(
    filtered: pd.DataFrame,
    colors: Dict[str, object],
    zone_lookup: Optional[pd.DataFrame],
    max_pairs: int = 500,
    cluster: str = ""
):
    st.subheader("Geospatial OD Flow Map — High-Flow Shuttle Routes")

    if zone_lookup is None:
        st.info("Taxi zone lookup with coordinates is missing, so the map cannot be drawn.")
        return

    st.markdown("### ⚙️ Map Controls")

    col1, col2 = st.columns(2)
    with col1:
        filter_mode = st.radio(
            "Filter Mode",
            ["Top-N routes", "Percentile cutoff"],
            horizontal=True
        )
    with col2:
        layer_type = st.radio(
            "Layer Type",
            ["Arc", "Line", "Path + Arrows"],
            horizontal=True
        )

    if filter_mode == "Top-N routes":
        top_n = st.slider("Show top N highest-flow routes", 10, 500, 150)
        percentile_cutoff = None
    else:
        percentile_cutoff = st.slider(
            "Flow percentile cutoff",
            50, 99, 85,
            help="E.g., 85 keeps only the top 15% highest-flow routes"
        )
        top_n = None

    min_count_threshold = st.slider(
        "Minimum trip count",
        0,
        int(filtered["count"].max()),
        50,
        help="Hide extremely low-flow routes"
    )

    clusters_available = sorted(filtered[cluster + "pickup"].unique())
    selected_clusters = st.multiselect(
        "Select pickup clusters",
        clusters_available,
        default=clusters_available
    )

    show_endpoints = st.checkbox("Show endpoints", True)
    show_labels = st.checkbox("Show route labels (top 40 only)", False)

    st.markdown("### 🎨 Color Controls")
    color_mode = st.radio(
        "Color mode",
        ["Cluster-based", "Flow-based", "Single color"],
        horizontal=True
    )

    if color_mode == "Single color":
        chosen_color = st.color_picker("Pick route color", "#2B83BA")

    od = (
        filtered.groupby([
            cluster + "pickup",
            cluster + "dropoff",
            "pickup_latitude", "pickup_longitude",
            "dropoff_latitude", "dropoff_longitude",
        ])["count"]
        .sum()
        .reset_index()
        .dropna()
    )

    od = od[od[cluster + "pickup"].isin(selected_clusters)]
    od = od[od["count"] >= min_count_threshold]

    if percentile_cutoff is not None:
        thresh = np.percentile(od["count"], percentile_cutoff)
        od = od[od["count"] >= thresh]

    if top_n is not None:
        od = od.sort_values("count", ascending=False).head(top_n)

    if od.empty:
        st.warning("No routes remain after filtering; adjust thresholds.")
        return

    max_count = od["count"].max()

    od["width"] = np.interp(
        od["count"],
        (od["count"].min(), od["count"].max()),
        (1.0, 18.0)
    )

    od["alpha"] = np.interp(
        od["count"],
        (od["count"].min(), od["count"].max()),
        (60, 230)
    ).astype(int)

    if color_mode == "Cluster-based":
        raw = (plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(int)
        cluster_color = {
            cid: raw[cid % 20].tolist() for cid in clusters_available
        }
        od["color"] = od.apply(
            lambda r: cluster_color[r[cluster + "pickup"]] + [int(r["alpha"])],
            axis=1
        )

    elif color_mode == "Flow-based":
        cmap = plt.cm.Blues
        od["color"] = od["count"].apply(
            lambda c: (np.array(cmap(c / max_count)[:3]) * 255).astype(int).tolist() + [200]
        )

    else:
        rgb = tuple(int(chosen_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        od["color"] = [list(rgb) + [200]] * len(od)

    if show_endpoints:
        endpoints = pd.concat([
            od[["pickup_latitude","pickup_longitude","count"]]
            .rename(columns={"pickup_latitude":"lat","pickup_longitude":"lon"})
            .assign(role="Pickup"),
            od[["dropoff_latitude","dropoff_longitude","count"]]
            .rename(columns={"dropoff_latitude":"lat","dropoff_longitude":"lon"})
            .assign(role="Dropoff"),
        ], ignore_index=True)
        endpoints["radius"] = np.interp(
            endpoints["count"],
            (endpoints["count"].min(), endpoints["count"].max()),
            (80, 350)
        )
        endpoints["color"] = endpoints["role"].map({
            "Pickup": [200, 80, 80, 140],
            "Dropoff": [80, 80, 200, 140]
        })
    else:
        endpoints = None

    layers = []

    if layer_type == "Arc":
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=od,
                get_source_position=["pickup_longitude", "pickup_latitude"],
                get_target_position=["dropoff_longitude", "dropoff_latitude"],
                get_source_color="color",
                get_target_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True,
            )
        )

    elif layer_type == "Line":
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=od,
                get_source_position=["pickup_longitude", "pickup_latitude"],
                get_target_position=["dropoff_longitude", "dropoff_latitude"],
                get_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True,
            )
        )

    else:
        od["path"] = od.apply(
            lambda r: [
                [r["pickup_longitude"], r["pickup_latitude"]],
                [r["dropoff_longitude"], r["dropoff_latitude"]],
            ],
            axis=1
        )

        layers.append(
            pdk.Layer(
                "PathLayer",
                data=od,
                get_path="path",
                get_color="color",
                width_scale=1,
                width_min_pixels=2,
                get_width="width",
                pickable=True,
                auto_highlight=True,
                get_tilt=15,
                get_height=0,
                get_rotation=0,
            )
        )

    if show_labels:
        label_data = od.head(40).copy()
        label_data["label"] = label_data.apply(
            lambda r: f"{int(r['count'])} trips", axis=1
        )
        label_data["lon"] = (label_data["pickup_longitude"] + label_data["dropoff_longitude"]) / 2
        label_data["lat"] = (label_data["pickup_latitude"] + label_data["dropoff_latitude"]) / 2

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=label_data,
                get_position=["lon", "lat"],
                get_text="label",
                get_size=16,
                get_color=[50, 50, 50],
                get_angle=0,
                get_alignment_baseline="'bottom'",
            )
        )

    if show_endpoints and endpoints is not None:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=endpoints,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                stroked=False,
                opacity=0.55,
            )
        )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0,
            zoom=9.6,
            pitch=45,
            bearing=20,
        ),
        layers=layers,
        tooltip={"text": "Trips: {count}"},
    )

    st.pydeck_chart(deck)


def render_cluster_map(
    filtered: pd.DataFrame,
    colors: Dict[str, object],
    algo_label: str,
    origin_cluster_col: str = "pickup",
    dest_cluster_col: str = "dropoff",
    origin_lat_col: str = "pickup_latitude",
    origin_lon_col: str = "pickup_longitude",
    dest_lat_col: str = "dropoff_latitude",
    dest_lon_col: str = "dropoff_longitude",
):
    st.subheader(f"{algo_label.upper()} cluster presence map")

    def _make_points(lat_col, lon_col, cluster_col, role_label):
        pts = (
            filtered.groupby([cluster_col, lat_col, lon_col])["count"]
            .sum()
            .reset_index()
            .rename(columns={
                lat_col: "lat",
                lon_col: "lon",
                cluster_col: "cluster"
            })
        )
        pts["role"] = role_label
        return pts

    origin_pts = _make_points(
        origin_lat_col, origin_lon_col, origin_cluster_col, "Pickup cluster"
    )
    dest_pts = _make_points(
        dest_lat_col, dest_lon_col, dest_cluster_col, "Dropoff cluster"
    )

    all_pts = pd.concat([origin_pts, dest_pts], ignore_index=True)

    if all_pts.empty:
        st.info("No cluster coordinates available in the current filtered selection.")
        return

    max_count = all_pts["count"].max()

    palette = colors["discrete"]
    if not isinstance(palette, list):
        palette = [palette]

    unique_clusters = sorted(all_pts["cluster"].unique())
    color_map = {
        c: _hex_to_rgb_tuple(palette[i % len(palette)])
        for i, c in enumerate(unique_clusters)
    }

    all_pts["color"] = all_pts["cluster"].map(
        lambda c: [*color_map[c], 185] if c in color_map else [120, 120, 120, 160]
    )

    all_pts["radius"] = all_pts["count"].apply(lambda v: max(120, (v / max_count) * 440))
    all_pts["cluster_label"] = all_pts["cluster"].apply(lambda c: f"Cluster {int(c)}")

    apply_map_height_style(620)

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=40.72,
            longitude=-73.95,
            zoom=11,
            pitch=35,
            bearing=15,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=all_pts,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                stroked=False,
                opacity=0.65,
                pickable=True,
                auto_highlight=True,
            )
        ],
        tooltip={
            "text": "{role}: {cluster_label}\nTrips: {count:,}\nLat/Lon: {lat:.4f}, {lon:.4f}"
        },
    )

    st.pydeck_chart(deck)

    st.caption(
        "Dots indicate where pickup/dropoff clusters are spatially concentrated, "
        "with size proportional to trip counts."
    )


def render_flow_section(flow: pd.DataFrame, colors: Dict[str, object]):
    st.header("Exploratory Mobility Visuals")
    primary_color = colors["primary"]
    accent_color = colors["accent"]
    min_date = date(2023, 1, 1)
    max_date = date(2025, 9, 30)
    
    start_default = st.session_state.get("start_date", min_date)
    start_date = st.sidebar.date_input(
        "Start date",
        value=start_default,
        min_value=min_date,
        max_value=max_date,
        key="start_date",
    )
    
    end_default = st.session_state.get("end_date", max_date)
    if end_default < start_date:
        end_default = start_date
    end_date = st.sidebar.date_input(
        "End date",
        value=end_default,
        min_value=start_date,
        max_value=max_date,
        key="end_date",
        help="Only dates on/after the chosen start date are shown.",
    )

    hour_options = list(range(24))
    start_hour_default = st.session_state.get("start_hour", 0)
    end_hour_default = st.session_state.get("end_hour", 23)
    start_hour = st.sidebar.selectbox(
        "Start hour",
        options=hour_options,
        index=hour_options.index(start_hour_default)
        if start_hour_default in hour_options
        else 0,
        key="start_hour",
    )
    end_hour_options = [h for h in hour_options if h >= start_hour]
    if "end_hour" in st.session_state and st.session_state["end_hour"] not in end_hour_options:
        st.session_state["end_hour"] = end_hour_options[-1]
    if end_hour_default not in end_hour_options:
        end_hour_default = end_hour_options[-1]
    end_hour = st.sidebar.selectbox(
        "End hour",
        options=end_hour_options,
        index=end_hour_options.index(end_hour_default),
        key="end_hour",
        help="Only hours on/after the chosen start hour are shown.",
    )
    holiday_only = st.sidebar.checkbox("Public holidays only", value=False)
    st.sidebar.markdown("**Cluster method**")
    cluster_default = st.session_state.get("cluster_algo", "kmeans")
    st.sidebar.markdown("<div class='cluster-selector'>", unsafe_allow_html=True)
    cluster_algo = st.sidebar.radio(
        "Cluster method",
        options=["kmeans", "dbscan"],
        format_func=lambda x: x.upper(),
        horizontal=True,
        key="cluster_selector",
        index=["kmeans", "dbscan"].index(cluster_default if cluster_default in ["kmeans", "dbscan"] else "kmeans"),
        label_visibility="collapsed",
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.session_state["cluster_algo"] = cluster_algo

    zone_lookup = load_zone_lookup()

    start_date = _to_date(start_date, min_date)
    end_date = _to_date(end_date, max_date)
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour
    filtered = flow[
        (flow["trip_date"].dt.date >= start_date)
        & (flow["trip_date"].dt.date <= end_date)
        & (flow["pickup_hour"].between(start_hour, end_hour))
    ].copy()
    filtered = filtered[filtered["Clustering"] == cluster_algo]
    if holiday_only:
        filtered = filtered[filtered["is_public_holiday"]]

    total_trips = int(filtered["count"].sum())
    mean_trips = (
        int(filtered.groupby("trip_date")["count"].sum().mean())
        if not filtered.empty
        else 0
    )
    with st.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            kpi_card("Trips in view", f"{total_trips:,}")
        with kpi2:
            kpi_card("Avg trips / day", f"{mean_trips:,}")
        with kpi3:
            kpi_card(
                "Filtered span (days)",
                str((end_date - start_date).days + 1),
                "Date window applied to visuals.",
            )

    if filtered.empty:
        st.warning("No trips match the current filters.")
        return

    render_flow_map(filtered, colors, zone_lookup)

    daily = filtered.groupby("trip_date")["count"].sum().reset_index()
    hourly = filtered.groupby("pickup_hour")["count"].sum().reset_index()
    od = (
        filtered.groupby(["pickup", "dropoff"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(15)
    )
    cluster_matrix = (
        filtered.groupby(["pickup", "dropoff"])["count"]
        .sum()
        .reset_index()
    )

    render_cluster_map(
        filtered,
        colors,
        cluster_algo,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily demand timeline")
        fig = px.line(
            daily,
            x="trip_date",
            y="count",
            markers=True,
            color_discrete_sequence=[primary_color],
            labels={"trip_date": "Date", "count": "Trips"},
            title="Trips per day",
        )
        fig.update_layout(legend_title_text="Daily demand", font=dict(size=13))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Hour-of-day profile")
        fig = px.bar(
            hourly,
            x="pickup_hour",
            y="count",
            color="pickup_hour",
            color_continuous_scale=colors["hour_scale"],
            labels={"pickup_hour": "Hour", "count": "Trips"},
            title="Trips by pickup hour",
        )
        fig.update_layout(coloraxis_colorbar_title="Hour", font=dict(size=13))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top origin–destination pairs")
    fig = px.bar(
        od,
        x="count",
        y=od.apply(lambda r: f"{int(r.pickup)} → {int(r.dropoff)}", axis=1),
        color="count",
        color_continuous_scale=colors["continuous"],
        labels={"count": "Trips", "y": "Pickup → Dropoff"},
        orientation="h",
        title="Most active OD pairs",
    )
    fig.update_layout(coloraxis_colorbar_title="Trips", font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("OD coverage curve")
    od_curve = od.assign(cum_share=od["count"].cumsum() / od["count"].sum())
    fig_curve = px.line(
        od_curve,
        x=od_curve.index + 1,
        y="cum_share",
        markers=True,
        color_discrete_sequence=[accent_color],
        labels={"x": "Top N OD pairs", "cum_share": "Cumulative trip share"},
        title="How quickly top OD pairs cover trips",
    )
    fig_curve.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_curve, use_container_width=True)

    st.subheader("Cluster-to-cluster flows")
    fig = px.density_heatmap(
        cluster_matrix,
        x="pickup",
        y="dropoff",
        z="count",
        color_continuous_scale=colors["heat_scale"],
        labels={
            "pickup": "Pickup cluster",
            "dropoff": "Drop-off cluster",
            "count": "Trips",
        },
        title=f"{cluster_algo.upper()} cluster connectivity",
    )
    fig.update_layout(font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

    heat = (
        filtered.groupby(["pickup_day", "pickup_hour"])["count"]
        .sum()
        .reset_index()
        .pivot(index="pickup_day", columns="pickup_hour", values="count")
        .reindex(index=sorted(filtered["pickup_day"].unique()))
    )
    st.subheader("Weekday × hour demand grid")
    if heat.empty:
        st.info("No data for heatmap in this filter range.")
    else:
        fig_heat = px.imshow(
            heat,
            aspect="auto",
            color_continuous_scale=colors["heat_scale"],
            labels={"x": "Hour", "y": "Day of week", "color": "Trips"},
            title="Hourly intensity by weekday",
        )
        fig_heat.update_layout(yaxis_title="Weekday")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("3D flow intensity (geo + time)")
    flows_3d = (
        filtered.groupby(["pickup", "dropoff", "pickup_hour"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(800)
    )
    if flows_3d.empty:
        st.info("No data to show for the current filters.")
    else:
        if "pickup_latitude" in flows_3d.columns and flows_3d["pickup_latitude"].notna().any():
            fig3d = px.scatter_3d(
                flows_3d,
                x="pickup_longitude",
                y="pickup_latitude",
                z="pickup_hour",
                color="count",
                size="count",
                opacity=0.8,
                labels={
                    "pickup_longitude": "Longitude",
                    "pickup_latitude": "Latitude",
                    "pickup_hour": "Hour",
                    "count": "Trips",
                },
                title="Top flows (pickup longitude/latitude × hour)",
                color_continuous_scale=colors["continuous"],
            )
        else:
            fig3d = px.scatter_3d(
                flows_3d,
                x="pickup",
                y="dropoff",
                z="pickup_hour",
                color="count",
                size="count",
                opacity=0.8,
                labels={
                    "pickup": "Pickup cluster",
                    "dropoff": "Drop-off cluster",
                    "pickup_hour": "Hour",
                    "count": "Trips",
                },
                title="Top flows (Clusters × hour)",
                color_continuous_scale=colors["continuous"],
            )
        fig3d.update_traces(marker=dict(line=dict(width=0)))
        st.plotly_chart(fig3d, use_container_width=True)


def main():
    colors = apply_visual_theme(False)

    st.title("Visualizing Potential Shuttle Routes – Team 13")
    st.markdown(
        "Interactive visual analytics over NYC taxi trips: clustering, OD corridors, "
        "and lightweight demand models aligned with the project plan."
    )
    flow = load_flow_data()
    st.session_state["flow_meta"] = {
        "rows": len(flow),
        "date_min": flow["trip_date"].min().date(),
        "date_max": flow["trip_date"].max().date(),
    }

    render_flow_section(flow, colors)


if __name__ == "__main__":
    main()