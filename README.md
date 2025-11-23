# NYC Taxi Shuttle Route Analysis

## Description

This package analyzes NYC yellow taxi trip data (2023-2025) to identify potential shuttle routes through spatial and temporal pattern analysis. The system processes millions of taxi trips to discover high-demand corridors that could be served by dedicated shuttle services.

The analysis pipeline consists of five preprocessing scripts that clean data, perform spatial clustering (KMeans and DBSCAN), conduct temporal analysis, build origin-destination (OD) networks, and identify "shuttle routes" — corridors with dramatic temporal variation in demand. The package includes an interactive Streamlit web application with three visualization modules: an Exploratory OD Explorer with 3D geospatial maps and filtering controls, a Temporal Shuttle Patterns analyzer showing routes with high temporal variation, and an OD Network Sankey diagram visualizing flow patterns by time of day.

Key innovations include volume-weighted spatial clustering, time-period-specific hub identification, and shuttle route detection that finds corridors where demand varies by 3x+ across different time periods. The system is designed for transportation planners and analysts seeking data-driven insights for shuttle route planning.

## Installation

**Prerequisites**

- Python 3.8 or higher
- `pip` package manager

**Step 1: Install required packages**

```bash
pip install pandas numpy geopandas scikit-learn streamlit plotly pydeck matplotlib holidays pyarrow
```

**Step 2: Project structure setup**

Ensure your directory structure matches:

```
DVA-NYC-TAXI/
├── taxi_raw/
│   └── taxt_data_23_25/ # Place raw .parquet files here
├── taxi_zones/
│   └── taxi_zones.shp       # Taxi zone shapefile (and .shx, .dbf, .prj)
├── taxi_processed/          # Will be created automatically for output files
├── 01_data_cleaning.py
├── 02_spatial_clustering.py
├── 03_spatio_temporal_clustering.py
├── 04_OD_network_analysis.py
├── 05_temporal_shuttle_patterns.py
├── App.py
├── visual_app.py
├── shuttle_pattern_app.py
├── od_sankey_app.py
└── taxi_zone_lookup.csv
```

**Step 3: Data preparation**

- Download NYC Yellow Taxi trip data (2023-2025) from the NYC TLC website.
- Place parquet files in `taxi_raw/taxt_data_23_25/`.
- Obtain the NYC Taxi Zones shapefile and place it in `taxi_zones/`.

### Environment (recommended)


```bash
# create a virtual environment named "venv"
python3 -m venv venv

# activate the virtual environment (zsh)
source venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install Python dependencies from requirements.txt
pip install -r requirements.txt
```

Note: `geopandas` may require system dependencies (GDAL/PROJ/GEOS). If you encounter installation issues on macOS, install these via Homebrew before installing Python packages:

```bash
brew install gdal geos proj
```
Then retry `pip install -r requirements.txt`.

## Execution

### Full pipeline (first-time setup)

Run preprocessing scripts in order:

```bash
# Step 1: Clean and aggregate raw data (10-30 minutes depending on data size)
python 01_data_cleaning.py

# Step 2: Perform spatial clustering (5-10 minutes)
python 02_spatial_clustering.py

# Step 3: Temporal clustering and hub identification (5-10 minutes)
python 03_spatio_temporal_clustering.py

# Step 4: Build OD networks (2-5 minutes)
python 04_OD_network_analysis.py

# Step 5: Identify shuttle patterns (2-5 minutes)
python 05_temporal_shuttle_patterns.py
```

Launch the interactive application after preprocessing is complete:

```bash
streamlit run App.py
```

The application will open in your default web browser at `http://localhost:8501`.

If processed data already exists, you can run the demo directly:

```bash
streamlit run App.py
```

## Application Usage

**Tab 1: Exploratory OD Explorer**

- Use the sidebar to filter by date range, hour range, and holidays.
- Select clustering algorithm (`KMeans` or `DBSCAN`).
- Explore the interactive 3D map with customizable layers.
- View demand timelines, hourly patterns, and cluster heatmaps.

**Tab 2: Temporal Shuttle Patterns**

- Adjust the slider to view top N metamorphic routes.
- Select individual routes to inspect detailed metrics.
- Analyze volume distribution across time periods.
- Identify routes with high temporal variation (ideal for time-specific shuttles).

**Tab 3: OD Network - Sankey View**

- Select time of day (morning/midday/evening/night).
- Adjust the top N routes slider.
- View flow patterns between origin and destination clusters.
- Examine detailed route statistics in the expandable table.