DESCRIPTION
This package analyzes NYC yellow taxi trip data (2023-2025) to discover high-demand corridors that could be served by dedicated shuttle services. The pipeline includes data cleaning, spatial clustering (KMeans/DBSCAN), temporal analysis, origin-destination (OD) network construction, and shuttle-route detection based on temporal volume variation. The project also includes an interactive Streamlit application with modules for exploratory OD visualization, temporal shuttle-patterns analysis, and an OD Sankey flow view.

INSTALLATION
Prerequisites:
- Python 3.8+
- `pip`
- (macOS) Homebrew if installing geospatial system deps

Recommended steps (macOS / zsh):
1) Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate
2) Upgrade pip:
   pip install --upgrade pip
3) Install Python dependencies from `requirements.txt`:
   pip install -r requirements.txt

If `geopandas` installation fails on macOS, install system libraries first:
   brew install gdal geos proj
then retry `pip install -r requirements.txt`.

EXECUTION
Full pipeline (first-time run): run the preprocessing scripts in order, then launch the app.
1) Clean and aggregate raw data:
   python 01_data_cleaning.py
2) Spatial clustering:
   python 02_spatial_clustering.py
3) Temporal clustering and hub identification:
   python 03_spatio_temporal_clustering.py
4) Build OD networks:
   python 04_OD_network_analysis.py
5) Identify shuttle patterns:
   python 05_temporal_shuttle_patterns.py

Launch the interactive demo (after preprocessing or if processed data exists):
   streamlit run App.py

The Streamlit app opens at: http://localhost:8501

DEMO VIDEO (optional)
Unlisted YouTube video (recommended): [No demo video provided — replace with your unlisted video URL here]

Notes
- Place raw taxi parquet files under `taxi_raw/taxt_data_23_25/` and the taxi zones shapefile under `taxi_zones/`.
- Processed output will be written to `taxi_processed/`.
- For reproducibility, consider pinning dependency versions in `requirements.txt`.

Parquet files
- Parquet trip data can be downloaded from the NYC Taxi & Limousine Commission (TLC) trip record data page:
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- The public S3 bucket mirror (index) is also available at:
  https://s3.amazonaws.com/nyc-tlc/trip+data/
- After downloading the relevant Parquet files (2023-2025), place them in `taxi_raw/taxt_data_23_25/` before running the preprocessing scripts.