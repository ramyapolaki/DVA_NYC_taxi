import streamlit as st
from pathlib import Path

# Import your existing app modules
from visual_app import apply_visual_theme, load_flow_data, render_flow_section
import shuttle_pattern_app
import od_sankey_app

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Mobility — Team 13",
    layout="wide",
    page_icon="🗽",
)

def main():
    # Apply your custom theme
    colors = apply_visual_theme(False)
    
    # Main title
    st.title("🗽 NYC Taxi Mobility Analysis — Team 13")
    st.markdown(
        "Interactive visual analytics over NYC taxi trips: clustering, OD corridors, "
        "and temporal shuttle patterns."
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Visual Explorer",
        "🚐 Temporal Shuttle Patterns", 
        "🔀 OD Network — Sankey View"
    ])
    
    # Tab 1: Visual OD Explorer
    with tab1:
        try:
            flow = load_flow_data()
            st.session_state["flow_meta"] = {
                "rows": len(flow),
                "date_min": flow["trip_date"].min().date(),
                "date_max": flow["trip_date"].max().date(),
            }
            render_flow_section(flow, colors)
        except Exception as e:
            st.error(f"Error loading OD Explorer: {str(e)}")
            st.info("Please ensure the data file 'grouped_clustered_23_25.parquet' exists in taxi_processed/")
    
    # Tab 2: Temporal Shuttle Patterns
    with tab2:
        try:
            shuttle_pattern_app.app(colors)
        except Exception as e:
            st.error(f"Error loading Shuttle Patterns: {str(e)}")
            st.info("Please ensure the data file 'metamorphic_routes_23_25.parquet' exists in taxi_processed/")
    
    # Tab 3: OD Network Sankey
    with tab3:
        try:
            od_sankey_app.app(colors)
        except Exception as e:
            st.error(f"Error loading OD Sankey: {str(e)}")
            st.info("Please ensure the data file 'od_network_23_25.parquet' exists in taxi_processed/")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "Team 13 | NYC Taxi Trip Analysis | Data Visualization & Analytics"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()