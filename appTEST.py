import streamlit as st
from pathlib import Path
from streamlit.components.v1 import html

st.set_page_config(
    page_title="SNCF & RTE Energy Dashboard",
    layout="wide"
)

ROOT = Path(__file__).parent
VIS = ROOT / "visualizations"
IMG = ROOT / "images"

st.title("⚡ SNCF & RTE Energy Optimization Dashboard")
st.markdown("""
**Client question:**  
How much could SNCF reduce emissions and costs by shifting operations to green energy surplus periods?
""")

# Sidebar navigation
page = st.sidebar.radio(
    "Dashboard",
    [
        "🏠 Overview",
        "📊 SNCF Emissions",
        "⚖️ GHG vs CO₂",
        "⚡ RTE Energy",
        "🔬 Deep Analysis"
    ]
)

def show_html(file, height=800):
    path = VIS / file
    if not path.exists():
        st.error(f"Missing {file}")
        return
    html(path.read_text(), height=height, scrolling=True)

# -----------------------
# Pages
# -----------------------

if page == "🏠 Overview":
    st.subheader("System Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(IMG / "capacity_time_series.png"), caption="Installed Capacity Over Time")
    with c2:
        st.image(str(IMG / "top_production_units.png"), caption="Top Production Units")

elif page == "📊 SNCF Emissions":
    st.subheader("SNCF – Emissions Overview")
    show_html("sncf_dashboard.html")

elif page == "⚖️ GHG vs CO₂":
    col1, col2 = st.columns(2)
    with col1:
        show_html("emissions_bar_chart.html", 600)
    with col2:
        show_html("co2_bar_chart.html", 600)
    show_html("emissions_time_series.html", 600)

elif page == "⚡ RTE Energy":
    st.subheader("Energy Production & Consumption (RTE)")
    show_html("emissions_heatmap.html", 700)

elif page == "🔬 Deep Analysis":
    show_html("emissions_scatter_matrix.html", 900)
