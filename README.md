# SNCF & RTE Energy Analysis

Interactive Streamlit application for analyzing SNCF greenhouse gas emissions, CO2 data, and RTE energy consumption/production data with granular analysis from macro to micro levels.

## Driving Question Asked by Our Client

**How much could we reduce SNCF's carbon footprint and energy costs if we could time-shift certain operations to coincide with surplus renewable energy on the French grid?**

-> This question actually asks us to understand how SNCF can optimize its rail transport system in real-time to reduce emissions and energy costs, by dynamically aligning train operations with the cleanest and cheapest available energy on the RTE grid. We therefore created the following features.

## Features

### 📊 Macro Overview
- Total emissions and energy consumption metrics
- High-level time series visualization
- Overall data summary statistics

### 🏭 Sector Analysis (Meso Level)
- Breakdown by sectors/categories
- Top N sectors visualization (bar charts and pie charts)
- Time series comparison across sectors
- Interactive sector selection

### 🔬 Micro Analysis (Detailed Level)
- Drill-down into specific sectors
- Sub-category breakdowns
- Detailed statistics and raw data views
- Granular exploration of individual sectors

### ⚖️ Comparative Analysis
- Side-by-side comparison of GHG and CO2 emissions
- Correlation analysis
- Cross-dataset insights

### ⚡ RTE Energy Analysis
- **Peak & Down Timing Analysis**:
  - Hourly patterns to identify peak consumption/production times
  - Day of week patterns
  - Monthly patterns with peak identification
  - Time series with peak and low periods highlighted
- **Production by Sector**:
  - Total production breakdown by sector
  - Interactive sector comparison
  - Time series analysis per sector
  - Sector summary statistics
- **Energy Consumption Analysis**:
  - Daily consumption patterns
  - Peak demand identification
  - Consumption trends over time

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using the project configuration:
```bash
pip install -e .
```

## Usage

Run the Streamlit app:

```bash
streamlit run dashboard.py
```

Or use the provided script:

```bash
./run_app.sh
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The application uses data from the `data/` folder:

### SNCF Data (in `data/sncf/`):
- `bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet` - Greenhouse gas emissions balance
- `emission-co2-perimetre-complet.parquet` - CO2 emissions complete perimeter

### RTE Data (in `data/rte/`):
- `Historique_consommation_JOUR_2023.xls` - Daily energy consumption history
- `RealisationDonneesProduction_2023.xls` - Energy production data by sector

## Navigation

1. **Macro Overview**: Start here for high-level insights
2. **Sector Analysis**: Explore emissions by different sectors/categories
3. **Micro Analysis**: Drill down into specific sectors for detailed breakdowns
4. **Comparison**: Compare GHG and CO2 emissions across different dimensions
5. **RTE Energy Analysis**: Analyze energy consumption patterns, peak timings, and production by sector

## Features

- **Interactive Visualizations**: Built with Plotly for zooming, panning, and hovering
- **Granular Analysis**: Progress from macro → meso → micro levels
- **Energy & Emissions Focus**: Specifically designed to identify which sectors use the most energy and emit the most carbon
- **Peak Timing Analysis**: Identify peak consumption and production times (hourly, daily, monthly patterns)
- **Production by Sector**: Detailed breakdown of energy production across different sectors
- **Flexible Data Handling**: Automatically detects date columns, numeric columns, and categorical columns
- **Real-time Filtering**: Use sidebar and dropdowns to filter and explore data
- **Multi-format Support**: Handles Parquet (SNCF) and Excel (RTE) data formats

## Conclusion from our Data Consultants

