# SNCF Emissions & Energy Analysis

Interactive Streamlit application for analyzing SNCF greenhouse gas emissions and CO2 data with granular analysis from macro to micro levels.

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
streamlit run sncf.py
```

Or use the provided script:

```bash
./run_app.sh
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The application uses two parquet files from the `data/` folder:
- `bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet` - Greenhouse gas emissions balance
- `emission-co2-perimetre-complet.parquet` - CO2 emissions complete perimeter

## Navigation

1. **Macro Overview**: Start here for high-level insights
2. **Sector Analysis**: Explore emissions by different sectors/categories
3. **Micro Analysis**: Drill down into specific sectors for detailed breakdowns
4. **Comparison**: Compare GHG and CO2 emissions across different dimensions

## Features

- **Interactive Visualizations**: Built with Plotly for zooming, panning, and hovering
- **Granular Analysis**: Progress from macro → meso → micro levels
- **Energy & Emissions Focus**: Specifically designed to identify which sectors use the most energy and emit the most carbon
- **Flexible Data Handling**: Automatically detects date columns, numeric columns, and categorical columns
- **Real-time Filtering**: Use sidebar and dropdowns to filter and explore data
