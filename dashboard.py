"""
SNCF & RTE Data Visualization - Streamlit App
Interactive visualizations for SNCF greenhouse gas emissions, CO2 data, and RTE energy data
with granular analysis from macro to micro levels.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="SNCF & RTE Energy Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the SNCF parquet data files and RTE preprocessed parquet files."""
    data_dir = Path(__file__).parent / "data"
    
    # Load SNCF data
    sncf_dir = data_dir / "sncf"
    emissions_df = pd.read_parquet(
        sncf_dir / "bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet"
    )
    
    co2_df = pd.read_parquet(
        sncf_dir / "emission-co2-perimetre-complet.parquet"
    )
    
    # Load RTE data from preprocessed parquet files
    rte_dir = data_dir / "rte"
    rte_consumption_df = None
    rte_production_df = None
    
    try:
        # Load consumption data from preprocessed parquet file
        consumption_file_parquet = rte_dir / "conso_mix_RTE_2023_processed.parquet"
        if consumption_file_parquet.exists():
            rte_consumption_df = pd.read_parquet(consumption_file_parquet)
            st.success("✅ Loaded RTE consumption data from preprocessed parquet file")
        else:
            st.warning("Preprocessed RTE consumption parquet file not found. Run data_processing.py first.")
    except Exception as e:
        st.warning(f"Could not load RTE consumption data: {e}")
    
    try:
        # Load production data from preprocessed parquet file
        production_file_parquet = rte_dir / "RealisationDonneesProduction_2023_processed.parquet"
        if production_file_parquet.exists():
            rte_production_df = pd.read_parquet(production_file_parquet)
            st.success("✅ Loaded RTE production data from preprocessed parquet file")
        else:
            st.warning("Preprocessed RTE production parquet file not found. Run data_processing.py first.")
    except Exception as e:
        st.warning(f"Could not load RTE production data: {e}")
    
    return emissions_df, co2_df, rte_consumption_df, rte_production_df


def detect_date_column(df):
    """Detect date/datetime columns in the dataframe."""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_cols.append(col)
        elif 'date' in col.lower() or 'annee' in col.lower() or 'year' in col.lower() or 'jour' in col.lower() or 'heure' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
    return date_cols[0] if date_cols else None


def detect_time_column(df):
    """Detect time/hour columns in the dataframe."""
    time_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if 'heure' in col_lower or 'hour' in col_lower or 'time' in col_lower or 'horaire' in col_lower:
            time_cols.append(col)
    return time_cols[0] if time_cols else None


def detect_numeric_columns(df, keywords=None):
    """Detect numeric columns that might represent emissions or energy."""
    if keywords is None:
        keywords = ['emission', 'co2', 'gaz', 'tonne', 'energie', 'energy', 'kwh', 'mwh', 'gwh', 'twh', 
                   'consommation', 'production', 'puissance', 'power', 'mw', 'gw', 'valeur', 'value']
    
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                numeric_cols.append(col)
    
    # If no keyword matches, return all numeric columns
    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols


def detect_categorical_columns(df):
    """Detect categorical columns that might represent sectors or categories."""
    categorical_keywords = ['secteur', 'sector', 'categorie', 'category', 'type', 'activite', 'activity', 
                           'perimetre', 'scope', 'source', 'origine', 'origin']
    
    cat_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in categorical_keywords):
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                cat_cols.append(col)
    
    # Also include low-cardinality object columns
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            if col not in cat_cols:
                cat_cols.append(col)
    
    return cat_cols


def prepare_time_series(df, date_col, value_col):
    """Prepare time series data."""
    df_ts = df.copy()
    if date_col and date_col in df_ts.columns:
        if df_ts[date_col].dtype != 'datetime64[ns]':
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts.dropna(subset=[date_col, value_col])
    return df_ts


def create_macro_overview(emissions_df, co2_df):
    """Create macro-level overview visualizations."""
    st.header("📊 Macro Overview - Total Emissions & Energy")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate totals
    emissions_numeric = detect_numeric_columns(emissions_df)
    co2_numeric = detect_numeric_columns(co2_df)
    
    total_emissions = emissions_df[emissions_numeric[0]].sum() if emissions_numeric else 0
    total_co2 = co2_df[co2_numeric[0]].sum() if co2_numeric else 0
    avg_emissions = emissions_df[emissions_numeric[0]].mean() if emissions_numeric else 0
    avg_co2 = co2_df[co2_numeric[0]].mean() if co2_numeric else 0
    
    with col1:
        st.metric("Total GHG Emissions", f"{total_emissions:,.0f}" if total_emissions > 0 else "N/A")
    with col2:
        st.metric("Total CO2 Emissions", f"{total_co2:,.0f}" if total_co2 > 0 else "N/A")
    with col3:
        st.metric("Avg GHG Emissions", f"{avg_emissions:,.0f}" if avg_emissions > 0 else "N/A")
    with col4:
        st.metric("Avg CO2 Emissions", f"{avg_co2:,.0f}" if avg_co2 > 0 else "N/A")
    
    # Time series overview
    date_col1 = detect_date_column(emissions_df)
    date_col2 = detect_date_column(co2_df)
    
    if date_col1 and emissions_numeric:
        fig = go.Figure()
        df_ts = prepare_time_series(emissions_df, date_col1, emissions_numeric[0])
        fig.add_trace(go.Scatter(
            x=df_ts[date_col1],
            y=df_ts[emissions_numeric[0]],
            mode='lines+markers',
            name='GHG Emissions',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty' if len(fig.data) > 0 else 'tozeroy'
        ))
        
        if date_col2 and co2_numeric:
            df_ts2 = prepare_time_series(co2_df, date_col2, co2_numeric[0])
            fig.add_trace(go.Scatter(
                x=df_ts2[date_col2],
                y=df_ts2[co2_numeric[0]],
                mode='lines+markers',
                name='CO2 Emissions',
                line=dict(color='#ff7f0e', width=3),
                yaxis='y2'
            ))
            fig.update_layout(yaxis2=dict(title="CO2 Emissions", overlaying='y', side='right'))
        
        fig.update_layout(
            title="Total Emissions Over Time (Macro View)",
            xaxis_title="Date",
            yaxis_title="GHG Emissions",
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


def create_sector_analysis(df, df_name="Emissions"):
    """Create sector-level analysis (meso level)."""
    st.header(f"🏭 Sector Analysis - {df_name}")
    
    categorical_cols = detect_categorical_columns(df)
    numeric_cols = detect_numeric_columns(df)
    
    if not categorical_cols or not numeric_cols:
        st.info(f"No categorical columns found for sector analysis in {df_name} data.")
        return None, None
    
    # Let user select which categorical column to use for sectors
    sector_col = st.selectbox(
        f"Select sector/category column for {df_name}:",
        categorical_cols,
        key=f"sector_col_{df_name}"
    )
    
    value_col = st.selectbox(
        f"Select value column for {df_name}:",
        numeric_cols,
        key=f"value_col_{df_name}"
    )
    
    # Aggregate by sector
    sector_summary = df.groupby(sector_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
    sector_summary = sector_summary.sort_values('sum', ascending=False)
    sector_summary.columns = [sector_col, 'Total', 'Average', 'Count']
    
    # Display top sectors
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart - top sectors
        top_n = st.slider(f"Top N sectors to display ({df_name}):", 5, 30, 15, key=f"top_n_{df_name}")
        top_sectors = sector_summary.head(top_n)
        
        fig = px.bar(
            top_sectors,
            x=sector_col,
            y='Total',
            title=f"Top {top_n} Sectors by Total {value_col}",
            labels={sector_col: 'Sector', 'Total': f'Total {value_col}'},
            color='Total',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            template='plotly_white'
        )
        fig.update_xaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart - sector distribution
        fig2 = px.pie(
            top_sectors,
            values='Total',
            names=sector_col,
            title=f"Sector Distribution (Top {top_n})",
            hole=0.4
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time series by sector
    date_col = detect_date_column(df)
    if date_col:
        st.subheader(f"Time Series by Sector - {df_name}")
        selected_sectors = st.multiselect(
            f"Select sectors to compare ({df_name}):",
            df[sector_col].unique(),
            default=list(df[sector_col].value_counts().head(5).index),
            key=f"selected_sectors_{df_name}"
        )
        
        if selected_sectors:
            fig3 = go.Figure()
            df_ts = df[df[sector_col].isin(selected_sectors)].copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
            df_ts = df_ts.sort_values(date_col)
            
            for sector in selected_sectors:
                sector_data = df_ts[df_ts[sector_col] == sector]
                sector_agg = sector_data.groupby(date_col)[value_col].sum().reset_index()
                fig3.add_trace(go.Scatter(
                    x=sector_agg[date_col],
                    y=sector_agg[value_col],
                    mode='lines+markers',
                    name=sector,
                    line=dict(width=2)
                ))
            
            fig3.update_layout(
                title=f"{value_col} Over Time by Sector",
                xaxis_title="Date",
                yaxis_title=value_col,
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    return sector_col, sector_summary


def create_micro_analysis(df, sector_col, sector_summary, df_name="Emissions"):
    """Create micro-level detailed analysis."""
    st.header(f"🔬 Micro Analysis - Detailed Breakdown ({df_name})")
    
    if sector_col is None:
        st.info("Please complete sector analysis first.")
        return
    
    # Select a sector to drill down
    selected_sector = st.selectbox(
        f"Select sector for detailed analysis ({df_name}):",
        sector_summary[sector_col].tolist(),
        key=f"selected_sector_{df_name}"
    )
    
    # Filter data for selected sector
    sector_data = df[df[sector_col] == selected_sector].copy()
    
    # Find other categorical columns for further breakdown
    all_categorical = detect_categorical_columns(sector_data)
    other_categorical = [col for col in all_categorical if col != sector_col]
    numeric_cols = detect_numeric_columns(sector_data)
    
    if other_categorical and numeric_cols:
        # Sub-category breakdown
        subcategory_col = st.selectbox(
            f"Select sub-category for {selected_sector}:",
            other_categorical,
            key=f"subcategory_{df_name}"
        )
        
        subcategory_summary = sector_data.groupby(subcategory_col)[numeric_cols[0]].agg(['sum', 'mean', 'count']).reset_index()
        subcategory_summary = subcategory_summary.sort_values('sum', ascending=False)
        subcategory_summary.columns = [subcategory_col, 'Total', 'Average', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                subcategory_summary.head(15),
                x=subcategory_col,
                y='Total',
                title=f"Breakdown of {selected_sector} by {subcategory_col}",
                labels={subcategory_col: subcategory_col, 'Total': f'Total {numeric_cols[0]}'},
                color='Total',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Detailed statistics table
            st.subheader(f"Statistics for {selected_sector}")
            st.dataframe(
                subcategory_summary.style.background_gradient(subset=['Total'], cmap='YlOrRd'),
                use_container_width=True
            )
    
    # Show raw data for selected sector
    with st.expander(f"View raw data for {selected_sector}"):
        st.dataframe(sector_data, use_container_width=True)


def create_comparison_view(emissions_df, co2_df):
    """Create comparison view between emissions and CO2."""
    st.header("⚖️ Comparative Analysis")
    
    emissions_numeric = detect_numeric_columns(emissions_df)
    co2_numeric = detect_numeric_columns(co2_df)
    emissions_cat = detect_categorical_columns(emissions_df)
    co2_cat = detect_categorical_columns(co2_df)
    
    if not emissions_numeric or not co2_numeric:
        st.info("Insufficient data for comparison.")
        return
    
    # Find common categorical columns
    common_cats = set(emissions_cat) & set(co2_cat)
    
    if common_cats:
        comparison_col = st.selectbox(
            "Select column for comparison:",
            list(common_cats)
        )
        
        # Aggregate both datasets
        emissions_agg = emissions_df.groupby(comparison_col)[emissions_numeric[0]].sum().reset_index()
        emissions_agg.columns = [comparison_col, 'GHG Emissions']
        
        co2_agg = co2_df.groupby(comparison_col)[co2_numeric[0]].sum().reset_index()
        co2_agg.columns = [comparison_col, 'CO2 Emissions']
        
        # Merge
        comparison_df = pd.merge(emissions_agg, co2_agg, on=comparison_col, how='outer')
        comparison_df = comparison_df.fillna(0)
        comparison_df = comparison_df.sort_values('GHG Emissions', ascending=False).head(15)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='GHG Emissions',
            x=comparison_df[comparison_col],
            y=comparison_df['GHG Emissions'],
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='CO2 Emissions',
            x=comparison_df[comparison_col],
            y=comparison_df['CO2 Emissions'],
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title=f"GHG vs CO2 Emissions by {comparison_col}",
            xaxis_title=comparison_col,
            yaxis_title="Emissions",
            barmode='group',
            height=500,
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation if both have same categories
        if len(comparison_df) > 1:
            correlation = comparison_df['GHG Emissions'].corr(comparison_df['CO2 Emissions'])
            st.metric("Correlation between GHG and CO2 Emissions", f"{correlation:.3f}")


def create_peak_analysis(df, date_col, value_col):
    """Analyze peak and down timings in energy consumption/production."""
    st.header("⏰ Peak & Down Timing Analysis")
    
    if df is None or date_col is None or value_col is None:
        st.info("No data available for peak analysis.")
        return
    
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[date_col, value_col])
    df_clean = df_clean.sort_values(date_col)
    
    # Extract time components
    df_clean['hour'] = df_clean[date_col].dt.hour
    df_clean['day_of_week'] = df_clean[date_col].dt.day_name()
    df_clean['month'] = df_clean[date_col].dt.month
    df_clean['day'] = df_clean[date_col].dt.date
    
    col1, col2, col3 = st.columns(3)
    
    # Find peaks and downs
    max_idx = df_clean[value_col].idxmax()
    min_idx = df_clean[value_col].idxmin()
    
    with col1:
        st.metric("Peak Value", f"{df_clean.loc[max_idx, value_col]:,.2f}")
        st.caption(f"Date: {df_clean.loc[max_idx, date_col]}")
    with col2:
        st.metric("Minimum Value", f"{df_clean.loc[min_idx, value_col]:,.2f}")
        st.caption(f"Date: {df_clean.loc[min_idx, date_col]}")
    with col3:
        st.metric("Average Value", f"{df_clean[value_col].mean():,.2f}")
        st.caption(f"Range: {df_clean[value_col].max() - df_clean[value_col].min():,.2f}")
    
    # Hourly pattern
    st.subheader("Hourly Pattern - Peak Times")
    hourly_avg = df_clean.groupby('hour')[value_col].agg(['mean', 'max', 'min']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['mean'],
        mode='lines+markers',
        name='Average',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty'
    ))
    fig.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['max'],
        mode='lines',
        name='Maximum',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=hourly_avg['hour'],
        y=hourly_avg['min'],
        mode='lines',
        name='Minimum',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))
    
    # Highlight peak hour
    peak_hour = hourly_avg.loc[hourly_avg['mean'].idxmax(), 'hour']
    fig.add_vline(x=peak_hour, line_dash="dot", line_color="red", 
                  annotation_text=f"Peak Hour: {int(peak_hour)}h")
    
    fig.update_layout(
        title="Average Consumption/Production by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title=value_col,
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week pattern
    st.subheader("Day of Week Pattern")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df_clean.groupby('day_of_week')[value_col].mean().reindex(day_order).reset_index()
    
    fig2 = px.bar(
        daily_avg,
        x='day_of_week',
        y=value_col,
        title="Average by Day of Week",
        labels={'day_of_week': 'Day of Week', value_col: value_col},
        color=value_col,
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(height=400, template='plotly_white', xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly pattern
    st.subheader("Monthly Pattern")
    monthly_avg = df_clean.groupby('month')[value_col].agg(['mean', 'max', 'min']).reset_index()
    monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%B')
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['mean'],
        name='Average',
        marker_color='steelblue'
    ))
    fig3.add_trace(go.Scatter(
        x=monthly_avg['month_name'],
        y=monthly_avg['max'],
        mode='markers',
        name='Peak',
        marker=dict(size=10, color='red', symbol='triangle-up')
    ))
    fig3.add_trace(go.Scatter(
        x=monthly_avg['month_name'],
        y=monthly_avg['min'],
        mode='markers',
        name='Minimum',
        marker=dict(size=10, color='green', symbol='triangle-down')
    ))
    fig3.update_layout(
        title="Monthly Average with Peaks and Minimums",
        xaxis_title="Month",
        yaxis_title=value_col,
        height=400,
        barmode='group',
        xaxis_tickangle=-45,
        template='plotly_white'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Time series with peaks highlighted
    st.subheader("Time Series with Peak Identification")
    df_clean['is_peak'] = df_clean[value_col] >= df_clean[value_col].quantile(0.95)
    df_clean['is_low'] = df_clean[value_col] <= df_clean[value_col].quantile(0.05)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_clean[date_col],
        y=df_clean[value_col],
        mode='lines',
        name='Normal',
        line=dict(color='lightblue', width=1)
    ))
    fig4.add_trace(go.Scatter(
        x=df_clean[df_clean['is_peak']][date_col],
        y=df_clean[df_clean['is_peak']][value_col],
        mode='markers',
        name='Peak (Top 5%)',
        marker=dict(size=8, color='red', symbol='circle')
    ))
    fig4.add_trace(go.Scatter(
        x=df_clean[df_clean['is_low']][date_col],
        y=df_clean[df_clean['is_low']][value_col],
        mode='markers',
        name='Low (Bottom 5%)',
        marker=dict(size=8, color='green', symbol='circle')
    ))
    fig4.update_layout(
        title="Time Series with Peak and Low Periods Highlighted",
        xaxis_title="Date",
        yaxis_title=value_col,
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig4, use_container_width=True)


def create_production_by_sector(df):
    """Create production analysis by sector."""
    st.header("🏭 Production by Sector")
    
    if df is None:
        st.info("No production data available.")
        return
    
    # Detect sector/category columns
    categorical_cols = detect_categorical_columns(df)
    numeric_cols = detect_numeric_columns(df, keywords=['production', 'puissance', 'power', 'mw', 'gw', 'valeur', 'value', 'energie'])
    
    if not categorical_cols or not numeric_cols:
        st.info("Could not identify sector or production columns. Showing available columns:")
        st.write("Categorical:", categorical_cols if categorical_cols else "None")
        st.write("Numeric:", numeric_cols if numeric_cols else "None")
        st.write("All columns:", df.columns.tolist())
        return
    
    # Let user select columns
    sector_col = st.selectbox("Select sector/category column:", categorical_cols, key="rte_sector_col")
    value_col = st.selectbox("Select production/value column:", numeric_cols, key="rte_value_col")
    
    # Aggregate by sector
    sector_summary = df.groupby(sector_col)[value_col].agg(['sum', 'mean', 'max', 'min', 'count']).reset_index()
    sector_summary = sector_summary.sort_values('sum', ascending=False)
    sector_summary.columns = [sector_col, 'Total', 'Average', 'Maximum', 'Minimum', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        top_n = st.slider("Top N sectors:", 5, 30, 15, key="rte_top_n")
        top_sectors = sector_summary.head(top_n)
        
        fig = px.bar(
            top_sectors,
            x=sector_col,
            y='Total',
            title=f"Total Production by Sector (Top {top_n})",
            labels={sector_col: 'Sector', 'Total': 'Total Production'},
            color='Total',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig2 = px.pie(
            top_sectors,
            values='Total',
            names=sector_col,
            title=f"Sector Distribution (Top {top_n})",
            hole=0.4
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time series by sector
    date_col = detect_date_column(df)
    if date_col:
        st.subheader("Production Over Time by Sector")
        selected_sectors = st.multiselect(
            "Select sectors to compare:",
            df[sector_col].unique(),
            default=list(df[sector_col].value_counts().head(5).index),
            key="rte_selected_sectors"
        )
        
        if selected_sectors:
            fig3 = go.Figure()
            df_ts = df[df[sector_col].isin(selected_sectors)].copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
            df_ts = df_ts.sort_values(date_col)
            
            for sector in selected_sectors:
                sector_data = df_ts[df_ts[sector_col] == sector]
                sector_agg = sector_data.groupby(date_col)[value_col].sum().reset_index()
                fig3.add_trace(go.Scatter(
                    x=sector_agg[date_col],
                    y=sector_agg[value_col],
                    mode='lines+markers',
                    name=sector,
                    line=dict(width=2)
                ))
            
            fig3.update_layout(
                title=f"{value_col} Over Time by Sector",
                xaxis_title="Date",
                yaxis_title=value_col,
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    # Summary table
    st.subheader("Sector Summary Statistics")
    st.dataframe(
        sector_summary.style.background_gradient(subset=['Total', 'Average'], cmap='YlOrRd'),
        use_container_width=True
    )


def create_stacked_production_chart(rte_production_df):
    """Create stacked area chart showing production by sector over time."""
    if rte_production_df is None:
        return None
    
    date_col = detect_date_column(rte_production_df)
    if date_col is None:
        return None
    
    # Get production sector columns (exclude date, Heures, Total)
    exclude_cols = ['date', 'Heures', 'Total']
    production_sectors = [col for col in rte_production_df.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(rte_production_df[col])]
    
    if not production_sectors:
        return None
    
    # Prepare data
    df = rte_production_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    
    # Create datetime column combining date and hour if Heures column exists
    time_col = detect_time_column(df)
    if time_col and 'Heures' in df.columns:
        # Extract hour from Heures column (format: "00:00-01:00" or "00:00")
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if '-' in hour_str:
                return int(hour_str.split('-')[0].split(':')[0])
            elif ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        
        df['hour'] = df['Heures'].apply(extract_hour)
        df['datetime'] = df[date_col] + pd.to_timedelta(df['hour'], unit='h')
    else:
        df['datetime'] = df[date_col]
    
    # Aggregate by datetime (sum all sectors for each time point)
    df_agg = df.groupby('datetime')[production_sectors].sum().reset_index()
    df_agg = df_agg.sort_values('datetime')
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Color palette for sectors
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    # Sort sectors by total contribution (for better visualization)
    sector_totals = df_agg[production_sectors].sum().sort_values(ascending=False)
    sorted_sectors = sector_totals.index.tolist()
    
    # Add traces with stackgroup (Plotly handles stacking automatically)
    for idx, sector in enumerate(sorted_sectors):
        values = df_agg[sector].fillna(0).values
        fig.add_trace(go.Scatter(
            x=df_agg['datetime'],
            y=values,
            mode='lines',
            name=sector,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)],
            line=dict(width=0.5),
            hovertemplate=f'<b>{sector}</b><br>' +
                         f'Date: %{{x}}<br>' +
                         f'Value: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Production Over Time by Sector (Stacked)",
        xaxis_title="Date",
        yaxis_title="Production (MW)",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_stacked_consumption_chart(rte_consumption_df):
    """Create stacked area chart showing consumption over time."""
    if rte_consumption_df is None:
        return None
    
    date_col = detect_date_column(rte_consumption_df)
    if date_col is None:
        return None
    
    # Get consumption columns (exclude date, Heures)
    exclude_cols = ['date', 'Heures']
    consumption_cols = [col for col in rte_consumption_df.columns 
                        if col not in exclude_cols and pd.api.types.is_numeric_dtype(rte_consumption_df[col])]
    
    if not consumption_cols:
        return None
    
    # Prepare data
    df = rte_consumption_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    
    # Create datetime column combining date and hour if Heures column exists
    time_col = detect_time_column(df)
    if time_col and 'Heures' in df.columns:
        # Extract hour from Heures column
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        
        df['hour'] = df['Heures'].apply(extract_hour)
        df['datetime'] = df[date_col] + pd.to_timedelta(df['hour'], unit='h')
    else:
        df['datetime'] = df[date_col]
    
    # Aggregate by datetime
    df_agg = df.groupby('datetime')[consumption_cols].sum().reset_index()
    df_agg = df_agg.sort_values('datetime')
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Set2
    
    # Sort columns by total contribution
    col_totals = df_agg[consumption_cols].sum().sort_values(ascending=False)
    sorted_cols = col_totals.index.tolist()
    
    # Add traces with stackgroup (Plotly handles stacking automatically)
    for idx, col in enumerate(sorted_cols):
        values = df_agg[col].fillna(0).values
        fig.add_trace(go.Scatter(
            x=df_agg['datetime'],
            y=values,
            mode='lines',
            name=col,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)],
            line=dict(width=0.5),
            hovertemplate=f'<b>{col}</b><br>' +
                         f'Date: %{{x}}<br>' +
                         f'Value: %{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Consumption Over Time (Stacked)",
        xaxis_title="Date",
        yaxis_title="Consumption (MW)",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_production_consumption_balance(rte_production_df, rte_consumption_df):
    """Create comparison chart showing production vs consumption balance over time."""
    if rte_production_df is None or rte_consumption_df is None:
        return None
    
    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    
    if date_col_prod is None or date_col_cons is None:
        return None
    
    # Prepare production data
    df_prod = rte_production_df.copy()
    df_prod[date_col_prod] = pd.to_datetime(df_prod[date_col_prod], errors='coerce')
    
    # Prepare consumption data
    df_cons = rte_consumption_df.copy()
    df_cons[date_col_cons] = pd.to_datetime(df_cons[date_col_cons], errors='coerce')
    
    # Create datetime columns
    if 'Heures' in df_prod.columns:
        def extract_hour_prod(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if '-' in hour_str:
                return int(hour_str.split('-')[0].split(':')[0])
            elif ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_prod['hour'] = df_prod['Heures'].apply(extract_hour_prod)
        df_prod['datetime'] = df_prod[date_col_prod] + pd.to_timedelta(df_prod['hour'], unit='h')
    else:
        df_prod['datetime'] = df_prod[date_col_prod]
    
    if 'Heures' in df_cons.columns:
        def extract_hour_cons(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_cons['hour'] = df_cons['Heures'].apply(extract_hour_cons)
        df_cons['datetime'] = df_cons[date_col_cons] + pd.to_timedelta(df_cons['hour'], unit='h')
    else:
        df_cons['datetime'] = df_cons[date_col_cons]
    
    # Get total production and consumption
    prod_total_col = 'Total' if 'Total' in df_prod.columns else None
    if prod_total_col is None:
        exclude_cols = ['date', 'Heures', 'datetime', 'hour']
        prod_cols = [col for col in df_prod.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_prod[col])]
        df_prod['Total_Production'] = df_prod[prod_cols].sum(axis=1)
        prod_total_col = 'Total_Production'
    
    cons_col = 'Consommation' if 'Consommation' in df_cons.columns else None
    if cons_col is None:
        exclude_cols = ['date', 'Heures', 'datetime', 'hour']
        cons_cols = [col for col in df_cons.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_cons[col])]
        if cons_cols:
            cons_col = cons_cols[0]
    
    if cons_col is None:
        return None
    
    # Aggregate by datetime
    prod_agg = df_prod.groupby('datetime')[prod_total_col].sum().reset_index()
    cons_agg = df_cons.groupby('datetime')[cons_col].sum().reset_index()
    
    # Merge
    balance_df = pd.merge(prod_agg, cons_agg, on='datetime', how='outer')
    balance_df = balance_df.sort_values('datetime')
    balance_df['Balance'] = balance_df[prod_total_col] - balance_df[cons_col]
    balance_df['Balance_Percent'] = (balance_df['Balance'] / balance_df[cons_col] * 100).fillna(0)
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Production vs Consumption Over Time', 'Balance (Production - Consumption)'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Production and Consumption
    fig.add_trace(
        go.Scatter(
            x=balance_df['datetime'],
            y=balance_df[prod_total_col],
            mode='lines',
            name='Total Production',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.3)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=balance_df['datetime'],
            y=balance_df[cons_col],
            mode='lines',
            name='Consumption',
            line=dict(color='#d62728', width=2),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.3)'
        ),
        row=1, col=1
    )
    
    # Balance
    fig.add_trace(
        go.Scatter(
            x=balance_df['datetime'],
            y=balance_df['Balance'],
            mode='lines',
            name='Balance',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ),
        row=2, col=1
    )
    
    # Add zero line for balance
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Balance (MW)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_hourly_sector_comparison(rte_production_df, rte_consumption_df):
    """Create comparison chart showing production and consumption by hour and sector over extended period."""
    if rte_production_df is None or rte_consumption_df is None:
        return None
    
    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    
    if date_col_prod is None or date_col_cons is None:
        return None
    
    # Prepare production data
    df_prod = rte_production_df.copy()
    df_prod[date_col_prod] = pd.to_datetime(df_prod[date_col_prod], errors='coerce')
    
    # Extract hour from Heures column
    if 'Heures' in df_prod.columns:
        def extract_hour_prod(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if '-' in hour_str:
                return int(hour_str.split('-')[0].split(':')[0])
            elif ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_prod['hour'] = df_prod['Heures'].apply(extract_hour_prod)
    else:
        df_prod['hour'] = df_prod[date_col_prod].dt.hour
    
    # Prepare consumption data
    df_cons = rte_consumption_df.copy()
    df_cons[date_col_cons] = pd.to_datetime(df_cons[date_col_cons], errors='coerce')
    
    if 'Heures' in df_cons.columns:
        def extract_hour_cons(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_cons['hour'] = df_cons['Heures'].apply(extract_hour_cons)
    else:
        df_cons['hour'] = df_cons[date_col_cons].dt.hour
    
    # Get production sectors
    exclude_cols = ['date', 'Heures', 'Total', 'hour']
    production_sectors = [col for col in df_prod.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_prod[col])]
    
    # Get consumption column
    cons_col = 'Consommation' if 'Consommation' in df_cons.columns else None
    if cons_col is None:
        exclude_cols = ['date', 'Heures', 'hour']
        cons_cols = [col for col in df_cons.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_cons[col])]
        if cons_cols:
            cons_col = cons_cols[0]
    
    if not production_sectors or cons_col is None:
        return None
    
    # Aggregate by hour
    prod_hourly = df_prod.groupby('hour')[production_sectors].mean().reset_index()
    cons_hourly = df_cons.groupby('hour')[cons_col].mean().reset_index()
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Production by Hour and Sector', 'Average Consumption by Hour'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Production by sector and hour (stacked)
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    # Sort sectors by total contribution
    sector_totals = prod_hourly[production_sectors].sum().sort_values(ascending=False)
    sorted_sectors = sector_totals.index.tolist()[:10]  # Limit to top 10 sectors for clarity
    
    for idx, sector in enumerate(sorted_sectors):
        values = prod_hourly[sector].fillna(0).values
        fig.add_trace(
            go.Scatter(
                x=prod_hourly['hour'],
                y=values,
                mode='lines',
                name=sector,
                stackgroup='one',
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5)
            ),
            row=1, col=1
        )
    
    # Consumption by hour
    fig.add_trace(
        go.Scatter(
            x=cons_hourly['hour'],
            y=cons_hourly[cons_col],
            mode='lines+markers',
            name='Consumption',
            line=dict(color='#d62728', width=3),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_yaxes(title_text="Production (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Consumption (MW)", row=1, col=2)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_weekly_production_chart(rte_production_df, week_type='average', selected_week=None):
    """Create stacked area chart showing production by sector for a week (per hour)."""
    if rte_production_df is None:
        return None
    
    date_col = detect_date_column(rte_production_df)
    if date_col is None:
        return None
    
    # Get production sector columns
    exclude_cols = ['date', 'Heures', 'Total']
    production_sectors = [col for col in rte_production_df.columns 
                         if col not in exclude_cols and pd.api.types.is_numeric_dtype(rte_production_df[col])]
    
    if not production_sectors:
        return None
    
    # Prepare data
    df = rte_production_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    
    # Extract hour from Heures column
    if 'Heures' in df.columns:
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if '-' in hour_str:
                return int(hour_str.split('-')[0].split(':')[0])
            elif ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df['hour'] = df['Heures'].apply(extract_hour)
    else:
        df['hour'] = df[date_col].dt.hour
    
    df['day_of_week'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df[date_col].dt.day_name()
    df['week'] = df[date_col].dt.isocalendar().week
    df['year'] = df[date_col].dt.year
    
    # Calculate hour of week (0-167: Monday 00:00 to Sunday 23:00)
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour']
    
    if week_type == 'average':
        # Average across all weeks
        df_agg = df.groupby('hour_of_week')[production_sectors].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        # Specific week
        if selected_week is None:
            # Use first available week
            selected_week = df['week'].iloc[0]
            selected_year = df['year'].iloc[0]
        else:
            selected_year = df[df['week'] == selected_week]['year'].iloc[0] if len(df[df['week'] == selected_week]) > 0 else df['year'].iloc[0]
        
        df_week = df[(df['week'] == selected_week) & (df['year'] == selected_year)]
        df_agg = df_week.groupby('hour_of_week')[production_sectors].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"
    
    # Create labels for x-axis (day and hour)
    df_agg['day_of_week'] = df_agg['hour_of_week'] // 24
    df_agg['hour'] = df_agg['hour_of_week'] % 24
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_agg['label'] = df_agg.apply(lambda row: f"{day_names[int(row['day_of_week'])]} {int(row['hour']):02d}:00", axis=1)
    
    # Create stacked area chart
    fig = go.Figure()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    # Sort sectors by total contribution
    sector_totals = df_agg[production_sectors].sum().sort_values(ascending=False)
    sorted_sectors = sector_totals.index.tolist()
    
    for idx, sector in enumerate(sorted_sectors):
        values = df_agg[sector].fillna(0).values
        fig.add_trace(go.Scatter(
            x=df_agg['hour_of_week'],
            y=values,
            mode='lines',
            name=sector,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)],
            line=dict(width=0.5),
            hovertemplate=f'<b>{sector}</b><br>' +
                         f'%{{customdata}}<br>' +
                         f'Value: %{{y:,.0f}}<extra></extra>',
            customdata=df_agg['label']
        ))
    
    # Update x-axis with day labels
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=24,
        ticktext=day_names,
        tickvals=list(range(0, 168, 24))
    )
    
    fig.update_layout(
        title=f"Production by Sector per Hour{title_suffix}",
        xaxis_title="Day of Week",
        yaxis_title="Production (MW)",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_weekly_consumption_chart(rte_consumption_df, week_type='average', selected_week=None):
    """Create stacked area chart showing consumption for a week (per hour)."""
    if rte_consumption_df is None:
        return None
    
    date_col = detect_date_column(rte_consumption_df)
    if date_col is None:
        return None
    
    # Get consumption columns
    exclude_cols = ['date', 'Heures']
    consumption_cols = [col for col in rte_consumption_df.columns 
                        if col not in exclude_cols and pd.api.types.is_numeric_dtype(rte_consumption_df[col])]
    
    if not consumption_cols:
        return None
    
    # Prepare data
    df = rte_consumption_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    
    # Extract hour from Heures column
    if 'Heures' in df.columns:
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df['hour'] = df['Heures'].apply(extract_hour)
    else:
        df['hour'] = df[date_col].dt.hour
    
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week'] = df[date_col].dt.isocalendar().week
    df['year'] = df[date_col].dt.year
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour']
    
    if week_type == 'average':
        df_agg = df.groupby('hour_of_week')[consumption_cols].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        if selected_week is None:
            selected_week = df['week'].iloc[0]
            selected_year = df['year'].iloc[0]
        else:
            selected_year = df[df['week'] == selected_week]['year'].iloc[0] if len(df[df['week'] == selected_week]) > 0 else df['year'].iloc[0]
        
        df_week = df[(df['week'] == selected_week) & (df['year'] == selected_year)]
        df_agg = df_week.groupby('hour_of_week')[consumption_cols].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"
    
    # Create labels
    df_agg['day_of_week'] = df_agg['hour_of_week'] // 24
    df_agg['hour'] = df_agg['hour_of_week'] % 24
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_agg['label'] = df_agg.apply(lambda row: f"{day_names[int(row['day_of_week'])]} {int(row['hour']):02d}:00", axis=1)
    
    # Create stacked area chart
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    col_totals = df_agg[consumption_cols].sum().sort_values(ascending=False)
    sorted_cols = col_totals.index.tolist()
    
    for idx, col in enumerate(sorted_cols):
        values = df_agg[col].fillna(0).values
        fig.add_trace(go.Scatter(
            x=df_agg['hour_of_week'],
            y=values,
            mode='lines',
            name=col,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)],
            line=dict(width=0.5),
            hovertemplate=f'<b>{col}</b><br>' +
                         f'%{{customdata}}<br>' +
                         f'Value: %{{y:,.0f}}<extra></extra>',
            customdata=df_agg['label']
        ))
    
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=24,
        ticktext=day_names,
        tickvals=list(range(0, 168, 24))
    )
    
    fig.update_layout(
        title=f"Consumption per Hour{title_suffix}",
        xaxis_title="Day of Week",
        yaxis_title="Consumption (MW)",
        height=600,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_weekly_balance_chart(rte_production_df, rte_consumption_df, week_type='average', selected_week=None):
    """Create balance chart showing production vs consumption for a week (per hour)."""
    if rte_production_df is None or rte_consumption_df is None:
        return None
    
    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    
    if date_col_prod is None or date_col_cons is None:
        return None
    
    # Prepare production data
    df_prod = rte_production_df.copy()
    df_prod[date_col_prod] = pd.to_datetime(df_prod[date_col_prod], errors='coerce')
    
    # Prepare consumption data
    df_cons = rte_consumption_df.copy()
    df_cons[date_col_cons] = pd.to_datetime(df_cons[date_col_cons], errors='coerce')
    
    # Extract hours
    if 'Heures' in df_prod.columns:
        def extract_hour_prod(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if '-' in hour_str:
                return int(hour_str.split('-')[0].split(':')[0])
            elif ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_prod['hour'] = df_prod['Heures'].apply(extract_hour_prod)
    else:
        df_prod['hour'] = df_prod[date_col_prod].dt.hour
    
    if 'Heures' in df_cons.columns:
        def extract_hour_cons(hour_str):
            if pd.isna(hour_str):
                return 0
            hour_str = str(hour_str)
            if ':' in hour_str:
                return int(hour_str.split(':')[0])
            return 0
        df_cons['hour'] = df_cons['Heures'].apply(extract_hour_cons)
    else:
        df_cons['hour'] = df_cons[date_col_cons].dt.hour
    
    # Calculate hour of week
    df_prod['day_of_week'] = df_prod[date_col_prod].dt.dayofweek
    df_prod['week'] = df_prod[date_col_prod].dt.isocalendar().week
    df_prod['year'] = df_prod[date_col_prod].dt.year
    df_prod['hour_of_week'] = df_prod['day_of_week'] * 24 + df_prod['hour']
    
    df_cons['day_of_week'] = df_cons[date_col_cons].dt.dayofweek
    df_cons['week'] = df_cons[date_col_cons].dt.isocalendar().week
    df_cons['year'] = df_cons[date_col_cons].dt.year
    df_cons['hour_of_week'] = df_cons['day_of_week'] * 24 + df_cons['hour']
    
    # Get totals
    prod_total_col = 'Total' if 'Total' in df_prod.columns else None
    if prod_total_col is None:
        exclude_cols = ['date', 'Heures', 'hour', 'day_of_week', 'week', 'year', 'hour_of_week']
        prod_cols = [col for col in df_prod.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_prod[col])]
        df_prod['Total_Production'] = df_prod[prod_cols].sum(axis=1)
        prod_total_col = 'Total_Production'
    
    cons_col = 'Consommation' if 'Consommation' in df_cons.columns else None
    if cons_col is None:
        exclude_cols = ['date', 'Heures', 'hour', 'day_of_week', 'week', 'year', 'hour_of_week']
        cons_cols = [col for col in df_cons.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_cons[col])]
        if cons_cols:
            cons_col = cons_cols[0]
    
    if cons_col is None:
        return None
    
    # Aggregate by hour of week
    if week_type == 'average':
        prod_agg = df_prod.groupby('hour_of_week')[prod_total_col].mean().reset_index()
        cons_agg = df_cons.groupby('hour_of_week')[cons_col].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        if selected_week is None:
            selected_week = df_prod['week'].iloc[0]
            selected_year = df_prod['year'].iloc[0]
        else:
            selected_year = df_prod[df_prod['week'] == selected_week]['year'].iloc[0] if len(df_prod[df_prod['week'] == selected_week]) > 0 else df_prod['year'].iloc[0]
        
        df_prod_week = df_prod[(df_prod['week'] == selected_week) & (df_prod['year'] == selected_year)]
        df_cons_week = df_cons[(df_cons['week'] == selected_week) & (df_cons['year'] == selected_year)]
        prod_agg = df_prod_week.groupby('hour_of_week')[prod_total_col].mean().reset_index()
        cons_agg = df_cons_week.groupby('hour_of_week')[cons_col].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"
    
    # Merge
    balance_df = pd.merge(prod_agg, cons_agg, on='hour_of_week', how='outer')
    balance_df = balance_df.sort_values('hour_of_week')
    balance_df['Balance'] = balance_df[prod_total_col] - balance_df[cons_col]
    
    # Create labels
    balance_df['day_of_week'] = balance_df['hour_of_week'] // 24
    balance_df['hour'] = balance_df['hour_of_week'] % 24
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    balance_df['label'] = balance_df.apply(lambda row: f"{day_names[int(row['day_of_week'])]} {int(row['hour']):02d}:00", axis=1)
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Production vs Consumption per Hour{title_suffix}', 'Balance (Production - Consumption)'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Production and Consumption
    fig.add_trace(
        go.Scatter(
            x=balance_df['hour_of_week'],
            y=balance_df[prod_total_col],
            mode='lines',
            name='Total Production',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.3)',
            hovertemplate='<b>Production</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>',
            customdata=balance_df['label']
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=balance_df['hour_of_week'],
            y=balance_df[cons_col],
            mode='lines',
            name='Consumption',
            line=dict(color='#d62728', width=2),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='<b>Consumption</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>',
            customdata=balance_df['label']
        ),
        row=1, col=1
    )
    
    # Balance
    fig.add_trace(
        go.Scatter(
            x=balance_df['hour_of_week'],
            y=balance_df['Balance'],
            mode='lines',
            name='Balance',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)',
            hovertemplate='<b>Balance</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>',
            customdata=balance_df['label']
        ),
        row=2, col=1
    )
    
    # Add zero line for balance
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update x-axes
    day_names_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=24,
        ticktext=day_names_short,
        tickvals=list(range(0, 168, 24)),
        row=1, col=1
    )
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=24,
        ticktext=day_names_short,
        tickvals=list(range(0, 168, 24)),
        title_text="Day of Week",
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Balance (MW)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_micro_rte_analysis(rte_consumption_df, rte_production_df):
    """Create micro RTE analysis showing weekly patterns per hour."""
    st.header("🔬 Micro RTE Analysis - Weekly Patterns per Hour")
    
    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE data available for micro analysis.")
        return
    
    # Week selection
    col1, col2 = st.columns(2)
    
    with col1:
        week_type = st.radio(
            "Select week type:",
            ["Average Week", "Specific Week"],
            key="rte_week_type"
        )
    
    selected_week = None
    if week_type == "Specific Week":
        # Get available weeks
        if rte_production_df is not None:
            date_col = detect_date_column(rte_production_df)
            if date_col:
                df_temp = rte_production_df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp['week'] = df_temp[date_col].dt.isocalendar().week
                df_temp['year'] = df_temp[date_col].dt.year
                available_weeks = sorted(df_temp[['week', 'year']].drop_duplicates().values.tolist())
                
                with col2:
                    if available_weeks:
                        week_options = [f"Week {w[0]}, {w[1]}" for w in available_weeks]
                        selected_week_str = st.selectbox("Select week:", week_options, key="rte_selected_week")
                        selected_week = int(selected_week_str.split(',')[0].split()[1])
    
    week_type_param = 'average' if week_type == "Average Week" else 'specific'
    
    # Production chart
    if rte_production_df is not None:
        st.subheader("📈 Production by Sector per Hour")
        fig_prod = create_weekly_production_chart(rte_production_df, week_type_param, selected_week)
        if fig_prod:
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Could not create weekly production chart.")
    
    # Consumption chart
    if rte_consumption_df is not None:
        st.subheader("📊 Consumption per Hour")
        fig_cons = create_weekly_consumption_chart(rte_consumption_df, week_type_param, selected_week)
        if fig_cons:
            st.plotly_chart(fig_cons, use_container_width=True)
        else:
            st.info("Could not create weekly consumption chart.")
    
    # Balance chart
    if rte_production_df is not None and rte_consumption_df is not None:
        st.subheader("⚖️ Production vs Consumption Balance per Hour")
        fig_balance = create_weekly_balance_chart(rte_production_df, rte_consumption_df, week_type_param, selected_week)
        if fig_balance:
            st.plotly_chart(fig_balance, use_container_width=True)
        else:
            st.info("Could not create weekly balance chart.")


def create_rte_analysis(rte_consumption_df, rte_production_df):
    """Create comprehensive RTE energy analysis with stacked charts and balance analysis."""
    st.header("⚡ RTE Energy Analysis")
    
    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE data available. Please ensure RTE data files are in the data/rte/ folder.")
        return
    
    # Production over time (stacked by sector)
    if rte_production_df is not None:
        st.subheader("📈 Production Over Time by Sector (Stacked)")
        fig_prod = create_stacked_production_chart(rte_production_df)
        if fig_prod:
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Could not create production chart. Check data structure.")
    
    # Consumption over time (stacked)
    if rte_consumption_df is not None:
        st.subheader("📊 Consumption Over Time (Stacked)")
        fig_cons = create_stacked_consumption_chart(rte_consumption_df)
        if fig_cons:
            st.plotly_chart(fig_cons, use_container_width=True)
        else:
            st.info("Could not create consumption chart. Check data structure.")
    
    # Production vs Consumption Balance
    if rte_production_df is not None and rte_consumption_df is not None:
        st.subheader("⚖️ Production vs Consumption Balance Over Time")
        fig_balance = create_production_consumption_balance(rte_production_df, rte_consumption_df)
        if fig_balance:
            st.plotly_chart(fig_balance, use_container_width=True)
        else:
            st.info("Could not create balance chart. Check data structure.")
    
    # Hourly comparison by sector
    if rte_production_df is not None and rte_consumption_df is not None:
        st.subheader("🕐 Hourly Patterns: Production by Sector vs Consumption")
        fig_hourly = create_hourly_sector_comparison(rte_production_df, rte_consumption_df)
        if fig_hourly:
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("Could not create hourly comparison chart. Check data structure.")


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">⚡ SNCF & RTE Energy Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Granular Analysis: From Macro to Micro View | Energy Consumption & Production")
    
    # Load data
    with st.spinner("Loading data..."):
        emissions_df, co2_df, rte_consumption_df, rte_production_df = load_data()
    
    # Sidebar filters
    st.sidebar.header("📋 Data Overview")
    st.sidebar.write(f"**Emissions Data:** {emissions_df.shape[0]} rows, {emissions_df.shape[1]} columns")
    st.sidebar.write(f"**CO2 Data:** {co2_df.shape[0]} rows, {co2_df.shape[1]} columns")
    
    if rte_consumption_df is not None:
        st.sidebar.write(f"**RTE Consumption:** {rte_consumption_df.shape[0]} rows, {rte_consumption_df.shape[1]} columns")
    if rte_production_df is not None:
        st.sidebar.write(f"**RTE Production:** {rte_production_df.shape[0]} rows, {rte_production_df.shape[1]} columns")
    
    # Show data preview
    with st.sidebar.expander("View Data Columns"):
        st.write("**Emissions columns:**", list(emissions_df.columns))
        st.write("**CO2 columns:**", list(co2_df.columns))
        if rte_consumption_df is not None:
            st.write("**RTE Consumption columns:**", list(rte_consumption_df.columns))
        if rte_production_df is not None:
            st.write("**RTE Production columns:**", list(rte_production_df.columns))
    
    # Analysis level selector
    analysis_options = ["Macro Overview", "Sector Analysis", "Micro Analysis", "Comparison"]
    if rte_consumption_df is not None or rte_production_df is not None:
        analysis_options.append("RTE Energy Analysis")
        analysis_options.append("Micro RTE Analysis")
    
    analysis_level = st.sidebar.radio(
        "Analysis Level",
        analysis_options,
        index=0
    )
    
    # Main content based on selected level
    if analysis_level == "Macro Overview":
        create_macro_overview(emissions_df, co2_df)
        
        # Show data summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emissions Data Summary")
            st.dataframe(emissions_df.describe(), use_container_width=True)
        with col2:
            st.subheader("CO2 Data Summary")
            st.dataframe(co2_df.describe(), use_container_width=True)
    
    elif analysis_level == "Sector Analysis":
        tab1, tab2 = st.tabs(["GHG Emissions", "CO2 Emissions"])
        
        with tab1:
            sector_col1, sector_summary1 = create_sector_analysis(emissions_df, "GHG Emissions")
            st.session_state['emissions_sector_col'] = sector_col1
            st.session_state['emissions_sector_summary'] = sector_summary1
        
        with tab2:
            sector_col2, sector_summary2 = create_sector_analysis(co2_df, "CO2 Emissions")
            st.session_state['co2_sector_col'] = sector_col2
            st.session_state['co2_sector_summary'] = sector_summary2
    
    elif analysis_level == "Micro Analysis":
        tab1, tab2 = st.tabs(["GHG Emissions", "CO2 Emissions"])
        
        with tab1:
            sector_col = st.session_state.get('emissions_sector_col')
            sector_summary = st.session_state.get('emissions_sector_summary')
            if sector_col and sector_summary is not None:
                create_micro_analysis(emissions_df, sector_col, sector_summary, "GHG Emissions")
            else:
                st.info("Please complete Sector Analysis for GHG Emissions first.")
        
        with tab2:
            sector_col = st.session_state.get('co2_sector_col')
            sector_summary = st.session_state.get('co2_sector_summary')
            if sector_col and sector_summary is not None:
                create_micro_analysis(co2_df, sector_col, sector_summary, "CO2 Emissions")
            else:
                st.info("Please complete Sector Analysis for CO2 Emissions first.")
    
    elif analysis_level == "Comparison":
        create_comparison_view(emissions_df, co2_df)
    
    elif analysis_level == "RTE Energy Analysis":
        create_rte_analysis(rte_consumption_df, rte_production_df)
    
    elif analysis_level == "Micro RTE Analysis":
        create_micro_rte_analysis(rte_consumption_df, rte_production_df)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Sources:** SNCF Greenhouse Gas Emissions, CO2 Data, and RTE Energy Data")


if __name__ == "__main__":
    main()
