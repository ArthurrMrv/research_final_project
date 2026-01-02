"""
SNCF Data Visualization - Streamlit App
Interactive visualizations for SNCF greenhouse gas emissions and CO2 data
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
    page_title="SNCF Emissions & Energy Analysis",
    page_icon="🚂",
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
    """Load the SNCF parquet data files."""
    data_dir = Path(__file__).parent / "data"
    
    # Load greenhouse gas emissions balance
    emissions_df = pd.read_parquet(
        data_dir / "bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet"
    )
    
    # Load CO2 emissions complete perimeter
    co2_df = pd.read_parquet(
        data_dir / "emission-co2-perimetre-complet.parquet"
    )
    
    return emissions_df, co2_df


def detect_date_column(df):
    """Detect date/datetime columns in the dataframe."""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_cols.append(col)
        elif 'date' in col.lower() or 'annee' in col.lower() or 'year' in col.lower() or 'annee' in col.lower():
            date_cols.append(col)
    return date_cols[0] if date_cols else None


def detect_numeric_columns(df, keywords=None):
    """Detect numeric columns that might represent emissions or energy."""
    if keywords is None:
        keywords = ['emission', 'co2', 'gaz', 'tonne', 'energie', 'energy', 'kwh', 'mwh', 'gwh', 'twh', 'consommation']
    
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


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">🚂 SNCF Emissions & Energy Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Granular Analysis: From Macro to Micro View")
    
    # Load data
    with st.spinner("Loading data..."):
        emissions_df, co2_df = load_data()
    
    # Sidebar filters
    st.sidebar.header("📋 Data Overview")
    st.sidebar.write(f"**Emissions Data:** {emissions_df.shape[0]} rows, {emissions_df.shape[1]} columns")
    st.sidebar.write(f"**CO2 Data:** {co2_df.shape[0]} rows, {co2_df.shape[1]} columns")
    
    # Show data preview
    with st.sidebar.expander("View Data Columns"):
        st.write("**Emissions columns:**", list(emissions_df.columns))
        st.write("**CO2 columns:**", list(co2_df.columns))
    
    # Analysis level selector
    analysis_level = st.sidebar.radio(
        "Analysis Level",
        ["Macro Overview", "Sector Analysis", "Micro Analysis", "Comparison"],
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
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** SNCF Greenhouse Gas Emissions and CO2 Data")


if __name__ == "__main__":
    main()
