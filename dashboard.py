"""
SNCF & RTE Data Visualization - Streamlit App
Interactive visualizations for SNCF greenhouse gas emissions, CO2 data, and RTE energy data
with granular analysis from macro to micro levels.

This version keeps your structure but makes the visualizations robust + meaningful even when
SNCF data is mostly "annual report" style (no real time series), and when columns vary.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SNCF & RTE Energy Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Robust helpers
# -----------------------------
def safe_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a series to datetime robustly."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def pick_best_numeric(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Pick the most meaningful numeric column:
    - prefer columns with highest non-null count
    - then highest absolute sum (avoids picking id-like columns)
    """
    if df is None or df.empty or not candidates:
        return None

    best = None
    best_score = (-1, -1.0)  # (non_null_count, abs_sum)

    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        non_null = int(s.notna().sum())
        abs_sum = float(s.fillna(0).abs().sum())
        score = (non_null, abs_sum)
        if score > best_score:
            best_score = score
            best = c

    return best


def pick_best_category(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Prefer a column that:
    - is object/category
    - has medium cardinality (2..50-ish)
    - has few missing values
    """
    if df is None or df.empty:
        return None

    best = None
    best_score = (-1, -1)  # (usable_count, -cardinality_penalty)

    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c]
        if not (s.dtype == "object" or str(s.dtype).startswith("category")):
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique < 2:
            continue
        # penalize very high cardinality
        penalty = abs(nunique - 12)  # sweet spot around ~12
        usable = int(s.notna().sum())
        score = (usable, -penalty)
        if score > best_score:
            best_score = score
            best = c

    return best


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    """Load SNCF parquet data files and RTE preprocessed parquet files (if present)."""
    data_dir = Path(__file__).parent / "data"

    # SNCF
    sncf_dir = data_dir / "sncf"
    emissions_df = pd.read_parquet(sncf_dir / "bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet")
    co2_df = pd.read_parquet(sncf_dir / "emission-co2-perimetre-complet.parquet")

    # RTE (processed)
    rte_dir = data_dir / "rte"
    consumption_file = rte_dir / "conso_mix_RTE_2023_processed.parquet"
    production_file = rte_dir / "RealisationDonneesProduction_2023_processed.parquet"

    rte_consumption_df = pd.read_parquet(consumption_file) if consumption_file.exists() else None
    rte_production_df = pd.read_parquet(production_file) if production_file.exists() else None

    return emissions_df, co2_df, rte_consumption_df, rte_production_df


# -----------------------------
# Column detection
# -----------------------------
def detect_date_column(df: pd.DataFrame) -> str | None:
    """Detect date/datetime columns (robust)."""
    if df is None or df.empty:
        return None

    # 1) true datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 2) by name heuristic
    name_candidates = []
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["date", "datetime", "time", "heure", "jour"]):
            name_candidates.append(col)

    # 3) try parsing candidates
    for col in name_candidates:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.6:
            return col

    return None


def detect_year_column(df: pd.DataFrame) -> str | None:
    """Detect a year/annual column for annual report-style data."""
    if df is None or df.empty:
        return None

    # Name-based first
    for col in df.columns:
        cl = col.lower()
        if cl in {"annee", "année", "year"} or "annee" in cl or "année" in cl or "year" in cl:
            # check numeric-ish years
            s = df[col]
            try:
                vals = pd.to_numeric(s, errors="coerce")
                if vals.notna().mean() > 0.6:
                    # sanity: years likely between 1900 and 2100
                    ok = vals.dropna().between(1900, 2100).mean() > 0.5
                    if ok:
                        return col
            except Exception:
                pass

    # Look for numeric column that looks like years
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        ok = vals.between(1900, 2100).mean() > 0.8
        if ok and vals.nunique() <= 30:
            return col

    return None


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Detect time/hour columns."""
    if df is None or df.empty:
        return None
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["heure", "hour", "time", "horaire"]):
            return col
    return None


def detect_numeric_columns(df: pd.DataFrame, keywords=None) -> list[str]:
    """Detect numeric columns likely representing emissions or energy."""
    if df is None or df.empty:
        return []
    if keywords is None:
        keywords = [
            "emission",
            "co2",
            "ges",
            "gaz",
            "tonne",
            "tco2",
            "energie",
            "energy",
            "kwh",
            "mwh",
            "gwh",
            "twh",
            "consommation",
            "production",
            "puissance",
            "power",
            "mw",
            "gw",
            "valeur",
            "value",
            "total",
        ]
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cl = col.lower()
            if any(k in cl for k in keywords):
                numeric_cols.append(col)

    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Detect categorical columns that might represent sectors or categories."""
    if df is None or df.empty:
        return []

    categorical_keywords = [
        "secteur",
        "sector",
        "categorie",
        "category",
        "type",
        "activite",
        "activité",
        "activity",
        "perimetre",
        "périmètre",
        "scope",
        "source",
        "origine",
        "origin",
        "poste",
        "famille",
        "libell",
        "label",
    ]

    cat_cols = []
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in categorical_keywords):
            if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
                cat_cols.append(col)

    # also include low-cardinality object cols
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique(dropna=True) < 50:
            if col not in cat_cols:
                cat_cols.append(col)

    return cat_cols


def prepare_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Prepare time series data."""
    df_ts = df.copy()
    if date_col and date_col in df_ts.columns:
        df_ts[date_col] = safe_datetime_series(df_ts[date_col])
        df_ts = df_ts.dropna(subset=[date_col, value_col])
        df_ts = df_ts.sort_values(date_col)
    return df_ts


# -----------------------------
# SNCF Views
# -----------------------------
def create_macro_overview(emissions_df: pd.DataFrame, co2_df: pd.DataFrame):
    """Macro overview, robust for annual-report style SNCF data."""
    st.header("📊 Macro Overview - Total Emissions & Energy")

    emissions_numeric = detect_numeric_columns(emissions_df)
    co2_numeric = detect_numeric_columns(co2_df)

    emissions_val = pick_best_numeric(emissions_df, emissions_numeric)
    co2_val = pick_best_numeric(co2_df, co2_numeric)

    col1, col2, col3, col4 = st.columns(4)

    total_emissions = emissions_df[emissions_val].sum() if emissions_val else np.nan
    total_co2 = co2_df[co2_val].sum() if co2_val else np.nan
    avg_emissions = emissions_df[emissions_val].mean() if emissions_val else np.nan
    avg_co2 = co2_df[co2_val].mean() if co2_val else np.nan

    with col1:
        st.metric("Total GHG Emissions", f"{total_emissions:,.0f}" if pd.notna(total_emissions) else "N/A")
    with col2:
        st.metric("Total CO2 Emissions", f"{total_co2:,.0f}" if pd.notna(total_co2) else "N/A")
    with col3:
        st.metric("Avg GHG Emissions", f"{avg_emissions:,.0f}" if pd.notna(avg_emissions) else "N/A")
    with col4:
        st.metric("Avg CO2 Emissions", f"{avg_co2:,.0f}" if pd.notna(avg_co2) else "N/A")

    # Try time series if there is date; else try annual bar by year; else show top categories
    date_col_ghg = detect_date_column(emissions_df)
    date_col_co2 = detect_date_column(co2_df)

    year_col_ghg = detect_year_column(emissions_df)
    year_col_co2 = detect_year_column(co2_df)

    if date_col_ghg and emissions_val:
        st.subheader("Emissions Over Time (if time granularity exists)")
        fig = go.Figure()
        df_ts = prepare_time_series(emissions_df, date_col_ghg, emissions_val)
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_col_ghg],
                y=df_ts[emissions_val],
                mode="lines+markers",
                name="GHG Emissions",
                line=dict(width=3),
            )
        )

        if date_col_co2 and co2_val:
            df_ts2 = prepare_time_series(co2_df, date_col_co2, co2_val)
            fig.add_trace(
                go.Scatter(
                    x=df_ts2[date_col_co2],
                    y=df_ts2[co2_val],
                    mode="lines+markers",
                    name="CO2 Emissions",
                    yaxis="y2",
                )
            )
            fig.update_layout(yaxis2=dict(title="CO2 Emissions", overlaying="y", side="right"))

        fig.update_layout(
            title="Total Emissions Over Time (Macro View)",
            xaxis_title="Date",
            yaxis_title="GHG Emissions",
            height=420,
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif year_col_ghg and emissions_val:
        st.subheader("Annual Emissions (detected year column)")
        df_y = emissions_df.copy()
        df_y[year_col_ghg] = pd.to_numeric(df_y[year_col_ghg], errors="coerce")
        annual = df_y.groupby(year_col_ghg)[emissions_val].sum().reset_index().dropna()
        annual = annual.sort_values(year_col_ghg)

        fig = px.bar(
            annual,
            x=year_col_ghg,
            y=emissions_val,
            title="GHG Emissions by Year",
            labels={year_col_ghg: "Year", emissions_val: "GHG Emissions"},
        )
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No date/year column detected for SNCF macro time series. Showing top categories instead.")
        cat_cols = detect_categorical_columns(emissions_df)
        cat_col = pick_best_category(emissions_df, cat_cols)
        if cat_col and emissions_val:
            top = (
                emissions_df.groupby(cat_col)[emissions_val]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fig = px.bar(top, x=cat_col, y=emissions_val, title=f"Top categories by {emissions_val}")
            fig.update_layout(xaxis_tickangle=-45, height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


def create_sector_analysis(df: pd.DataFrame, df_name="Emissions"):
    """Sector-level analysis (meso). Returns sector_col, sector_summary, value_col."""
    st.header(f"🏭 Sector Analysis - {df_name}")

    categorical_cols = detect_categorical_columns(df)
    numeric_cols = detect_numeric_columns(df)

    if not categorical_cols or not numeric_cols:
        st.info(f"No categorical or numeric columns found for sector analysis in {df_name} data.")
        return None, None, None

    # Prefer a good default (but keep your selectboxes)
    default_sector = pick_best_category(df, categorical_cols) or categorical_cols[0]
    default_value = pick_best_numeric(df, numeric_cols) or numeric_cols[0]

    sector_col = st.selectbox(
        f"Select sector/category column for {df_name}:",
        categorical_cols,
        index=categorical_cols.index(default_sector) if default_sector in categorical_cols else 0,
        key=f"sector_col_{df_name}",
    )

    value_col = st.selectbox(
        f"Select value column for {df_name}:",
        numeric_cols,
        index=numeric_cols.index(default_value) if default_value in numeric_cols else 0,
        key=f"value_col_{df_name}",
    )

    sector_summary = df.groupby(sector_col)[value_col].agg(["sum", "mean", "count"]).reset_index()
    sector_summary = sector_summary.sort_values("sum", ascending=False)
    sector_summary.columns = [sector_col, "Total", "Average", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(f"Top N sectors to display ({df_name}):", 5, 30, 15, key=f"top_n_{df_name}")
        top_sectors = sector_summary.head(top_n)

        fig = px.bar(
            top_sectors,
            x=sector_col,
            y="Total",
            title=f"Top {top_n} Sectors by Total {value_col}",
            labels={sector_col: "Sector", "Total": f"Total {value_col}"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=500, template="plotly_white")
        fig.update_xaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            top_sectors,
            values="Total",
            names=sector_col,
            title=f"Sector Distribution (Top {top_n})",
            hole=0.4,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    # Time series by sector (only if date exists)
    date_col = detect_date_column(df)
    year_col = detect_year_column(df)

    if date_col:
        st.subheader(f"Time Series by Sector - {df_name}")
        selected_sectors = st.multiselect(
            f"Select sectors to compare ({df_name}):",
            sorted(df[sector_col].dropna().unique().tolist()),
            default=list(df[sector_col].value_counts().head(5).index),
            key=f"selected_sectors_{df_name}",
        )

        if selected_sectors:
            fig3 = go.Figure()
            df_ts = df[df[sector_col].isin(selected_sectors)].copy()
            df_ts[date_col] = safe_datetime_series(df_ts[date_col])
            df_ts = df_ts.dropna(subset=[date_col, value_col]).sort_values(date_col)

            for sector in selected_sectors:
                sector_data = df_ts[df_ts[sector_col] == sector]
                sector_agg = sector_data.groupby(date_col)[value_col].sum().reset_index()
                fig3.add_trace(
                    go.Scatter(
                        x=sector_agg[date_col],
                        y=sector_agg[value_col],
                        mode="lines+markers",
                        name=str(sector),
                        line=dict(width=2),
                    )
                )

            fig3.update_layout(
                title=f"{value_col} Over Time by Sector",
                xaxis_title="Date",
                yaxis_title=value_col,
                height=500,
                hovermode="x unified",
                template="plotly_white",
            )
            st.plotly_chart(fig3, use_container_width=True)

    elif year_col:
        st.subheader(f"Annual Comparison by Sector - {df_name}")
        selected_sectors = st.multiselect(
            f"Select sectors to compare ({df_name}):",
            sorted(df[sector_col].dropna().unique().tolist()),
            default=list(df[sector_col].value_counts().head(5).index),
            key=f"selected_sectors_year_{df_name}",
        )
        if selected_sectors:
            df_y = df[df[sector_col].isin(selected_sectors)].copy()
            df_y[year_col] = pd.to_numeric(df_y[year_col], errors="coerce")
            df_y = df_y.dropna(subset=[year_col, value_col])

            figy = px.line(
                df_y.groupby([year_col, sector_col])[value_col].sum().reset_index(),
                x=year_col,
                y=value_col,
                color=sector_col,
                markers=True,
                title=f"{value_col} by Year and Sector",
            )
            figy.update_layout(height=500, template="plotly_white")
            st.plotly_chart(figy, use_container_width=True)

    return sector_col, sector_summary, value_col


def create_micro_analysis(df: pd.DataFrame, sector_col: str, sector_summary: pd.DataFrame, value_col: str, df_name="Emissions"):
    """Micro-level detailed analysis (uses the SAME value_col selected in sector analysis)."""
    st.header(f"🔬 Micro Analysis - Detailed Breakdown ({df_name})")

    if sector_col is None or sector_summary is None or value_col is None:
        st.info("Please complete sector analysis first.")
        return

    selected_sector = st.selectbox(
        f"Select sector for detailed analysis ({df_name}):",
        sector_summary[sector_col].tolist(),
        key=f"selected_sector_{df_name}",
    )

    sector_data = df[df[sector_col] == selected_sector].copy()

    all_categorical = detect_categorical_columns(sector_data)
    other_categorical = [col for col in all_categorical if col != sector_col]

    if other_categorical and value_col in sector_data.columns:
        subcategory_col = st.selectbox(
            f"Select sub-category for {selected_sector}:",
            other_categorical,
            key=f"subcategory_{df_name}",
        )

        subcategory_summary = (
            sector_data.groupby(subcategory_col)[value_col].agg(["sum", "mean", "count"]).reset_index()
        )
        subcategory_summary = subcategory_summary.sort_values("sum", ascending=False)
        subcategory_summary.columns = [subcategory_col, "Total", "Average", "Count"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                subcategory_summary.head(15),
                x=subcategory_col,
                y="Total",
                title=f"Breakdown of {selected_sector} by {subcategory_col}",
                labels={subcategory_col: subcategory_col, "Total": f"Total {value_col}"},
            )
            fig.update_layout(xaxis_tickangle=-45, height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"Statistics for {selected_sector}")
            st.dataframe(subcategory_summary, use_container_width=True)

    else:
        st.info("No additional sub-category columns detected for drill-down.")

    with st.expander(f"View raw data for {selected_sector}"):
        st.dataframe(sector_data, use_container_width=True)


def create_comparison_view(emissions_df: pd.DataFrame, co2_df: pd.DataFrame):
    """Comparison view between emissions and CO2 (robust; chooses sensible category/value columns)."""
    st.header("⚖️ Comparative Analysis")

    emissions_numeric = detect_numeric_columns(emissions_df)
    co2_numeric = detect_numeric_columns(co2_df)
    emissions_cat = detect_categorical_columns(emissions_df)
    co2_cat = detect_categorical_columns(co2_df)

    ghg_val = pick_best_numeric(emissions_df, emissions_numeric)
    co2_val = pick_best_numeric(co2_df, co2_numeric)

    if not ghg_val or not co2_val:
        st.info("Insufficient numeric columns detected for comparison.")
        return

    # Find common category columns; otherwise fall back to the "best" in either dataset
    common_cats = list(set(emissions_cat) & set(co2_cat))
    if common_cats:
        default_cat = pick_best_category(emissions_df, common_cats) or common_cats[0]
        comparison_col = st.selectbox("Select column for comparison:", common_cats, index=common_cats.index(default_cat))
    else:
        fallback = pick_best_category(emissions_df, emissions_cat) or pick_best_category(co2_df, co2_cat)
        if fallback is None:
            st.info("No shared or usable categorical columns found for comparison.")
            return
        comparison_col = st.selectbox("Select column for comparison:", [fallback])

    # Aggregate
    emissions_agg = emissions_df.groupby(comparison_col)[ghg_val].sum().reset_index()
    emissions_agg.columns = [comparison_col, "GHG Emissions"]

    co2_agg = co2_df.groupby(comparison_col)[co2_val].sum().reset_index()
    co2_agg.columns = [comparison_col, "CO2 Emissions"]

    comparison_df = pd.merge(emissions_agg, co2_agg, on=comparison_col, how="outer").fillna(0)
    comparison_df = comparison_df.sort_values("GHG Emissions", ascending=False).head(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="GHG Emissions", x=comparison_df[comparison_col], y=comparison_df["GHG Emissions"]))
    fig.add_trace(go.Bar(name="CO2 Emissions", x=comparison_df[comparison_col], y=comparison_df["CO2 Emissions"]))

    fig.update_layout(
        title=f"GHG vs CO2 Emissions by {comparison_col}",
        xaxis_title=comparison_col,
        yaxis_title="Emissions",
        barmode="group",
        height=520,
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(comparison_df) > 1:
        corr = comparison_df["GHG Emissions"].corr(comparison_df["CO2 Emissions"])
        st.metric("Correlation between GHG and CO2 (top 15 categories)", f"{corr:.3f}")


# -----------------------------
# RTE Views
# -----------------------------
def create_peak_analysis(df: pd.DataFrame, date_col: str, value_col: str):
    """Analyze peak and down timings in energy consumption/production."""
    st.header("⏰ Peak & Down Timing Analysis")

    if df is None or date_col is None or value_col is None:
        st.info("No data available for peak analysis.")
        return

    df_clean = df.copy()
    df_clean[date_col] = safe_datetime_series(df_clean[date_col])
    df_clean = df_clean.dropna(subset=[date_col, value_col]).sort_values(date_col)

    # Extract time components
    df_clean["hour"] = df_clean[date_col].dt.hour
    df_clean["day_of_week"] = df_clean[date_col].dt.day_name()
    df_clean["month"] = df_clean[date_col].dt.month

    col1, col2, col3 = st.columns(3)

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
    hourly_avg = df_clean.groupby("hour")[value_col].agg(["mean", "max", "min"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_avg["hour"], y=hourly_avg["mean"], mode="lines+markers", name="Average"))
    fig.add_trace(go.Scatter(x=hourly_avg["hour"], y=hourly_avg["max"], mode="lines", name="Maximum", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=hourly_avg["hour"], y=hourly_avg["min"], mode="lines", name="Minimum", line=dict(dash="dash")))

    peak_hour = hourly_avg.loc[hourly_avg["mean"].idxmax(), "hour"]
    fig.add_vline(x=peak_hour, line_dash="dot", annotation_text=f"Peak Hour: {int(peak_hour)}h")

    fig.update_layout(
        title="Average Consumption/Production by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title=value_col,
        height=420,
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Day of week
    st.subheader("Day of Week Pattern")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_avg = df_clean.groupby("day_of_week")[value_col].mean().reindex(day_order).reset_index()

    fig2 = px.bar(daily_avg, x="day_of_week", y=value_col, title="Average by Day of Week")
    fig2.update_layout(height=420, template="plotly_white", xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly
    st.subheader("Monthly Pattern")
    monthly_avg = df_clean.groupby("month")[value_col].agg(["mean", "max", "min"]).reset_index()
    monthly_avg["month_name"] = pd.to_datetime(monthly_avg["month"], format="%m").dt.strftime("%B")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=monthly_avg["month_name"], y=monthly_avg["mean"], name="Average"))
    fig3.add_trace(go.Scatter(x=monthly_avg["month_name"], y=monthly_avg["max"], mode="markers", name="Peak", marker=dict(symbol="triangle-up", size=10)))
    fig3.add_trace(go.Scatter(x=monthly_avg["month_name"], y=monthly_avg["min"], mode="markers", name="Minimum", marker=dict(symbol="triangle-down", size=10)))
    fig3.update_layout(
        title="Monthly Average with Peaks and Minimums",
        xaxis_title="Month",
        yaxis_title=value_col,
        height=420,
        xaxis_tickangle=-45,
        template="plotly_white",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Time series with peaks
    st.subheader("Time Series with Peak Identification")
    df_clean["is_peak"] = df_clean[value_col] >= df_clean[value_col].quantile(0.95)
    df_clean["is_low"] = df_clean[value_col] <= df_clean[value_col].quantile(0.05)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_clean[date_col], y=df_clean[value_col], mode="lines", name="Normal"))
    fig4.add_trace(go.Scatter(x=df_clean[df_clean["is_peak"]][date_col], y=df_clean[df_clean["is_peak"]][value_col], mode="markers", name="Peak (Top 5%)"))
    fig4.add_trace(go.Scatter(x=df_clean[df_clean["is_low"]][date_col], y=df_clean[df_clean["is_low"]][value_col], mode="markers", name="Low (Bottom 5%)"))
    fig4.update_layout(
        title="Time Series with Peak and Low Periods Highlighted",
        xaxis_title="Date",
        yaxis_title=value_col,
        height=520,
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig4, use_container_width=True)


def create_stacked_production_chart(rte_production_df: pd.DataFrame):
    """Stacked area chart: production by sector over time."""
    if rte_production_df is None or rte_production_df.empty:
        return None

    date_col = detect_date_column(rte_production_df)
    if date_col is None:
        return None

    exclude = {c.lower() for c in ["date", "heures", "total", "datetime", "hour"]}
    production_sectors = [
        col for col in rte_production_df.columns
        if col.lower() not in exclude and pd.api.types.is_numeric_dtype(rte_production_df[col])
    ]
    if not production_sectors:
        return None

    df = rte_production_df.copy()
    df[date_col] = safe_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Build datetime with Heures if exists
    if "Heures" in df.columns:
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            s = str(hour_str)
            if "-" in s:
                return int(s.split("-")[0].split(":")[0])
            if ":" in s:
                return int(s.split(":")[0])
            return 0

        df["hour"] = df["Heures"].apply(extract_hour)
        df["datetime"] = df[date_col] + pd.to_timedelta(df["hour"], unit="h")
    else:
        df["datetime"] = df[date_col]

    df_agg = df.groupby("datetime")[production_sectors].sum().reset_index().sort_values("datetime")

    # Sort sectors by contribution
    sorted_sectors = df_agg[production_sectors].sum().sort_values(ascending=False).index.tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

    for idx, sector in enumerate(sorted_sectors):
        fig.add_trace(
            go.Scatter(
                x=df_agg["datetime"],
                y=df_agg[sector].fillna(0),
                mode="lines",
                name=sector,
                stackgroup="one",
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5),
            )
        )

    fig.update_layout(
        title="Production Over Time by Sector (Stacked)",
        xaxis_title="Date",
        yaxis_title="Production (MW)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


def create_stacked_consumption_chart(rte_consumption_df: pd.DataFrame):
    """Stacked area chart: consumption components over time (or consumption total if only one)."""
    if rte_consumption_df is None or rte_consumption_df.empty:
        return None

    date_col = detect_date_column(rte_consumption_df)
    if date_col is None:
        return None

    exclude = {c.lower() for c in ["date", "heures", "datetime", "hour"]}
    consumption_cols = [
        col for col in rte_consumption_df.columns
        if col.lower() not in exclude and pd.api.types.is_numeric_dtype(rte_consumption_df[col])
    ]
    if not consumption_cols:
        return None

    df = rte_consumption_df.copy()
    df[date_col] = safe_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if "Heures" in df.columns:
        def extract_hour(hour_str):
            if pd.isna(hour_str):
                return 0
            s = str(hour_str)
            if ":" in s:
                return int(s.split(":")[0])
            return 0

        df["hour"] = df["Heures"].apply(extract_hour)
        df["datetime"] = df[date_col] + pd.to_timedelta(df["hour"], unit="h")
    else:
        df["datetime"] = df[date_col]

    df_agg = df.groupby("datetime")[consumption_cols].sum().reset_index().sort_values("datetime")

    sorted_cols = df_agg[consumption_cols].sum().sort_values(ascending=False).index.tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for idx, col in enumerate(sorted_cols):
        fig.add_trace(
            go.Scatter(
                x=df_agg["datetime"],
                y=df_agg[col].fillna(0),
                mode="lines",
                name=col,
                stackgroup="one",
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5),
            )
        )

    fig.update_layout(
        title="Consumption Over Time (Stacked)",
        xaxis_title="Date",
        yaxis_title="Consumption (MW)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


def create_production_consumption_balance(rte_production_df: pd.DataFrame, rte_consumption_df: pd.DataFrame):
    """Production vs consumption balance over time."""
    if rte_production_df is None or rte_consumption_df is None:
        return None

    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    if date_col_prod is None or date_col_cons is None:
        return None

    df_prod = rte_production_df.copy()
    df_cons = rte_consumption_df.copy()
    df_prod[date_col_prod] = safe_datetime_series(df_prod[date_col_prod])
    df_cons[date_col_cons] = safe_datetime_series(df_cons[date_col_cons])

    # datetime
    if "Heures" in df_prod.columns:
        def extract_hour_prod(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if "-" in s:
                return int(s.split("-")[0].split(":")[0])
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_prod["hour"] = df_prod["Heures"].apply(extract_hour_prod)
        df_prod["datetime"] = df_prod[date_col_prod] + pd.to_timedelta(df_prod["hour"], unit="h")
    else:
        df_prod["datetime"] = df_prod[date_col_prod]

    if "Heures" in df_cons.columns:
        def extract_hour_cons(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_cons["hour"] = df_cons["Heures"].apply(extract_hour_cons)
        df_cons["datetime"] = df_cons[date_col_cons] + pd.to_timedelta(df_cons["hour"], unit="h")
    else:
        df_cons["datetime"] = df_cons[date_col_cons]

    # totals
    prod_total_col = "Total" if "Total" in df_prod.columns else None
    if prod_total_col is None:
        exclude = {c.lower() for c in ["date", "heures", "datetime", "hour", "total"]}
        prod_cols = [c for c in df_prod.columns if c.lower() not in exclude and pd.api.types.is_numeric_dtype(df_prod[c])]
        if not prod_cols:
            return None
        df_prod["Total_Production"] = df_prod[prod_cols].sum(axis=1)
        prod_total_col = "Total_Production"

    # consumption column
    cons_candidates = []
    for c in df_cons.columns:
        cl = c.lower()
        if "consomm" in cl or "consumption" in cl:
            cons_candidates.append(c)
    cons_candidates += detect_numeric_columns(df_cons, keywords=["consomm", "mw", "power", "value", "total"])
    cons_col = pick_best_numeric(df_cons, list(dict.fromkeys(cons_candidates)))
    if cons_col is None:
        return None

    prod_agg = df_prod.groupby("datetime")[prod_total_col].sum().reset_index()
    cons_agg = df_cons.groupby("datetime")[cons_col].sum().reset_index()

    balance_df = pd.merge(prod_agg, cons_agg, on="datetime", how="outer").sort_values("datetime")
    balance_df["Balance"] = balance_df[prod_total_col] - balance_df[cons_col]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Production vs Consumption Over Time", "Balance (Production - Consumption)"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(go.Scatter(x=balance_df["datetime"], y=balance_df[prod_total_col], mode="lines", name="Total Production", fill="tozeroy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=balance_df["datetime"], y=balance_df[cons_col], mode="lines", name="Consumption", fill="tozeroy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=balance_df["datetime"], y=balance_df["Balance"], mode="lines", name="Balance", fill="tozeroy"), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Balance (MW)", row=2, col=1)

    fig.update_layout(height=800, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def create_hourly_sector_comparison(rte_production_df: pd.DataFrame, rte_consumption_df: pd.DataFrame):
    """Hourly pattern: average production by sector vs average consumption."""
    if rte_production_df is None or rte_consumption_df is None:
        return None

    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    if date_col_prod is None or date_col_cons is None:
        return None

    df_prod = rte_production_df.copy()
    df_cons = rte_consumption_df.copy()
    df_prod[date_col_prod] = safe_datetime_series(df_prod[date_col_prod])
    df_cons[date_col_cons] = safe_datetime_series(df_cons[date_col_cons])

    # hours
    if "Heures" in df_prod.columns:
        def extract_hour_prod(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if "-" in s:
                return int(s.split("-")[0].split(":")[0])
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_prod["hour"] = df_prod["Heures"].apply(extract_hour_prod)
    else:
        df_prod["hour"] = df_prod[date_col_prod].dt.hour

    if "Heures" in df_cons.columns:
        def extract_hour_cons(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_cons["hour"] = df_cons["Heures"].apply(extract_hour_cons)
    else:
        df_cons["hour"] = df_cons[date_col_cons].dt.hour

    # production sectors
    exclude = {c.lower() for c in ["date", "heures", "total", "hour"]}
    production_sectors = [
        col for col in df_prod.columns
        if col.lower() not in exclude and pd.api.types.is_numeric_dtype(df_prod[col])
    ]
    if not production_sectors:
        return None

    # consumption column
    cons_candidates = []
    for c in df_cons.columns:
        if "consomm" in c.lower() or "consumption" in c.lower():
            cons_candidates.append(c)
    cons_candidates += detect_numeric_columns(df_cons, keywords=["consomm", "mw", "power", "value", "total"])
    cons_col = pick_best_numeric(df_cons, list(dict.fromkeys(cons_candidates)))
    if cons_col is None:
        return None

    prod_hourly = df_prod.groupby("hour")[production_sectors].mean().reset_index()
    cons_hourly = df_cons.groupby("hour")[cons_col].mean().reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Average Production by Hour (Top Sectors)", "Average Consumption by Hour"),
    )

    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    sector_totals = prod_hourly[production_sectors].sum().sort_values(ascending=False)
    sorted_sectors = sector_totals.index.tolist()[:10]

    for idx, sector in enumerate(sorted_sectors):
        fig.add_trace(
            go.Scatter(
                x=prod_hourly["hour"],
                y=prod_hourly[sector].fillna(0),
                mode="lines",
                name=sector,
                stackgroup="one",
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=cons_hourly["hour"],
            y=cons_hourly[cons_col].fillna(0),
            mode="lines+markers",
            name="Consumption",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_yaxes(title_text="Production (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Consumption (MW)", row=1, col=2)

    fig.update_layout(height=600, hovermode="x unified", template="plotly_white", legend=dict(orientation="v", y=1, x=1.02))
    return fig


def create_weekly_production_chart(rte_production_df: pd.DataFrame, week_type="average", selected_week=None):
    """Weekly (hour-of-week) stacked production chart."""
    if rte_production_df is None or rte_production_df.empty:
        return None
    date_col = detect_date_column(rte_production_df)
    if date_col is None:
        return None

    exclude = {c.lower() for c in ["date", "heures", "total"]}
    production_sectors = [
        col for col in rte_production_df.columns
        if col.lower() not in exclude and pd.api.types.is_numeric_dtype(rte_production_df[col])
    ]
    if not production_sectors:
        return None

    df = rte_production_df.copy()
    df[date_col] = safe_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if "Heures" in df.columns:
        def extract_hour(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if "-" in s:
                return int(s.split("-")[0].split(":")[0])
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df["hour"] = df["Heures"].apply(extract_hour)
    else:
        df["hour"] = df[date_col].dt.hour

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["year"] = df[date_col].dt.year.astype(int)
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]

    if week_type == "average":
        df_agg = df.groupby("hour_of_week")[production_sectors].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        if selected_week is None:
            selected_week = int(df["week"].iloc[0])
        selected_year = int(df[df["week"] == selected_week]["year"].iloc[0]) if (df["week"] == selected_week).any() else int(df["year"].iloc[0])
        df_week = df[(df["week"] == selected_week) & (df["year"] == selected_year)]
        df_agg = df_week.groupby("hour_of_week")[production_sectors].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"

    df_agg["day_of_week"] = df_agg["hour_of_week"] // 24
    df_agg["hour"] = df_agg["hour_of_week"] % 24
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_agg["label"] = df_agg.apply(lambda r: f"{day_names[int(r['day_of_week'])]} {int(r['hour']):02d}:00", axis=1)

    fig = go.Figure()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    sorted_sectors = df_agg[production_sectors].sum().sort_values(ascending=False).index.tolist()

    for idx, sector in enumerate(sorted_sectors):
        fig.add_trace(
            go.Scatter(
                x=df_agg["hour_of_week"],
                y=df_agg[sector].fillna(0),
                mode="lines",
                name=sector,
                stackgroup="one",
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5),
                customdata=df_agg["label"],
                hovertemplate=f"<b>{sector}</b><br>%{{customdata}}<br>Value: %{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=24,
        ticktext=day_names,
        tickvals=list(range(0, 168, 24)),
    )
    fig.update_layout(
        title=f"Production by Sector per Hour{title_suffix}",
        xaxis_title="Day of Week",
        yaxis_title="Production (MW)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", y=1, x=1.02),
    )
    return fig


def create_weekly_consumption_chart(rte_consumption_df: pd.DataFrame, week_type="average", selected_week=None):
    """Weekly (hour-of-week) consumption chart."""
    if rte_consumption_df is None or rte_consumption_df.empty:
        return None
    date_col = detect_date_column(rte_consumption_df)
    if date_col is None:
        return None

    exclude = {c.lower() for c in ["date", "heures"]}
    consumption_cols = [
        col for col in rte_consumption_df.columns
        if col.lower() not in exclude and pd.api.types.is_numeric_dtype(rte_consumption_df[col])
    ]
    if not consumption_cols:
        return None

    df = rte_consumption_df.copy()
    df[date_col] = safe_datetime_series(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if "Heures" in df.columns:
        def extract_hour(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df["hour"] = df["Heures"].apply(extract_hour)
    else:
        df["hour"] = df[date_col].dt.hour

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["year"] = df[date_col].dt.year.astype(int)
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]

    if week_type == "average":
        df_agg = df.groupby("hour_of_week")[consumption_cols].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        if selected_week is None:
            selected_week = int(df["week"].iloc[0])
        selected_year = int(df[df["week"] == selected_week]["year"].iloc[0]) if (df["week"] == selected_week).any() else int(df["year"].iloc[0])
        df_week = df[(df["week"] == selected_week) & (df["year"] == selected_year)]
        df_agg = df_week.groupby("hour_of_week")[consumption_cols].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"

    df_agg["day_of_week"] = df_agg["hour_of_week"] // 24
    df_agg["hour"] = df_agg["hour_of_week"] % 24
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_agg["label"] = df_agg.apply(lambda r: f"{day_names[int(r['day_of_week'])]} {int(r['hour']):02d}:00", axis=1)

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    sorted_cols = df_agg[consumption_cols].sum().sort_values(ascending=False).index.tolist()

    for idx, col in enumerate(sorted_cols):
        fig.add_trace(
            go.Scatter(
                x=df_agg["hour_of_week"],
                y=df_agg[col].fillna(0),
                mode="lines",
                name=col,
                stackgroup="one",
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0.5),
                customdata=df_agg["label"],
                hovertemplate=f"<b>{col}</b><br>%{{customdata}}<br>Value: %{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=24,
        ticktext=day_names,
        tickvals=list(range(0, 168, 24)),
    )
    fig.update_layout(
        title=f"Consumption per Hour{title_suffix}",
        xaxis_title="Day of Week",
        yaxis_title="Consumption (MW)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", y=1, x=1.02),
    )
    return fig


def create_weekly_balance_chart(rte_production_df: pd.DataFrame, rte_consumption_df: pd.DataFrame, week_type="average", selected_week=None):
    """Weekly (hour-of-week) balance chart."""
    if rte_production_df is None or rte_consumption_df is None:
        return None

    date_col_prod = detect_date_column(rte_production_df)
    date_col_cons = detect_date_column(rte_consumption_df)
    if date_col_prod is None or date_col_cons is None:
        return None

    df_prod = rte_production_df.copy()
    df_cons = rte_consumption_df.copy()
    df_prod[date_col_prod] = safe_datetime_series(df_prod[date_col_prod])
    df_cons[date_col_cons] = safe_datetime_series(df_cons[date_col_cons])

    # hours
    if "Heures" in df_prod.columns:
        def extract_hour_prod(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if "-" in s:
                return int(s.split("-")[0].split(":")[0])
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_prod["hour"] = df_prod["Heures"].apply(extract_hour_prod)
    else:
        df_prod["hour"] = df_prod[date_col_prod].dt.hour

    if "Heures" in df_cons.columns:
        def extract_hour_cons(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if ":" in s:
                return int(s.split(":")[0])
            return 0
        df_cons["hour"] = df_cons["Heures"].apply(extract_hour_cons)
    else:
        df_cons["hour"] = df_cons[date_col_cons].dt.hour

    # hour of week
    df_prod["day_of_week"] = df_prod[date_col_prod].dt.dayofweek
    df_prod["week"] = df_prod[date_col_prod].dt.isocalendar().week.astype(int)
    df_prod["year"] = df_prod[date_col_prod].dt.year.astype(int)
    df_prod["hour_of_week"] = df_prod["day_of_week"] * 24 + df_prod["hour"]

    df_cons["day_of_week"] = df_cons[date_col_cons].dt.dayofweek
    df_cons["week"] = df_cons[date_col_cons].dt.isocalendar().week.astype(int)
    df_cons["year"] = df_cons[date_col_cons].dt.year.astype(int)
    df_cons["hour_of_week"] = df_cons["day_of_week"] * 24 + df_cons["hour"]

    # totals
    prod_total_col = "Total" if "Total" in df_prod.columns else None
    if prod_total_col is None:
        exclude = {c.lower() for c in ["date", "heures", "hour", "day_of_week", "week", "year", "hour_of_week"]}
        prod_cols = [c for c in df_prod.columns if c.lower() not in exclude and pd.api.types.is_numeric_dtype(df_prod[c])]
        if not prod_cols:
            return None
        df_prod["Total_Production"] = df_prod[prod_cols].sum(axis=1)
        prod_total_col = "Total_Production"

    # consumption column
    cons_candidates = []
    for c in df_cons.columns:
        if "consomm" in c.lower() or "consumption" in c.lower():
            cons_candidates.append(c)
    cons_candidates += detect_numeric_columns(df_cons, keywords=["consomm", "mw", "power", "value", "total"])
    cons_col = pick_best_numeric(df_cons, list(dict.fromkeys(cons_candidates)))
    if cons_col is None:
        return None

    if week_type == "average":
        prod_agg = df_prod.groupby("hour_of_week")[prod_total_col].mean().reset_index()
        cons_agg = df_cons.groupby("hour_of_week")[cons_col].mean().reset_index()
        title_suffix = " (Average Week)"
    else:
        if selected_week is None:
            selected_week = int(df_prod["week"].iloc[0])
        selected_year = int(df_prod[df_prod["week"] == selected_week]["year"].iloc[0]) if (df_prod["week"] == selected_week).any() else int(df_prod["year"].iloc[0])
        df_prod_week = df_prod[(df_prod["week"] == selected_week) & (df_prod["year"] == selected_year)]
        df_cons_week = df_cons[(df_cons["week"] == selected_week) & (df_cons["year"] == selected_year)]
        prod_agg = df_prod_week.groupby("hour_of_week")[prod_total_col].mean().reset_index()
        cons_agg = df_cons_week.groupby("hour_of_week")[cons_col].mean().reset_index()
        title_suffix = f" (Week {selected_week}, {selected_year})"

    balance_df = pd.merge(prod_agg, cons_agg, on="hour_of_week", how="outer").sort_values("hour_of_week")
    balance_df["Balance"] = balance_df[prod_total_col] - balance_df[cons_col]

    balance_df["day_of_week"] = balance_df["hour_of_week"] // 24
    balance_df["hour"] = balance_df["hour_of_week"] % 24
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    balance_df["label"] = balance_df.apply(lambda r: f"{day_names[int(r['day_of_week'])]} {int(r['hour']):02d}:00", axis=1)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Production vs Consumption per Hour{title_suffix}", "Balance (Production - Consumption)"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Scatter(
            x=balance_df["hour_of_week"], y=balance_df[prod_total_col],
            mode="lines", name="Total Production", fill="tozeroy",
            customdata=balance_df["label"],
            hovertemplate="<b>Production</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=balance_df["hour_of_week"], y=balance_df[cons_col],
            mode="lines", name="Consumption", fill="tozeroy",
            customdata=balance_df["label"],
            hovertemplate="<b>Consumption</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=balance_df["hour_of_week"], y=balance_df["Balance"],
            mode="lines", name="Balance", fill="tozeroy",
            customdata=balance_df["label"],
            hovertemplate="<b>Balance</b><br>%{customdata}<br>Value: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    day_names_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=24, ticktext=day_names_short, tickvals=list(range(0, 168, 24)), row=1, col=1)
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=24, ticktext=day_names_short, tickvals=list(range(0, 168, 24)), title_text="Day of Week", row=2, col=1)

    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Balance (MW)", row=2, col=1)

    fig.update_layout(height=800, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def create_micro_rte_analysis(rte_consumption_df: pd.DataFrame, rte_production_df: pd.DataFrame):
    """Micro RTE analysis: weekly patterns."""
    st.header("🔬 Micro RTE Analysis - Weekly Patterns per Hour")

    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE data available for micro analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        week_type = st.radio("Select week type:", ["Average Week", "Specific Week"], key="rte_week_type")

    selected_week = None
    if week_type == "Specific Week":
        if rte_production_df is not None:
            date_col = detect_date_column(rte_production_df)
            if date_col:
                df_temp = rte_production_df.copy()
                df_temp[date_col] = safe_datetime_series(df_temp[date_col])
                df_temp = df_temp.dropna(subset=[date_col])
                df_temp["week"] = df_temp[date_col].dt.isocalendar().week.astype(int)
                df_temp["year"] = df_temp[date_col].dt.year.astype(int)
                available_weeks = sorted(df_temp[["week", "year"]].drop_duplicates().values.tolist())

                with col2:
                    if available_weeks:
                        week_options = [f"Week {w}, {y}" for w, y in available_weeks]
                        selected_week_str = st.selectbox("Select week:", week_options, key="rte_selected_week")
                        selected_week = int(selected_week_str.split(",")[0].split()[1])

    week_type_param = "average" if week_type == "Average Week" else "specific"

    if rte_production_df is not None:
        st.subheader("📈 Production by Sector per Hour")
        fig_prod = create_weekly_production_chart(rte_production_df, week_type_param, selected_week)
        if fig_prod:
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Could not create weekly production chart.")

    if rte_consumption_df is not None:
        st.subheader("📊 Consumption per Hour")
        fig_cons = create_weekly_consumption_chart(rte_consumption_df, week_type_param, selected_week)
        if fig_cons:
            st.plotly_chart(fig_cons, use_container_width=True)
        else:
            st.info("Could not create weekly consumption chart.")

    if rte_production_df is not None and rte_consumption_df is not None:
        st.subheader("⚖️ Production vs Consumption Balance per Hour")
        fig_balance = create_weekly_balance_chart(rte_production_df, rte_consumption_df, week_type_param, selected_week)
        if fig_balance:
            st.plotly_chart(fig_balance, use_container_width=True)
        else:
            st.info("Could not create weekly balance chart.")


def create_rte_analysis(rte_consumption_df: pd.DataFrame, rte_production_df: pd.DataFrame):
    """Comprehensive RTE analysis (stacked + balance + hourly + peak analysis)."""
    st.header("⚡ RTE Energy Analysis")

    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE data available. Ensure processed parquet files exist in data/rte/")
        return

    if rte_production_df is not None:
        st.subheader("📈 Production Over Time by Sector (Stacked)")
        fig_prod = create_stacked_production_chart(rte_production_df)
        if fig_prod:
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Could not create production chart. Check data structure.")

    if rte_consumption_df is not None:
        st.subheader("📊 Consumption Over Time (Stacked)")
        fig_cons = create_stacked_consumption_chart(rte_consumption_df)
        if fig_cons:
            st.plotly_chart(fig_cons, use_container_width=True)
        else:
            st.info("Could not create consumption chart. Check data structure.")

    if rte_production_df is not None and rte_consumption_df is not None:
        st.subheader("⚖️ Production vs Consumption Balance Over Time")
        fig_balance = create_production_consumption_balance(rte_production_df, rte_consumption_df)
        if fig_balance:
            st.plotly_chart(fig_balance, use_container_width=True)
        else:
            st.info("Could not create balance chart. Check data structure.")

        st.subheader("🕐 Hourly Patterns: Production by Sector vs Consumption")
        fig_hourly = create_hourly_sector_comparison(rte_production_df, rte_consumption_df)
        if fig_hourly:
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("Could not create hourly comparison chart. Check data structure.")

    # Peak analysis: focus on consumption (most relevant to time-shifting story)
    st.subheader("⏰ Peak & Low Timing (Consumption)")
    if rte_consumption_df is not None:
        date_col = detect_date_column(rte_consumption_df)
        candidates = []
        for c in rte_consumption_df.columns:
            if "consomm" in c.lower() or "consumption" in c.lower():
                candidates.append(c)
        candidates += detect_numeric_columns(rte_consumption_df, keywords=["consomm", "mw", "power", "value", "total"])
        val_col = pick_best_numeric(rte_consumption_df, list(dict.fromkeys(candidates)))
        if date_col and val_col:
            create_peak_analysis(rte_consumption_df, date_col, val_col)
        else:
            st.info("Could not detect consumption date/value columns for peak analysis.")


# -----------------------------
# Main
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">⚡ SNCF & RTE Energy Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Granular Analysis: From Macro to Micro View | Energy Consumption & Production")

    with st.spinner("Loading data..."):
        emissions_df, co2_df, rte_consumption_df, rte_production_df = load_data()

    # Sidebar overview
    st.sidebar.header("📋 Data Overview")
    st.sidebar.write(f"**Emissions Data:** {emissions_df.shape[0]} rows, {emissions_df.shape[1]} columns")
    st.sidebar.write(f"**CO2 Data:** {co2_df.shape[0]} rows, {co2_df.shape[1]} columns")

    if rte_consumption_df is not None:
        st.sidebar.write(f"**RTE Consumption:** {rte_consumption_df.shape[0]} rows, {rte_consumption_df.shape[1]} columns")
    else:
        st.sidebar.warning("RTE consumption parquet not found. (Expected: data/rte/conso_mix_RTE_2023_processed.parquet)")

    if rte_production_df is not None:
        st.sidebar.write(f"**RTE Production:** {rte_production_df.shape[0]} rows, {rte_production_df.shape[1]} columns")
    else:
        st.sidebar.warning("RTE production parquet not found. (Expected: data/rte/RealisationDonneesProduction_2023_processed.parquet)")

    with st.sidebar.expander("View Data Columns"):
        st.write("**Emissions columns:**", list(emissions_df.columns))
        st.write("**CO2 columns:**", list(co2_df.columns))
        if rte_consumption_df is not None:
            st.write("**RTE Consumption columns:**", list(rte_consumption_df.columns))
        if rte_production_df is not None:
            st.write("**RTE Production columns:**", list(rte_production_df.columns))

    # Navigation
    analysis_options = ["Macro Overview", "Sector Analysis", "Micro Analysis", "Comparison"]
    if rte_consumption_df is not None or rte_production_df is not None:
        analysis_options += ["RTE Energy Analysis", "Micro RTE Analysis"]

    analysis_level = st.sidebar.radio("Analysis Level", analysis_options, index=0)

    if analysis_level == "Macro Overview":
        create_macro_overview(emissions_df, co2_df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emissions Data Summary (numeric)")
            st.dataframe(emissions_df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)
        with col2:
            st.subheader("CO2 Data Summary (numeric)")
            st.dataframe(co2_df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

    elif analysis_level == "Sector Analysis":
        tab1, tab2 = st.tabs(["GHG Emissions", "CO2 Emissions"])

        with tab1:
            sector_col1, sector_summary1, value_col1 = create_sector_analysis(emissions_df, "GHG Emissions")
            st.session_state["emissions_sector_col"] = sector_col1
            st.session_state["emissions_sector_summary"] = sector_summary1
            st.session_state["emissions_value_col"] = value_col1

        with tab2:
            sector_col2, sector_summary2, value_col2 = create_sector_analysis(co2_df, "CO2 Emissions")
            st.session_state["co2_sector_col"] = sector_col2
            st.session_state["co2_sector_summary"] = sector_summary2
            st.session_state["co2_value_col"] = value_col2

    elif analysis_level == "Micro Analysis":
        tab1, tab2 = st.tabs(["GHG Emissions", "CO2 Emissions"])

        with tab1:
            sector_col = st.session_state.get("emissions_sector_col")
            sector_summary = st.session_state.get("emissions_sector_summary")
            value_col = st.session_state.get("emissions_value_col")
            if sector_col and sector_summary is not None and value_col:
                create_micro_analysis(emissions_df, sector_col, sector_summary, value_col, "GHG Emissions")
            else:
                st.info("Please complete Sector Analysis for GHG Emissions first (and select a value column).")

        with tab2:
            sector_col = st.session_state.get("co2_sector_col")
            sector_summary = st.session_state.get("co2_sector_summary")
            value_col = st.session_state.get("co2_value_col")
            if sector_col and sector_summary is not None and value_col:
                create_micro_analysis(co2_df, sector_col, sector_summary, value_col, "CO2 Emissions")
            else:
                st.info("Please complete Sector Analysis for CO2 Emissions first (and select a value column).")

    elif analysis_level == "Comparison":
        create_comparison_view(emissions_df, co2_df)

    elif analysis_level == "RTE Energy Analysis":
        create_rte_analysis(rte_consumption_df, rte_production_df)

    elif analysis_level == "Micro RTE Analysis":
        create_micro_rte_analysis(rte_consumption_df, rte_production_df)

    st.markdown("---")
    st.markdown("**Data Sources:** SNCF Greenhouse Gas Emissions, CO2 Data, and RTE Energy Data")


if __name__ == "__main__":
    main()
