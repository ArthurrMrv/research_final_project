# dashboard.py
"""
Run:
  streamlit run dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Optional, Tuple, Dict


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
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        text-align: center;
        opacity: 0.85;
        margin-bottom: 1.25rem;
    }
    .hint {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load SNCF parquet data + RTE processed parquet data.

    Expected paths:
      ./data/sncf/bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet
      ./data/sncf/emission-co2-perimetre-complet.parquet
      ./data/rte/conso_mix_RTE_2023_processed.parquet
      ./data/rte/RealisationDonneesProduction_2023_processed.parquet
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    # SNCF
    sncf_dir = data_dir / "sncf"
    emissions_path = sncf_dir / "bilans-des-emissions-de-gaz-a-effet-de-serre-sncf.parquet"
    co2_path = sncf_dir / "emission-co2-perimetre-complet.parquet"

    if not emissions_path.exists():
        raise FileNotFoundError(f"Missing SNCF GHG parquet: {emissions_path}")
    if not co2_path.exists():
        raise FileNotFoundError(f"Missing SNCF CO2 parquet: {co2_path}")

    emissions_df = pd.read_parquet(emissions_path)
    co2_df = pd.read_parquet(co2_path)

    # RTE (processed)
    rte_dir = data_dir / "rte"
    cons_path = rte_dir / "conso_mix_RTE_2023_processed.parquet"
    prod_path = rte_dir / "RealisationDonneesProduction_2023_processed.parquet"

    rte_consumption_df = pd.read_parquet(cons_path) if cons_path.exists() else None
    rte_production_df = pd.read_parquet(prod_path) if prod_path.exists() else None

    return emissions_df, co2_df, rte_consumption_df, rte_production_df


# -----------------------------
# Column detection utilities
# -----------------------------
DATE_KEYWORDS = ["date", "jour", "annee", "year", "time", "temps"]
HOUR_KEYWORDS = ["heure", "heures", "hour", "horaire"]

SECTOR_KEYWORDS = [
    "secteur",
    "sector",
    "categorie",
    "category",
    "poste",
    "postes",
    "type",
    "activite",
    "activité",
    "activity",
    "perimetre",
    "périmètre",
    "scope",
    "source",
    "famille",
    "libelle",
    "libellé",
    "designation",
    "désignation",
]

VALUE_KEYWORDS = [
    "emission",
    "émission",
    "ges",
    "ghg",
    "co2",
    "tonne",
    "tco2",
    "kgco2",
    "gco2",
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


def _lower(s: str) -> str:
    return str(s).lower()


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    # prefer actual datetime dtype
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    # else search by keyword
    candidates = []
    for c in df.columns:
        name = _lower(c)
        if any(k in name for k in DATE_KEYWORDS):
            candidates.append(c)
    return candidates[0] if candidates else None


def detect_hour_column(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for c in df.columns:
        name = _lower(c)
        if any(k in name for k in HOUR_KEYWORDS):
            candidates.append(c)
    # In your processed RTE files you likely have "Heures"
    if "Heures" in df.columns:
        return "Heures"
    return candidates[0] if candidates else None


def detect_numeric_columns(df: pd.DataFrame, keywords: Optional[List[str]] = None) -> List[str]:
    if keywords is None:
        keywords = VALUE_KEYWORDS

    numeric_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            name = _lower(c)
            if any(k in name for k in keywords):
                numeric_cols.append(c)

    # fallback: any numeric cols
    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    cat_cols = []
    for c in df.columns:
        name = _lower(c)
        if any(k in name for k in SECTOR_KEYWORDS):
            if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
                cat_cols.append(c)

    # low-cardinality object columns (useful for grouping)
    for c in df.columns:
        if df[c].dtype == "object" and df[c].nunique(dropna=True) < 60:
            if c not in cat_cols:
                cat_cols.append(c)

    return cat_cols


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


# -----------------------------
# RTE helpers: datetime reconstruction + renewable share
# -----------------------------
def extract_hour_from_heures(val) -> int:
    """
    Heures formats seen:
      - "00:00-01:00"
      - "00:00"
      - "00:15" (consumption sometimes 15-min steps)
    We'll extract the first hour integer.
    """
    if pd.isna(val):
        return 0
    s = str(val)
    if "-" in s:
        s = s.split("-")[0]
    if ":" in s:
        try:
            return int(s.split(":")[0])
        except Exception:
            return 0
    return 0


def add_datetime_from_date_and_heures(df: pd.DataFrame, date_col: str, heures_col: Optional[str]) -> pd.DataFrame:
    out = ensure_datetime(df, date_col)
    if not date_col or date_col not in out.columns:
        return out

    if heures_col and heures_col in out.columns:
        out["hour"] = out[heures_col].apply(extract_hour_from_heures)
        out["datetime"] = out[date_col] + pd.to_timedelta(out["hour"], unit="h")
    else:
        out["datetime"] = out[date_col]

    return out


def get_total_production(df_prod: pd.DataFrame) -> pd.Series:
    if df_prod is None or df_prod.empty:
        return pd.Series(dtype=float)

    if "Total" in df_prod.columns and pd.api.types.is_numeric_dtype(df_prod["Total"]):
        return df_prod["Total"]

    exclude = {"date", "datetime", "Heures", "hour"}
    prod_cols = [c for c in df_prod.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_prod[c])]
    if not prod_cols:
        return pd.Series(np.nan, index=df_prod.index)
    return df_prod[prod_cols].sum(axis=1)


def get_total_consumption(df_cons: pd.DataFrame) -> pd.Series:
    if df_cons is None or df_cons.empty:
        return pd.Series(dtype=float)

    # common column in RTE consumption file
    if "Consommation" in df_cons.columns and pd.api.types.is_numeric_dtype(df_cons["Consommation"]):
        return df_cons["Consommation"]

    exclude = {"date", "datetime", "Heures", "hour"}
    cons_cols = [c for c in df_cons.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_cons[c])]
    if not cons_cols:
        return pd.Series(np.nan, index=df_cons.index)
    # take first numeric as "main" if ambiguous
    return df_cons[cons_cols[0]]


def guess_renewable_columns(df_prod: pd.DataFrame) -> List[str]:
    """
    Heuristic: renewable-ish labels in French RTE mix often include:
      - "Solaire", "Eolien", "Hydraulique", "Bioénergies"
      - also 'Solar', 'Wind', 'Hydro', 'Biomass'
    """
    if df_prod is None or df_prod.empty:
        return []
    candidates = []
    for c in df_prod.columns:
        if not pd.api.types.is_numeric_dtype(df_prod[c]):
            continue
        name = _lower(c)
        if any(k in name for k in ["sol", "eol", "wind", "hydro", "hydr", "bio", "renouvel", "pv"]):
            if c not in ["Total"]:
                candidates.append(c)
    return candidates


# -----------------------------
# Visualization blocks (same “structure” as your original)
# -----------------------------
def create_macro_overview(emissions_df: pd.DataFrame, co2_df: pd.DataFrame,
                          rte_consumption_df: Optional[pd.DataFrame], rte_production_df: Optional[pd.DataFrame]) -> None:
    st.header("📊 Macro Overview — Big Picture (Annual Report)")

    # SNCF totals
    e_num = detect_numeric_columns(emissions_df)
    c_num = detect_numeric_columns(co2_df)

    e_val = e_num[0] if e_num else None
    c_val = c_num[0] if c_num else None

    total_ghg = emissions_df[e_val].sum() if e_val else np.nan
    total_co2 = co2_df[c_val].sum() if c_val else np.nan

    # RTE totals (2023)
    total_cons = np.nan
    total_prod = np.nan
    if rte_consumption_df is not None:
        dcol = detect_date_column(rte_consumption_df) or "date"
        hcol = detect_hour_column(rte_consumption_df)
        cons = add_datetime_from_date_and_heures(rte_consumption_df, dcol, hcol)
        total_cons = get_total_consumption(cons).sum()

    if rte_production_df is not None:
        dcol = detect_date_column(rte_production_df) or "date"
        hcol = detect_hour_column(rte_production_df)
        prod = add_datetime_from_date_and_heures(rte_production_df, dcol, hcol)
        total_prod = get_total_production(prod).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SNCF GHG (sum)", f"{total_ghg:,.0f}" if np.isfinite(total_ghg) else "N/A")
    col2.metric("Total SNCF CO₂ (sum)", f"{total_co2:,.0f}" if np.isfinite(total_co2) else "N/A")
    col3.metric("RTE Total Consumption 2023 (sum)", f"{total_cons:,.0f} MW" if np.isfinite(total_cons) else "N/A")
    col4.metric("RTE Total Production 2023 (sum)", f"{total_prod:,.0f} MW" if np.isfinite(total_prod) else "N/A")

    st.markdown(
        '<div class="hint">Tip: If the totals look weird, pick a different value column in “Sector Analysis”. '
        "Some datasets contain multiple numeric fields.</div>",
        unsafe_allow_html=True,
    )

    # SNCF: top contributors quick chart (GHG + CO2)
    st.subheader("Top contributors (SNCF)")
    e_cat = detect_categorical_columns(emissions_df)
    c_cat = detect_categorical_columns(co2_df)

    left, right = st.columns(2)

    with left:
        st.caption("GHG — top categories")
        if e_cat and e_val:
            cat = st.selectbox("GHG category column", e_cat, key="macro_ghg_cat")
            top = (
                emissions_df.groupby(cat, dropna=False)[e_val]
                .sum()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
                .rename(columns={e_val: "Total"})
            )
            fig = px.bar(top, x="Total", y=cat, orientation="h", title="Top 12 — GHG")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not detect a good categorical/value column for GHG.")

    with right:
        st.caption("CO₂ — top categories")
        if c_cat and c_val:
            cat = st.selectbox("CO₂ category column", c_cat, key="macro_co2_cat")
            top = (
                co2_df.groupby(cat, dropna=False)[c_val]
                .sum()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
                .rename(columns={c_val: "Total"})
            )
            fig = px.bar(top, x="Total", y=cat, orientation="h", title="Top 12 — CO₂")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not detect a good categorical/value column for CO₂.")

    # RTE: monthly totals (good for annual report)
    st.subheader("RTE monthly patterns (2023)")
    if rte_consumption_df is None and rte_production_df is None:
        st.info("No RTE processed parquet found. (Run data_processing.py or check ./data/rte files.)")
        return

    if rte_consumption_df is not None:
        dcol = detect_date_column(rte_consumption_df) or "date"
        hcol = detect_hour_column(rte_consumption_df)
        cons = add_datetime_from_date_and_heures(rte_consumption_df, dcol, hcol)
        cons["month"] = pd.to_datetime(cons["datetime"]).dt.to_period("M").dt.to_timestamp()
        cons_val = get_total_consumption(cons)
        cons_m = cons.assign(cons=cons_val).groupby("month")["cons"].sum().reset_index()
    else:
        cons_m = None

    if rte_production_df is not None:
        dcol = detect_date_column(rte_production_df) or "date"
        hcol = detect_hour_column(rte_production_df)
        prod = add_datetime_from_date_and_heures(rte_production_df, dcol, hcol)
        prod["month"] = pd.to_datetime(prod["datetime"]).dt.to_period("M").dt.to_timestamp()
        prod_val = get_total_production(prod)
        prod_m = prod.assign(prod=prod_val).groupby("month")["prod"].sum().reset_index()
    else:
        prod_m = None

    fig = go.Figure()
    if prod_m is not None and not prod_m.empty:
        fig.add_trace(go.Scatter(x=prod_m["month"], y=prod_m["prod"], mode="lines+markers", name="Production (sum)"))
    if cons_m is not None and not cons_m.empty:
        fig.add_trace(go.Scatter(x=cons_m["month"], y=cons_m["cons"], mode="lines+markers", name="Consumption (sum)"))
    fig.update_layout(height=420, title="Monthly totals — Production vs Consumption", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def create_sector_analysis(df: pd.DataFrame, df_name: str = "Dataset") -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame]]:
    st.header(f"🏭 Sector Analysis — {df_name}")

    categorical_cols = detect_categorical_columns(df)
    numeric_cols = detect_numeric_columns(df)

    if not categorical_cols or not numeric_cols:
        st.info(f"No good categorical/numeric columns detected for {df_name}.")
        with st.expander("Show columns"):
            st.write(df.columns.tolist())
        return None, None, None

    # Let user pick grouping + value
    sector_col = st.selectbox(f"Sector/category column ({df_name})", categorical_cols, key=f"sector_{df_name}")
    value_col = st.selectbox(f"Value column ({df_name})", numeric_cols, key=f"value_{df_name}")

    # Aggregate
    sector_summary = (
        df.groupby(sector_col, dropna=False)[value_col]
        .agg(Total="sum", Average="mean", Count="count")
        .reset_index()
        .sort_values("Total", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(f"Top N sectors ({df_name})", 5, 40, 15, key=f"topn_{df_name}")
        top = sector_summary.head(top_n)
        fig = px.bar(top, x="Total", y=sector_col, orientation="h", title=f"Top {top_n} by Total")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top = sector_summary.head(12)
        fig2 = px.pie(top, values="Total", names=sector_col, hole=0.45, title="Top 12 share")
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Summary table")
    st.dataframe(sector_summary, use_container_width=True)

    return sector_col, value_col, sector_summary


def create_micro_analysis(df: pd.DataFrame, sector_col: str, value_col: str, sector_summary: pd.DataFrame, df_name: str = "Dataset") -> None:
    st.header(f"🔬 Micro Analysis — {df_name}")

    if sector_col is None or value_col is None or sector_summary is None:
        st.info("Run Sector Analysis first so we know which columns to drill into.")
        return

    selected_sector = st.selectbox(
        f"Choose one {sector_col} to drill down",
        sector_summary[sector_col].astype(str).tolist(),
        key=f"micro_pick_{df_name}",
    )

    # keep original types for filtering
    sector_data = df[df[sector_col].astype(str) == str(selected_sector)].copy()

    st.write(f"Rows in selection: **{len(sector_data):,}**")

    other_cats = [c for c in detect_categorical_columns(sector_data) if c != sector_col]
    if other_cats:
        sub_col = st.selectbox("Optional: break down by", other_cats, key=f"micro_sub_{df_name}")
        sub = (
            sector_data.groupby(sub_col, dropna=False)[value_col]
            .agg(Total="sum", Average="mean", Count="count")
            .reset_index()
            .sort_values("Total", ascending=False)
        )

        left, right = st.columns(2)
        with left:
            fig = px.bar(sub.head(20), x="Total", y=sub_col, orientation="h", title=f"Top 20 {sub_col} within {selected_sector}")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(sub, use_container_width=True)
    else:
        st.info("No other categorical columns detected for a second-level breakdown.")

    with st.expander("Show raw rows for this selection"):
        st.dataframe(sector_data, use_container_width=True)


def create_comparison_view(emissions_df: pd.DataFrame, co2_df: pd.DataFrame) -> None:
    st.header("⚖️ Comparative Analysis — GHG vs CO₂")

    e_num = detect_numeric_columns(emissions_df)
    c_num = detect_numeric_columns(co2_df)
    if not e_num or not c_num:
        st.info("Not enough numeric columns detected to compare.")
        return

    e_cat = detect_categorical_columns(emissions_df)
    c_cat = detect_categorical_columns(co2_df)
    common = list(sorted(set(e_cat).intersection(set(c_cat))))

    if not common:
        st.info("No common categorical column detected between GHG and CO₂ datasets.")
        st.caption("You can still compare by choosing separate columns below.")
        left, right = st.columns(2)
        with left:
            e_group = st.selectbox("GHG group-by column", e_cat if e_cat else emissions_df.columns.tolist(), key="cmp_e_group")
        with right:
            c_group = st.selectbox("CO₂ group-by column", c_cat if c_cat else co2_df.columns.tolist(), key="cmp_c_group")

        e_val = st.selectbox("GHG value column", e_num, key="cmp_e_val")
        c_val = st.selectbox("CO₂ value column", c_num, key="cmp_c_val")

        e_agg = emissions_df.groupby(e_group, dropna=False)[e_val].sum().reset_index().rename(columns={e_val: "GHG"})
        c_agg = co2_df.groupby(c_group, dropna=False)[c_val].sum().reset_index().rename(columns={c_val: "CO2"})
        # Different keys: show as two charts side-by-side
        col1, col2 = st.columns(2)
        with col1:
            top = e_agg.sort_values("GHG", ascending=False).head(15)
            fig = px.bar(top, x="GHG", y=e_group, orientation="h", title="GHG — Top 15")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top = c_agg.sort_values("CO2", ascending=False).head(15)
            fig = px.bar(top, x="CO2", y=c_group, orientation="h", title="CO₂ — Top 15")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        return

    # Common comparison path
    group_col = st.selectbox("Compare using this shared column", common, key="cmp_common")
    e_val = st.selectbox("GHG value column", e_num, key="cmp_e_val2")
    c_val = st.selectbox("CO₂ value column", c_num, key="cmp_c_val2")

    e_agg = emissions_df.groupby(group_col, dropna=False)[e_val].sum().reset_index().rename(columns={e_val: "GHG"})
    c_agg = co2_df.groupby(group_col, dropna=False)[c_val].sum().reset_index().rename(columns={c_val: "CO2"})

    comp = pd.merge(e_agg, c_agg, on=group_col, how="outer").fillna(0.0)
    comp["GHG"] = pd.to_numeric(comp["GHG"], errors="coerce").fillna(0.0)
    comp["CO2"] = pd.to_numeric(comp["CO2"], errors="coerce").fillna(0.0)
    comp = comp.sort_values("GHG", ascending=False).head(20)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="GHG", x=comp[group_col].astype(str), y=comp["GHG"]))
    fig.add_trace(go.Bar(name="CO₂", x=comp[group_col].astype(str), y=comp["CO2"]))
    fig.update_layout(barmode="group", height=520, title=f"GHG vs CO₂ by {group_col}", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    if len(comp) > 2:
        corr = comp["GHG"].corr(comp["CO2"])
        st.metric("Correlation on Top 20 groups", f"{corr:.3f}")


# -----------------------------
# RTE analysis pages (meaningful for annual report)
# -----------------------------
def create_rte_analysis(rte_consumption_df: Optional[pd.DataFrame], rte_production_df: Optional[pd.DataFrame]) -> None:
    st.header("⚡ RTE Energy Analysis — 2023 Patterns")

    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE processed parquet found in ./data/rte/")
        st.stop()

    # Build “datetime” for each dataset
    cons = None
    prod = None

    if rte_consumption_df is not None:
        dcol = detect_date_column(rte_consumption_df) or "date"
        hcol = detect_hour_column(rte_consumption_df)
        cons = add_datetime_from_date_and_heures(rte_consumption_df, dcol, hcol)
        cons["consumption_total"] = get_total_consumption(cons)

    if rte_production_df is not None:
        dcol = detect_date_column(rte_production_df) or "date"
        hcol = detect_hour_column(rte_production_df)
        prod = add_datetime_from_date_and_heures(rte_production_df, dcol, hcol)
        prod["production_total"] = get_total_production(prod)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    if prod is not None:
        k1.metric("Production (sum)", f"{prod['production_total'].sum():,.0f} MW")
        k2.metric("Production (mean)", f"{prod['production_total'].mean():,.1f} MW")
    else:
        k1.metric("Production (sum)", "N/A")
        k2.metric("Production (mean)", "N/A")

    if cons is not None:
        k3.metric("Consumption (sum)", f"{cons['consumption_total'].sum():,.0f} MW")
        k4.metric("Consumption (mean)", f"{cons['consumption_total'].mean():,.1f} MW")
    else:
        k3.metric("Consumption (sum)", "N/A")
        k4.metric("Consumption (mean)", "N/A")

    # -----------------
    # Section: Time series (daily aggregated to keep it readable)
    # -----------------
    st.subheader("Daily time series (aggregated)")
    fig = go.Figure()

    if prod is not None:
        prod_daily = prod.groupby(pd.to_datetime(prod["datetime"]).dt.date)["production_total"].sum().reset_index()
        prod_daily.columns = ["day", "production_sum"]
        fig.add_trace(go.Scatter(x=prod_daily["day"], y=prod_daily["production_sum"], mode="lines", name="Production (daily sum)"))

    if cons is not None:
        cons_daily = cons.groupby(pd.to_datetime(cons["datetime"]).dt.date)["consumption_total"].sum().reset_index()
        cons_daily.columns = ["day", "consumption_sum"]
        fig.add_trace(go.Scatter(x=cons_daily["day"], y=cons_daily["consumption_sum"], mode="lines", name="Consumption (daily sum)"))

    fig.update_layout(height=420, hovermode="x unified", title="Daily totals — Production vs Consumption")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------
    # Section: Balance (if both)
    # -----------------
    if prod is not None and cons is not None:
        st.subheader("Balance — Production minus Consumption")
        balance = pd.merge(
            prod[["datetime", "production_total"]],
            cons[["datetime", "consumption_total"]],
            on="datetime",
            how="inner",
        ).dropna()
        balance["balance"] = balance["production_total"] - balance["consumption_total"]

        # downsample for speed if needed
        if len(balance) > 20000:
            balance_plot = balance.iloc[:: max(1, len(balance) // 20000)].copy()
        else:
            balance_plot = balance

        figb = go.Figure()
        figb.add_trace(go.Scatter(x=balance_plot["datetime"], y=balance_plot["balance"], mode="lines", name="Balance (MW)"))
        figb.add_hline(y=0, line_dash="dash")
        figb.update_layout(height=420, title="Balance over time (MW)", hovermode="x unified")
        st.plotly_chart(figb, use_container_width=True)

        # Identify “surplus” periods (top quantile)
        q = st.slider("Define 'surplus' threshold (quantile)", 0.80, 0.99, 0.95, 0.01, key="surplus_q")
        thr = balance["balance"].quantile(q)
        share = (balance["balance"] >= thr).mean() * 100
        st.caption(f"Surplus threshold = **{thr:,.0f} MW** (top {100*(1-q):.0f}% of hours). Share of hours flagged: **{share:.1f}%**")

    # -----------------
    # Section: Production mix (stacked) + renewable share
    # -----------------
    if prod is not None:
        st.subheader("Production mix (stacked, hourly)")

        exclude = {"date", "datetime", "Heures", "hour", "Total", "production_total"}
        sector_cols = [c for c in prod.columns if c not in exclude and pd.api.types.is_numeric_dtype(prod[c])]

        if sector_cols:
            # For readability: keep top sectors by total contribution
            totals = prod[sector_cols].sum().sort_values(ascending=False)
            top_k = st.slider("How many production sectors to stack", 5, min(20, len(totals)), min(10, len(totals)), key="stack_k")
            keep = totals.head(top_k).index.tolist()

            df_stack = prod[["datetime"] + keep].dropna(subset=["datetime"]).copy()
            df_stack = df_stack.sort_values("datetime")

            figm = go.Figure()
            for col in keep[::-1]:  # reverse so biggest ends up closer to bottom visually
                figm.add_trace(
                    go.Scatter(
                        x=df_stack["datetime"],
                        y=df_stack[col].fillna(0),
                        mode="lines",
                        stackgroup="one",
                        name=str(col),
                    )
                )
            figm.update_layout(height=520, hovermode="x unified", title="Production by sector (stacked)")
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("Could not detect production sector columns beyond totals.")

        # Renewable share
        st.subheader("Renewable share (heuristic from column names)")
        renew_cols = guess_renewable_columns(prod)
        if renew_cols:
            prod["renewable_total"] = prod[renew_cols].sum(axis=1)
            prod["renewable_share"] = np.where(
                prod["production_total"] > 0,
                prod["renewable_total"] / prod["production_total"],
                np.nan,
            )

            daily = prod.groupby(pd.to_datetime(prod["datetime"]).dt.date)["renewable_share"].mean().reset_index()
            daily.columns = ["day", "renewable_share_mean"]

            fig_rs = px.line(daily, x="day", y="renewable_share_mean", title="Average renewable share per day")
            fig_rs.update_layout(height=420, yaxis_tickformat=".0%")
            st.plotly_chart(fig_rs, use_container_width=True)

            # Top renewable windows
            q = st.slider("Define 'very renewable' threshold (quantile)", 0.80, 0.99, 0.90, 0.01, key="ren_q")
            thr = prod["renewable_share"].quantile(q)
            st.caption(f"Very-renewable threshold: share ≥ **{thr:.0%}** (top {100*(1-q):.0f}% of hours).")

            # Hour-of-day profile for renewable share
            prof = prod.dropna(subset=["renewable_share"]).copy()
            prof["hour_of_day"] = pd.to_datetime(prof["datetime"]).dt.hour
            by_hour = prof.groupby("hour_of_day")["renewable_share"].mean().reset_index()
            fig_h = px.bar(by_hour, x="hour_of_day", y="renewable_share", title="Average renewable share by hour of day")
            fig_h.update_layout(height=380, yaxis_tickformat=".0%")
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No renewable-looking columns detected (e.g., solaire/eolien/hydraulique/bio).")

    # -----------------
    # Section: Consumption weekly profile (good for annual report)
    # -----------------
    if cons is not None:
        st.subheader("Consumption profile — average week (hourly)")
        dfw = cons.dropna(subset=["datetime", "consumption_total"]).copy()
        dt = pd.to_datetime(dfw["datetime"])
        dfw["dow"] = dt.dt.dayofweek  # 0 Monday
        dfw["hour"] = dt.dt.hour
        dfw["hour_of_week"] = dfw["dow"] * 24 + dfw["hour"]

        avg_week = dfw.groupby("hour_of_week")["consumption_total"].mean().reset_index()
        avg_week["day"] = (avg_week["hour_of_week"] // 24).astype(int)
        avg_week["hour"] = (avg_week["hour_of_week"] % 24).astype(int)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        avg_week["label"] = avg_week.apply(lambda r: f"{day_names[r['day']]} {r['hour']:02d}:00", axis=1)

        figw = go.Figure()
        figw.add_trace(go.Scatter(x=avg_week["hour_of_week"], y=avg_week["consumption_total"], mode="lines", name="Avg consumption"))
        figw.update_xaxes(
            tickmode="linear",
            tick0=0,
            dtick=24,
            tickvals=list(range(0, 168, 24)),
            ticktext=day_names,
        )
        figw.update_layout(height=420, title="Average week — consumption (MW)", hovermode="x unified")
        st.plotly_chart(figw, use_container_width=True)


def create_micro_rte_analysis(rte_consumption_df: Optional[pd.DataFrame], rte_production_df: Optional[pd.DataFrame]) -> None:
    """
    A more “drill-down” RTE page:
      - choose a month
      - inspect hour-of-day patterns for consumption, production, renewable share
    """
    st.header("🔬 Micro RTE Analysis — Drill-down (month / hour patterns)")

    if rte_consumption_df is None and rte_production_df is None:
        st.warning("No RTE processed parquet found.")
        return

    cons = None
    prod = None
    if rte_consumption_df is not None:
        dcol = detect_date_column(rte_consumption_df) or "date"
        hcol = detect_hour_column(rte_consumption_df)
        cons = add_datetime_from_date_and_heures(rte_consumption_df, dcol, hcol)
        cons["consumption_total"] = get_total_consumption(cons)

    if rte_production_df is not None:
        dcol = detect_date_column(rte_production_df) or "date"
        hcol = detect_hour_column(rte_production_df)
        prod = add_datetime_from_date_and_heures(rte_production_df, dcol, hcol)
        prod["production_total"] = get_total_production(prod)

    # month selector based on whichever exists
    months = []
    if prod is not None:
        months += sorted(pd.to_datetime(prod["datetime"]).dt.to_period("M").dropna().unique().astype(str).tolist())
    if cons is not None:
        months += sorted(pd.to_datetime(cons["datetime"]).dt.to_period("M").dropna().unique().astype(str).tolist())
    months = sorted(list(dict.fromkeys(months)))  # unique, stable order

    if not months:
        st.info("Could not infer months from RTE datetimes.")
        return

    month_choice = st.selectbox("Select month", months, index=0, key="micro_month")
    p = pd.Period(month_choice, freq="M")
    start = p.start_time
    end = p.end_time

    c1, c2, c3 = st.columns(3)

    if prod is not None:
        mp = prod[(pd.to_datetime(prod["datetime"]) >= start) & (pd.to_datetime(prod["datetime"]) <= end)].copy()
        mp = mp.dropna(subset=["datetime", "production_total"])
        if not mp.empty:
            mp["hour"] = pd.to_datetime(mp["datetime"]).dt.hour
            by_hour = mp.groupby("hour")["production_total"].mean().reset_index()
            fig = px.line(by_hour, x="hour", y="production_total", markers=True, title="Avg production by hour (selected month)")
            fig.update_layout(height=360)
            c1.plotly_chart(fig, use_container_width=True)
        else:
            c1.info("No production rows in selected month.")

    if cons is not None:
        mc = cons[(pd.to_datetime(cons["datetime"]) >= start) & (pd.to_datetime(cons["datetime"]) <= end)].copy()
        mc = mc.dropna(subset=["datetime", "consumption_total"])
        if not mc.empty:
            mc["hour"] = pd.to_datetime(mc["datetime"]).dt.hour
            by_hour = mc.groupby("hour")["consumption_total"].mean().reset_index()
            fig = px.line(by_hour, x="hour", y="consumption_total", markers=True, title="Avg consumption by hour (selected month)")
            fig.update_layout(height=360)
            c2.plotly_chart(fig, use_container_width=True)
        else:
            c2.info("No consumption rows in selected month.")

    if prod is not None:
        renew_cols = guess_renewable_columns(prod)
        if renew_cols and not mp.empty:
            mp["renewable_total"] = mp[renew_cols].sum(axis=1)
            mp["renewable_share"] = np.where(mp["production_total"] > 0, mp["renewable_total"] / mp["production_total"], np.nan)
            by_hour = mp.groupby(pd.to_datetime(mp["datetime"]).dt.hour)["renewable_share"].mean().reset_index()
            fig = px.bar(by_hour, x="datetime", y="renewable_share", title="Avg renewable share by hour (selected month)")
            fig.update_layout(height=360, yaxis_tickformat=".0%")
            c3.plotly_chart(fig, use_container_width=True)
        else:
            c3.info("Renewable share not available (no renewable-like columns detected).")

    st.divider()
    st.subheader("Selected month — time series")
    fig_ts = go.Figure()
    if prod is not None and not mp.empty:
        fig_ts.add_trace(go.Scatter(x=mp["datetime"], y=mp["production_total"], mode="lines", name="Production"))
    if cons is not None and not mc.empty:
        fig_ts.add_trace(go.Scatter(x=mc["datetime"], y=mc["consumption_total"], mode="lines", name="Consumption"))
    fig_ts.update_layout(height=420, hovermode="x unified", title=f"{month_choice} — production vs consumption")
    st.plotly_chart(fig_ts, use_container_width=True)


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.markdown('<div class="main-header">⚡ SNCF & RTE Energy Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Granular analysis (macro → meso → micro) using your local 2023 files</div>', unsafe_allow_html=True)

    # Load data
    try:
        with st.spinner("Loading local parquet data..."):
            emissions_df, co2_df, rte_consumption_df, rte_production_df = load_data()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

    # Sidebar overview
    st.sidebar.header("📋 Data Overview")
    st.sidebar.write(f"**SNCF GHG:** {emissions_df.shape[0]} rows × {emissions_df.shape[1]} cols")
    st.sidebar.write(f"**SNCF CO₂:** {co2_df.shape[0]} rows × {co2_df.shape[1]} cols")
    if rte_consumption_df is not None:
        st.sidebar.write(f"**RTE Consumption:** {rte_consumption_df.shape[0]} rows × {rte_consumption_df.shape[1]} cols")
    else:
        st.sidebar.write("**RTE Consumption:** not found (missing processed parquet)")
    if rte_production_df is not None:
        st.sidebar.write(f"**RTE Production:** {rte_production_df.shape[0]} rows × {rte_production_df.shape[1]} cols")
    else:
        st.sidebar.write("**RTE Production:** not found (missing processed parquet)")

    with st.sidebar.expander("👀 Show columns"):
        st.write("**SNCF GHG columns:**", list(emissions_df.columns))
        st.write("**SNCF CO₂ columns:**", list(co2_df.columns))
        if rte_consumption_df is not None:
            st.write("**RTE Consumption columns:**", list(rte_consumption_df.columns))
        if rte_production_df is not None:
            st.write("**RTE Production columns:**", list(rte_production_df.columns))

    # Navigation
    options = ["Macro Overview", "Sector Analysis", "Micro Analysis", "Comparison", "RTE Energy Analysis", "Micro RTE Analysis"]
    analysis_level = st.sidebar.radio("Analysis Level", options, index=0)

    # State holders for micro pages
    if "ghg_sector_col" not in st.session_state:
        st.session_state["ghg_sector_col"] = None
        st.session_state["ghg_value_col"] = None
        st.session_state["ghg_sector_summary"] = None

    if "co2_sector_col" not in st.session_state:
        st.session_state["co2_sector_col"] = None
        st.session_state["co2_value_col"] = None
        st.session_state["co2_sector_summary"] = None

    # Render
    if analysis_level == "Macro Overview":
        create_macro_overview(emissions_df, co2_df, rte_consumption_df, rte_production_df)

        st.subheader("Quick numeric summaries (SNCF)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("SNCF GHG — describe() on numeric columns")
            st.dataframe(emissions_df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)
        with c2:
            st.caption("SNCF CO₂ — describe() on numeric columns")
            st.dataframe(co2_df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

    elif analysis_level == "Sector Analysis":
        tab1, tab2 = st.tabs(["SNCF GHG", "SNCF CO₂"])

        with tab1:
            sec, val, summ = create_sector_analysis(emissions_df, "SNCF GHG")
            st.session_state["ghg_sector_col"] = sec
            st.session_state["ghg_value_col"] = val
            st.session_state["ghg_sector_summary"] = summ

        with tab2:
            sec, val, summ = create_sector_analysis(co2_df, "SNCF CO₂")
            st.session_state["co2_sector_col"] = sec
            st.session_state["co2_value_col"] = val
            st.session_state["co2_sector_summary"] = summ

    elif analysis_level == "Micro Analysis":
        tab1, tab2 = st.tabs(["SNCF GHG", "SNCF CO₂"])

        with tab1:
            if st.session_state["ghg_sector_col"] and st.session_state["ghg_value_col"] and st.session_state["ghg_sector_summary"] is not None:
                create_micro_analysis(
                    emissions_df,
                    st.session_state["ghg_sector_col"],
                    st.session_state["ghg_value_col"],
                    st.session_state["ghg_sector_summary"],
                    "SNCF GHG",
                )
            else:
                st.info("Go to Sector Analysis (SNCF GHG) first to select the grouping/value columns.")

        with tab2:
            if st.session_state["co2_sector_col"] and st.session_state["co2_value_col"] and st.session_state["co2_sector_summary"] is not None:
                create_micro_analysis(
                    co2_df,
                    st.session_state["co2_sector_col"],
                    st.session_state["co2_value_col"],
                    st.session_state["co2_sector_summary"],
                    "SNCF CO₂",
                )
            else:
                st.info("Go to Sector Analysis (SNCF CO₂) first to select the grouping/value columns.")

    elif analysis_level == "Comparison":
        create_comparison_view(emissions_df, co2_df)

    elif analysis_level == "RTE Energy Analysis":
        create_rte_analysis(rte_consumption_df, rte_production_df)

    elif analysis_level == "Micro RTE Analysis":
        create_micro_rte_analysis(rte_consumption_df, rte_production_df)

    # Footer
    st.markdown("---")
    st.markdown("**Data Sources:** SNCF emissions & CO₂ parquet files + RTE processed 2023 parquet files (local).")


if __name__ == "__main__":
    main()
