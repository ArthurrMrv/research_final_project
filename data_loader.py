import pandas as pd
import streamlit as st
from config import RTE_PROD_PARQUET, RTE_CONS_PARQUET, SNCF_EMISSIONS_1, SNCF_EMISSIONS_2

def _safe_read_parquet(path):
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_rte_production():
    df = _safe_read_parquet(RTE_PROD_PARQUET)
    if df is None:
        raise FileNotFoundError(f"Missing: {RTE_PROD_PARQUET}")
    return df

@st.cache_data(show_spinner=False)
def load_rte_consumption():
    df = _safe_read_parquet(RTE_CONS_PARQUET)
    if df is None:
        raise FileNotFoundError(f"Missing: {RTE_CONS_PARQUET}")
    return df

@st.cache_data(show_spinner=False)
def load_sncf_emissions():
    dfs = {}
    df1 = _safe_read_parquet(SNCF_EMISSIONS_1)
    df2 = _safe_read_parquet(SNCF_EMISSIONS_2)
    if df1 is not None:
        dfs["sncf_ges_bilan"] = df1
    if df2 is not None:
        dfs["sncf_co2_perimetre"] = df2
    if not dfs:
        dfs["sncf_placeholder"] = pd.DataFrame({"info": ["No SNCF parquet found in data/sncf"]})
    return dfs
