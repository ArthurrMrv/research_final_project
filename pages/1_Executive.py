import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_rte_production, load_rte_consumption
from features_rte import build_hourly_prod_timeseries, build_15min_demand_timeseries, compute_renewable_share, join_prod_demand, surplus_proxy
from simulator import make_synthetic_sncf_load, shift_load_greedy, score_from_renewables, emissions_proxy

st.title("📌 Executive Summary")

df_prod = compute_renewable_share(build_hourly_prod_timeseries(load_rte_production()))
df_dem = build_15min_demand_timeseries(load_rte_consumption())
df_join = surplus_proxy(join_prod_demand(df_prod, df_dem), top_green_pct=0.15)
df_join["green_score"] = score_from_renewables(df_join)

st.sidebar.header("Scenario")
shiftable_share = st.sidebar.slider("Shiftable share of SNCF electricity load", 0.0, 0.5, 0.15, 0.01)
window_hours = st.sidebar.selectbox("Max shift window", [1, 2, 4, 8, 12], index=2)
annual_mwh = st.sidebar.number_input("Assumed annual SNCF electricity (MWh)", min_value=1000.0, value=1_000_000.0, step=50_000.0)

freq_minutes = 15
window_steps = int((window_hours * 60) / freq_minutes)

ts = pd.to_datetime(df_join["ts"])
sncf_load = make_synthetic_sncf_load(pd.DatetimeIndex(ts), annual_mwh=annual_mwh)
df_join["sncf_mwh"] = sncf_load.values

sim = shift_load_greedy(
    df_join,
    load_col="sncf_mwh",
    score_col="green_score",
    shiftable_share=shiftable_share,
    window_steps=window_steps,
    freq_minutes=freq_minutes
)

base_em = emissions_proxy(sim["sncf_mwh"], sim["fossil_share"])
opt_em = emissions_proxy(sim["sncf_mwh_opt"], sim["fossil_share"])

c1, c2, c3 = st.columns(3)
c1.metric("Baseline CO₂ (proxy, tCO₂)", f"{base_em:,.0f}")
c2.metric("Optimized CO₂ (proxy, tCO₂)", f"{opt_em:,.0f}", delta=f"{(opt_em-base_em):,.0f}")
c3.metric("Estimated reduction", f"{(base_em-opt_em):,.0f} tCO₂", f"{(base_em-opt_em)/base_em*100:.2f}%")

st.caption("CO₂ is a proxy based on fossil share. Replace with real carbon intensity when available.")

view_days = st.slider("Show days", 1, 14, 7)
end = sim["ts"].max()
start = end - pd.Timedelta(days=view_days)
sub = sim[(sim["ts"] >= start) & (sim["ts"] <= end)].copy()

fig = px.line(
    sub,
    x="ts",
    y=["sncf_mwh", "sncf_mwh_opt"],
    title="SNCF load (synthetic) – baseline vs shifted"
)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(sub, x="ts", y="renewable_share", title="Renewable share (RTE production)")
st.plotly_chart(fig2, use_container_width=True)
