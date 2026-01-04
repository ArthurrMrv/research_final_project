import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_loader import load_rte_production, load_rte_consumption
from features_rte import (
    build_hourly_prod_timeseries,
    build_15min_demand_timeseries,
    compute_renewable_share,
    compute_modelled_carbon_intensity,
    join_prod_demand,
)
from simulator import (
    make_synthetic_sncf_load,
    build_objective_score,
    shift_load_constrained,
    compute_consumption_weighted_ci,
    compute_total_emissions_tco2,
)

st.title("🧠 Time-shifting Simulator (client-grade)")

# ------------------------
# Load + feature engineering
# ------------------------
df_prod = load_rte_production()
df_prod = build_hourly_prod_timeseries(df_prod)
df_prod = compute_renewable_share(df_prod)
df_prod = compute_modelled_carbon_intensity(df_prod)

df_dem = load_rte_consumption()
df_dem = build_15min_demand_timeseries(df_dem)

df = join_prod_demand(df_prod, df_dem)
df = df.sort_values("ts").reset_index(drop=True)

# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("Controls")

mode = st.sidebar.selectbox("Objective", ["Minimize CO₂ (modelled CI)", "Maximize renewable share"], index=0)
objective_mode = "co2" if mode.startswith("Minimize") else "green"

shiftable_share = st.sidebar.slider("Shiftable share of SNCF electricity load", 0.0, 0.6, 0.15, 0.01)
window_hours = st.sidebar.selectbox("Max shift window (hours)", [1, 2, 4, 8, 12], index=2)

cap_multiplier = st.sidebar.slider("Max extra load per timestep (capacity cap)", 1.0, 2.0, 1.20, 0.05)

block_peak = st.sidebar.checkbox("Block shifting INTO peak hours", value=True)
peak_start = st.sidebar.selectbox("Peak start hour", list(range(0, 24)), index=7)
peak_end = st.sidebar.selectbox("Peak end hour (exclusive)", list(range(1, 25)), index=10)

use_ramp = st.sidebar.checkbox("Limit ramping (smooth changes)", value=True)
ramp_limit_multiplier = st.sidebar.slider("Ramp limit (× baseline per step)", 0.0, 1.0, 0.15, 0.05) if use_ramp else None

distance_penalty = st.sidebar.slider("Prefer short shifts (penalty)", 0.0, 2.0, 0.20, 0.05)
st.sidebar.caption("Higher penalty discourages long shifts even if cleaner.")

# SNCF load input
st.sidebar.header("SNCF load input")
load_mode = st.sidebar.radio("Load source", ["Synthetic (assumption)", "Upload CSV (ts, sncf_mwh)"], index=0)
annual_mwh = st.sidebar.number_input("Annual electricity (MWh) if synthetic", min_value=10_000.0, value=1_000_000.0, step=50_000.0)

if load_mode.startswith("Upload"):
    up = st.sidebar.file_uploader("Upload CSV with columns: ts, sncf_mwh", type=["csv"])
    if up is not None:
        user = pd.read_csv(up)
        user["ts"] = pd.to_datetime(user["ts"])
        user = user.sort_values("ts")
        df = df.merge(user[["ts", "sncf_mwh"]], on="ts", how="left")
        if df["sncf_mwh"].isna().any():
            st.error("Uploaded time series does not match all RTE 15-min timestamps. Please align timestamps exactly.")
            st.stop()
    else:
        df["sncf_mwh"] = make_synthetic_sncf_load(pd.DatetimeIndex(df["ts"]), annual_mwh=annual_mwh).values
else:
    df["sncf_mwh"] = make_synthetic_sncf_load(pd.DatetimeIndex(df["ts"]), annual_mwh=annual_mwh).values

# ------------------------
# Run constrained shifting
# ------------------------
freq_minutes = 15
window_steps = int((window_hours * 60) / freq_minutes)

score = build_objective_score(df, mode=objective_mode)
no_shift_hours = (peak_start, peak_end) if block_peak else None

sim = shift_load_constrained(
    df=df,
    load_col="sncf_mwh",
    score=score,
    shiftable_share=shiftable_share,
    window_steps=window_steps,
    freq_minutes=freq_minutes,
    cap_multiplier=cap_multiplier,
    no_shift_hours=no_shift_hours,
    ramp_limit_multiplier=ramp_limit_multiplier,
    distance_penalty=distance_penalty,
)

# ------------------------
# KPIs (client-grade)
# ------------------------
ci_col = "gco2_per_kwh_model" if "gco2_per_kwh_model" in sim.columns else None
if ci_col is None:
    st.warning("Carbon intensity not found. Did you add compute_modelled_carbon_intensity()?")

base_ci = compute_consumption_weighted_ci(sim["sncf_mwh"], sim[ci_col]) if ci_col else float("nan")
opt_ci = compute_consumption_weighted_ci(sim["sncf_mwh_opt"], sim[ci_col]) if ci_col else float("nan")

base_tco2 = compute_total_emissions_tco2(sim["sncf_mwh"], sim[ci_col]) if ci_col else float("nan")
opt_tco2 = compute_total_emissions_tco2(sim["sncf_mwh_opt"], sim[ci_col]) if ci_col else float("nan")

shifted_mwh = sim.attrs.get("shifted_mwh_total", float("nan"))
avg_shift_min = sim.attrs.get("avg_shift_minutes", 0.0)
cap_viol = sim.attrs.get("cap_violations", 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline emissions (tCO₂)", f"{base_tco2:,.0f}")
k2.metric("Optimized emissions (tCO₂)", f"{opt_tco2:,.0f}", delta=f"{(opt_tco2-base_tco2):,.0f}")
k3.metric("Avg CI consumed (gCO₂/kWh)", f"{base_ci:.1f} → {opt_ci:.1f}", delta=f"{(opt_ci-base_ci):.1f}")
k4.metric("Energy shifted (MWh)", f"{shifted_mwh:,.0f}")

st.caption(
    f"Avg shift distance: {avg_shift_min:.0f} minutes | Capacity cap violations: {cap_viol} (should be 0)"
)

# ------------------------
# Plots: make it feel real
# ------------------------
st.subheader("1) What changed over time? (select a window)")

days = st.slider("Days to display", 1, 30, 7)
end = sim["ts"].max()
start = end - pd.Timedelta(days=days)
sub = sim[(sim["ts"] >= start) & (sim["ts"] <= end)].copy()

fig = px.line(sub, x="ts", y=["sncf_mwh", "sncf_mwh_opt"], title="SNCF electricity load: baseline vs shifted")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(sub, x="ts", y=[ci_col], title="Grid carbon intensity (modelled gCO₂/kWh)")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("2) Average day profile (this is the ‘aha’ chart)")
sim_day = sim.copy()
sim_day["hour"] = pd.to_datetime(sim_day["ts"]).dt.hour
avg = sim_day.groupby("hour")[["sncf_mwh", "sncf_mwh_opt"]].mean().reset_index()
fig3 = px.line(avg, x="hour", y=["sncf_mwh", "sncf_mwh_opt"], title="Average day load shape: baseline vs shifted")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("3) Are we consuming electricity when it’s cleaner?")
# quartiles of CI
tmp = sim[[ci_col, "sncf_mwh", "sncf_mwh_opt"]].copy()
tmp["ci_bin"] = pd.qcut(tmp[ci_col].rank(method="first"), 4, labels=["Cleanest 25%", "Q2", "Q3", "Dirtiest 25%"])

dist = tmp.groupby("ci_bin")[["sncf_mwh", "sncf_mwh_opt"]].sum().reset_index()
dist = dist.melt(id_vars="ci_bin", var_name="scenario", value_name="mwh")
fig4 = px.bar(dist, x="ci_bin", y="mwh", color="scenario", barmode="group",
              title="Total SNCF MWh consumed by grid cleanliness quartile")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("4) From → To shifting by hour (summary)")
mat = sim.attrs.get("to_hour_matrix", None)
if mat is not None:
    mat_df = pd.DataFrame(mat, index=[f"from {h:02d}" for h in range(24)], columns=[f"to {h:02d}" for h in range(24)])
    # show only meaningful values
    st.dataframe(mat_df.round(1), use_container_width=True)
else:
    st.info("Shift matrix not available.")

st.subheader("Export")
export_cols = ["ts", "sncf_mwh", "sncf_mwh_opt", ci_col, "renewable_share", "Consommation"]
export_cols = [c for c in export_cols if c in sim.columns]
st.download_button(
    "Download full simulation CSV",
    sim[export_cols].to_csv(index=False).encode("utf-8"),
    file_name="simulation_results.csv",
    mime="text/csv",
)
