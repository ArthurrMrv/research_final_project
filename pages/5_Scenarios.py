import streamlit as st
import pandas as pd

from data_loader import load_rte_production, load_rte_consumption
from features_rte import build_hourly_prod_timeseries, build_15min_demand_timeseries, compute_renewable_share, join_prod_demand
from simulator import make_synthetic_sncf_load, shift_load_greedy, score_from_renewables, emissions_proxy

st.title("📊 Scenario Comparison")

df_prod = compute_renewable_share(build_hourly_prod_timeseries(load_rte_production()))
df_dem = build_15min_demand_timeseries(load_rte_consumption())
df = join_prod_demand(df_prod, df_dem)
df = df.sort_values("ts").reset_index(drop=True)

df["green_score"] = score_from_renewables(df)

annual_mwh = st.number_input("Assumed annual SNCF electricity (MWh)", min_value=1000.0, value=1_000_000.0, step=50_000.0)
df["sncf_mwh"] = make_synthetic_sncf_load(pd.DatetimeIndex(df["ts"]), annual_mwh=annual_mwh).values

presets = [
    ("Low-risk", 0.05, 1),
    ("Medium", 0.15, 4),
    ("Aggressive", 0.30, 12),
]

rows = []
for name, share, hours in presets:
    sim = shift_load_greedy(
        df, "sncf_mwh", "green_score",
        shiftable_share=share,
        window_steps=int((hours*60)/15),
        freq_minutes=15
    )
    base_em = emissions_proxy(sim["sncf_mwh"], sim["fossil_share"])
    opt_em = emissions_proxy(sim["sncf_mwh_opt"], sim["fossil_share"])
    rows.append({
        "Scenario": name,
        "Shiftable share": share,
        "Window (h)": hours,
        "Baseline CO₂ (tCO₂, proxy)": round(base_em, 0),
        "Optimized CO₂ (tCO₂, proxy)": round(opt_em, 0),
        "Reduction (tCO₂)": round(base_em - opt_em, 0),
        "Reduction (%)": round((base_em - opt_em) / base_em * 100, 2),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.caption("All values are proxy estimates unless you provide real SNCF load & real carbon intensity / prices.")
