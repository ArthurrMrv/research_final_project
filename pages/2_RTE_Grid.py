import streamlit as st
import plotly.express as px
import pandas as pd

from data_loader import load_rte_production, load_rte_consumption
from features_rte import build_hourly_prod_timeseries, build_15min_demand_timeseries, compute_renewable_share, join_prod_demand, surplus_proxy

st.title("⚡ RTE Grid Explorer")

df_prod = compute_renewable_share(build_hourly_prod_timeseries(load_rte_production()))
df_dem = build_15min_demand_timeseries(load_rte_consumption())
df_join = surplus_proxy(join_prod_demand(df_prod, df_dem), top_green_pct=0.15)

tab1, tab2 = st.tabs(["Production mix", "Demand + surplus proxy"])

with tab1:
    st.subheader("Hourly renewable share (from production mix)")
    fig = px.line(df_prod, x="ts", y="renewable_share", title="Renewable share (hourly)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top green hours")
    top = df_prod.sort_values("renewable_share", ascending=False).head(20)[["ts", "renewable_share", "Total"]]
    st.dataframe(top, use_container_width=True)

with tab2:
    st.subheader("15-min demand with green-window flag")
    fig = px.line(df_join, x="ts", y="Consommation", title="RTE demand (15-min)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Surplus proxy score (0..2)")
    fig2 = px.line(df_join, x="ts", y="surplus_score", title="Surplus proxy score")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Best upcoming windows (by score, then renewable share)")
    best = df_join.sort_values(["surplus_score", "renewable_share"], ascending=False).head(50)[["ts", "surplus_score", "renewable_share", "Consommation"]]
    st.dataframe(best, use_container_width=True)
