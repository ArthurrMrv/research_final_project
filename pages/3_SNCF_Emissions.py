import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_sncf_emissions

st.title("🚆 SNCF Emissions Explorer")

dfs = load_sncf_emissions()
dataset_name = st.selectbox("Choose SNCF dataset", list(dfs.keys()))
df = dfs[dataset_name].copy()

st.write("Shape:", df.shape)
st.dataframe(df.head(50), use_container_width=True)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols]

if num_cols:
    value_col = st.selectbox("Numeric column to plot", num_cols)
    if cat_cols:
        group_col = st.selectbox("Group by", cat_cols)
        agg = df.groupby(group_col, dropna=False)[value_col].sum().reset_index().sort_values(value_col, ascending=False).head(30)
        fig = px.bar(agg, x=group_col, y=value_col, title=f"{value_col} by {group_col}")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns detected in this dataset.")
