import streamlit as st

st.set_page_config(page_title="SNCF x RTE – Time-shifting", layout="wide")

st.title("SNCF x RTE – Time-shifting to reduce emissions & energy costs")

st.markdown("""
Use the sidebar to navigate:
- **Executive**: headline impact and KPIs  
- **RTE Grid**: when the grid is cleanest / surplus proxy  
- **Simulator**: what-if shifting model  
- **SNCF Emissions**: baseline footprint breakdown  
- **Scenarios**: compare strategies
""")

