import streamlit as st

st.set_page_config(
    page_title="Zillow Market Intelligence",
    page_icon="🏠",
    layout="wide",
)

st.title("Zillow Market Intelligence")
st.caption("Early-warning housing market monitoring, forecasting, and regime detection")

st.info(
    "Phase 0 scaffold: data sources and analytical contracts are being locked before "
    "model development."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Markets monitored", "—")
col2.metric("Regime transitions", "—")
col3.metric("High-confidence alerts", "—")
col4.metric("Forecast MAE", "—")

st.subheader("Product workflow")
st.write("Monitor → Detect → Diagnose → Forecast → Prioritize → Act → Measure")

st.subheader("Planned views")
st.markdown(
    """
    - Executive Market Monitor
    - Market Explorer
    - Metro Deep Dive
    - Drivers and explanations
    - Alerts
    - Model Health
    """
)
