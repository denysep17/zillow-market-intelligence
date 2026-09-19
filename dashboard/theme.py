from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
  --z-blue: #006AFF;
  --ink: #101828;
  --muted: #667085;
  --line: #EAECF0;
  --panel: #FFFFFF;
  --page: #F7F8FA;
}
.stApp {
  background: var(--page);
}
.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 1480px;
}
[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid var(--line);
}
h1, h2, h3 {
  color: var(--ink);
  letter-spacing: -0.025em;
}
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px 18px;
}
[data-testid="stMetricLabel"] {
  color: var(--muted);
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}
.product-kicker {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #667085;
  font-weight: 700;
}
.product-title {
  font-size: 2.1rem;
  font-weight: 760;
  line-height: 1.1;
  color: #101828;
  margin: 0.25rem 0 0.4rem 0;
}
.product-subtitle {
  color: #667085;
  font-size: 1rem;
  margin-bottom: 0.8rem;
}
.status-pill {
  display: inline-block;
  border: 1px solid #B2CCFF;
  background: #EFF4FF;
  color: #004EEB;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.76rem;
  font-weight: 700;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(
    kicker: str,
    title: str,
    subtitle: str,
    status: str | None = None,
) -> None:
    status_html = (
        f'<span class="status-pill">{status}</span>' if status else ""
    )
    st.markdown(
        f"""
        <div class="product-kicker">{kicker}</div>
        <div class="product-title">{title}</div>
        <div class="product-subtitle">{subtitle}</div>
        {status_html}
        """,
        unsafe_allow_html=True,
    )
