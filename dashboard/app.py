from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.charts import (
    PLOT_CONFIG,
    cooling_risk_histogram,
    driver_contributions,
    forecast_interval_chart,
    market_matrix,
    metro_market_signals,
    metro_zhvi_history,
    model_health_mae,
    regime_distribution,
)
from dashboard.data import (
    build_dashboard_bundle,
    load_dashboard_bundle,
    metro_drivers,
    metro_history,
)
from dashboard.theme import apply_theme, page_header

st.set_page_config(
    page_title="Zillow Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


@st.cache_resource(show_spinner=False)
def get_bundle():
    snapshot = Path("data/processed/dashboard_bundle.pkl")
    if snapshot.exists():
        return load_dashboard_bundle(snapshot)
    return build_dashboard_bundle()


def require_data():
    mart = Path("data/processed/mart_market_monthly.parquet")
    if not mart.exists():
        st.error(
            "The canonical mart is missing. Run the data pipeline first: "
            "python -m src.ingestion.download; "
            "python -m src.transformation.build_interim; "
            "python -m src.transformation.build_mart."
        )
        st.stop()


require_data()

with st.spinner("Loading market intelligence..."):
    bundle = get_bundle()

scored = bundle.scored.copy()
table = bundle.model_table

st.sidebar.markdown("## Zillow Market Intelligence")
st.sidebar.caption("Decision-support prototype")
view = st.sidebar.radio(
    "Navigate",
    [
        "Executive Monitor",
        "Market Explorer",
        "Metro Deep Dive",
        "Alerts",
        "Model Health",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(f"Latest market data: {bundle.latest_month:%b %Y}")
st.sidebar.caption(
    f"Forecast labels through: {bundle.forecast_training_end:%b %Y}"
)

if view == "Executive Monitor":
    page_header(
        "EXECUTIVE MARKET MONITOR",
        "Where is the housing market changing?",
        (
            "Current regime mix, forward momentum, uncertainty, and transition "
            "risk across U.S. metros."
        ),
        status=f"Data through {bundle.latest_month:%B %Y}",
    )

    high_alerts = int(scored["high_confidence_alert"].fillna(False).sum())
    cooling_share = float(scored["rule_regime"].eq("cooling").mean())
    median_forecast = float(scored["forecast_3m"].median())
    median_risk = float(scored["cooling_risk"].median())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metros monitored", f"{scored['market_id'].nunique():,}")
    c2.metric("Cooling regime", f"{cooling_share:.1%}")
    c3.metric("High-confidence alerts", f"{high_alerts:,}")
    c4.metric("Median 3M forecast", f"{median_forecast:.1%}")

    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Market state matrix")
        st.caption(
            "Bubble size reflects cooling-transition risk. Hover for metro detail."
        )
        st.plotly_chart(
            market_matrix(scored),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        st.subheader("Current regime mix")
        st.plotly_chart(
            regime_distribution(scored),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        st.metric(
            "Median cooling-transition risk",
            f"{median_risk:.1%}",
        )

    st.subheader("Priority metros")
    priority = scored[
        [
            "market_name",
            "state_name",
            "rule_regime",
            "forecast_3m",
            "forecast_lower",
            "forecast_upper",
            "cooling_risk",
            "reliability",
            "high_confidence_alert",
        ]
    ].head(15).copy()
    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_name": "Metro",
            "state_name": "State",
            "rule_regime": "Regime",
            "forecast_3m": st.column_config.NumberColumn(
                "3M forecast",
                format="%.1f%%",
            ),
            "forecast_lower": st.column_config.NumberColumn(
                "Lower",
                format="%.1f%%",
            ),
            "forecast_upper": st.column_config.NumberColumn(
                "Upper",
                format="%.1f%%",
            ),
            "cooling_risk": st.column_config.ProgressColumn(
                "Cooling risk",
                min_value=0,
                max_value=1,
                format="%.0f%%",
            ),
            "reliability": "Reliability",
            "high_confidence_alert": "Alert",
        },
    )

elif view == "Market Explorer":
    page_header(
        "MARKET EXPLORER",
        "Compare metros across supply, momentum, and transition risk.",
        "Use product-style filters to narrow the market universe.",
    )

    states = sorted(scored["state_name"].dropna().unique().tolist())
    regimes = sorted(scored["rule_regime"].dropna().unique().tolist())

    f1, f2, f3 = st.columns([1, 1, 1])
    selected_states = f1.multiselect("State", states)
    selected_regimes = f2.multiselect("Regime", regimes)
    risk_floor = f3.slider(
        "Minimum cooling risk",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.0f%%",
    )

    filtered = scored.copy()
    if selected_states:
        filtered = filtered.loc[filtered["state_name"].isin(selected_states)]
    if selected_regimes:
        filtered = filtered.loc[
            filtered["rule_regime"].isin(selected_regimes)
        ]
    filtered = filtered.loc[filtered["cooling_risk"].ge(risk_floor)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Metros in view", f"{len(filtered):,}")
    c2.metric(
        "Median 3M forecast",
        f"{filtered['forecast_3m'].median():.1%}" if len(filtered) else "—",
    )
    c3.metric(
        "Median cooling risk",
        f"{filtered['cooling_risk'].median():.1%}" if len(filtered) else "—",
    )

    st.plotly_chart(
        market_matrix(filtered),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.dataframe(
        filtered[
            [
                "market_name",
                "state_name",
                "size_rank",
                "rule_regime",
                "zhvi_growth_12m",
                "inventory_growth_12m",
                "price_cut_share",
                "forecast_3m",
                "cooling_risk",
                "reliability",
            ]
        ].sort_values("cooling_risk", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_name": "Metro",
            "state_name": "State",
            "size_rank": "Size rank",
            "rule_regime": "Regime",
            "zhvi_growth_12m": st.column_config.NumberColumn(
                "ZHVI 12M",
                format="%.1f%%",
            ),
            "inventory_growth_12m": st.column_config.NumberColumn(
                "Inventory 12M",
                format="%.1f%%",
            ),
            "price_cut_share": st.column_config.NumberColumn(
                "Price-cut share",
                format="%.1f%%",
            ),
            "forecast_3m": st.column_config.NumberColumn(
                "3M forecast",
                format="%.1f%%",
            ),
            "cooling_risk": st.column_config.ProgressColumn(
                "Cooling risk",
                min_value=0,
                max_value=1,
                format="%.0f%%",
            ),
            "reliability": "Reliability",
        },
    )

elif view == "Metro Deep Dive":
    page_header(
        "METRO DEEP DIVE",
        "Diagnose one market before acting on an alert.",
        (
            "Combine current regime, forecast, uncertainty, supply conditions, "
            "and historical context."
        ),
    )

    choices = (
        scored.sort_values(["size_rank", "market_name"])
        .assign(
            label=lambda x: x["market_name"]
            + " · "
            + x["state_name"].fillna("")
        )
    )
    label_to_id = dict(zip(choices["label"], choices["market_id"], strict=True))
    selected_label = st.selectbox("Metro", list(label_to_id))
    market_id = int(label_to_id[selected_label])

    current = scored.loc[scored["market_id"].eq(market_id)].iloc[0]
    history = metro_history(table, market_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current regime", str(current["rule_regime"]).title())
    c2.metric("3M forecast", f"{current['forecast_3m']:.1%}")
    c3.metric("Cooling risk", f"{current['cooling_risk']:.1%}")
    c4.metric("Reliability", current["reliability"])

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Home-value trajectory")
        st.plotly_chart(
            metro_zhvi_history(history),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        st.subheader("Forecast range")
        st.plotly_chart(
            forecast_interval_chart(current),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        st.caption(
            "Empirical interval width is calibrated by Zillow size-rank cohort."
        )

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Supply and market heat")
        st.plotly_chart(
            metro_market_signals(history),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        st.subheader("Current diagnostics")
        diag = pd.DataFrame(
            {
                "Metric": [
                    "ZHVI growth (12M)",
                    "Inventory growth (12M)",
                    "New listings growth (12M)",
                    "Price-cut share",
                    "Days to pending",
                    "Market Heat",
                ],
                "Value": [
                    f"{current['zhvi_growth_12m']:.1%}",
                    f"{current['inventory_growth_12m']:.1%}",
                    f"{current['new_listings_growth_12m']:.1%}",
                    f"{current['price_cut_share']:.1%}",
                    f"{current['days_to_pending']:.0f}",
                    f"{current['market_heat']:.0f}",
                ],
            }
        )
        st.dataframe(diag, hide_index=True, use_container_width=True)

    st.subheader("What is moving the point forecast?")
    st.caption(
        "Largest Elastic Net feature contributions for this metro. "
        "These are predictive contributions, not causal effects."
    )
    local_drivers = metro_drivers(bundle.drivers, market_id)
    st.plotly_chart(
        driver_contributions(local_drivers),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

elif view == "Alerts":
    page_header(
        "EARLY WARNING",
        "Which metros deserve analyst attention now?",
        (
            "Precision-oriented transition alerts prioritize markets at elevated "
            "risk of entering a cooling regime within 1–3 months."
        ),
    )

    c1, c2, c3 = st.columns(3)
    alert_frame = scored.loc[
        scored["high_confidence_alert"].fillna(False)
    ].copy()
    c1.metric("High-confidence alerts", f"{len(alert_frame):,}")
    c2.metric(
        "Alert threshold",
        (
            f"{scored['alert_threshold'].dropna().median():.1%}"
            if scored["alert_threshold"].notna().any()
            else "—"
        ),
    )
    c3.metric(
        "Median alerted risk",
        (
            f"{alert_frame['cooling_risk'].median():.1%}"
            if len(alert_frame)
            else "—"
        ),
    )

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Cooling-transition risk distribution")
        st.plotly_chart(
            cooling_risk_histogram(scored),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        st.subheader("Historical policy performance")
        st.metric("Precision-oriented precision", "51.4%")
        st.metric("Recall", "19.8%")
        st.metric("Median lead time", "1 month")
        st.caption(
            "Historical out-of-time metrics from v0.4. Threshold is selected "
            "from training data only within each evaluation fold."
        )

    st.subheader("Current alerts")
    st.dataframe(
        alert_frame[
            [
                "market_name",
                "state_name",
                "rule_regime",
                "forecast_3m",
                "cooling_risk",
                "inventory_growth_12m",
                "price_cut_share",
                "reliability",
            ]
        ].sort_values("cooling_risk", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_name": "Metro",
            "state_name": "State",
            "rule_regime": "Current regime",
            "forecast_3m": st.column_config.NumberColumn(
                "3M forecast",
                format="%.1f%%",
            ),
            "cooling_risk": st.column_config.ProgressColumn(
                "Cooling risk",
                min_value=0,
                max_value=1,
                format="%.0f%%",
            ),
            "inventory_growth_12m": st.column_config.NumberColumn(
                "Inventory 12M",
                format="%.1f%%",
            ),
            "price_cut_share": st.column_config.NumberColumn(
                "Price cuts",
                format="%.1f%%",
            ),
            "reliability": "Reliability",
        },
    )

elif view == "Model Health":
    page_header(
        "MODEL HEALTH",
        "Is the forecasting system still trustworthy?",
        (
            "Validation, uncertainty calibration, and known limitations are "
            "visible alongside model performance."
        ),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elastic Net MAE", "0.871 pp")
    c2.metric("Directional accuracy", "77.36%")
    c3.metric("Interval coverage", "90.53%")
    c4.metric("Alert ROC AUC", "0.804")

    st.subheader("Out-of-time forecast comparison")
    st.plotly_chart(
        model_health_mae(),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.subheader("Publication-lag sensitivity")
    lag_table = pd.DataFrame(
        {
            "Scenario": [
                "Contemporaneous",
                "Non-price lag 1M",
                "Non-price lag 2M",
                "All dynamic lag 1M",
            ],
            "MAE": ["0.871 pp", "0.928 pp", "0.989 pp", "1.143 pp"],
            "MAE degradation": ["0.0%", "+6.6%", "+13.5%", "+31.2%"],
            "Direction": ["77.4%", "76.5%", "76.2%", "70.4%"],
        }
    )
    st.dataframe(lag_table, hide_index=True, use_container_width=True)
    st.caption(
        "Stress scenarios test stale feature availability; they do not claim "
        "a specific Zillow publication SLA."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Where forecast error increases")
        st.markdown(
            """
            - High-volatility quartile: **1.276 pp MAE**
            - Cooling regime: **0.989 pp MAE**
            - Rank-301+ metros: **0.952 pp MAE**
            - Worst validation fold: **1.347 pp MAE**
            """
        )
    with right:
        st.subheader("Alert failure modes")
        st.markdown(
            """
            - Precision policy recall: **19.8%**
            - Realized cooling entries missed: **80.2%**
            - False-alert share: **48.6%**
            - Transition probability is a prioritization signal, not certainty
            """
        )

    st.subheader("Validation design")
    st.markdown(
        """
        - 8 non-overlapping rolling-origin folds
        - 3-month target-horizon purge
        - No random train/test split
        - Training-fold-only imputation
        - Feature-family ablation
        - Empirical uncertainty calibration by market-size cohort
        """
    )

    st.warning(
        "Reliability should be reduced for smaller metros, high-volatility "
        "contexts, cooling regimes, or materially delayed input features."
    )
