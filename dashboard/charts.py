from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def market_matrix(scored: pd.DataFrame) -> go.Figure:
    frame = scored.dropna(
        subset=["inventory_growth_12m", "forecast_3m", "cooling_risk"]
    ).copy()

    fig = px.scatter(
        frame,
        x="inventory_growth_12m",
        y="forecast_3m",
        size="cooling_risk",
        hover_name="market_name",
        hover_data={
            "state_name": True,
            "rule_regime": True,
            "cooling_risk": ":.1%",
            "forecast_3m": ":.1%",
            "inventory_growth_12m": ":.1%",
            "price_cut_share": ":.1%",
            "size_rank": True,
        },
        labels={
            "inventory_growth_12m": "Inventory growth (12M)",
            "forecast_3m": "Forecast ZHVI growth (3M)",
            "cooling_risk": "Cooling-transition risk",
        },
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#D0D5DD")
    fig.add_hline(y=0, line_dash="dot", line_color="#D0D5DD")
    fig.update_traces(marker={"opacity": 0.65, "line": {"width": 0}})
    fig.update_layout(
        height=540,
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".1%")
    return fig


def regime_distribution(scored: pd.DataFrame) -> go.Figure:
    counts = (
        scored["rule_regime"]
        .fillna("unknown")
        .value_counts(normalize=True)
        .rename_axis("regime")
        .reset_index(name="share")
    )
    fig = px.bar(
        counts,
        x="regime",
        y="share",
        text_auto=".1%",
        labels={"regime": "", "share": "Share of metros"},
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def cooling_risk_histogram(scored: pd.DataFrame) -> go.Figure:
    frame = scored.dropna(subset=["cooling_risk"]).copy()
    fig = px.histogram(
        frame,
        x="cooling_risk",
        nbins=24,
        labels={"cooling_risk": "Cooling-transition probability"},
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        bargap=0.06,
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def metro_zhvi_history(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["month"],
            y=history["zhvi"],
            mode="lines",
            name="ZHVI",
            line={"width": 2.5},
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
        yaxis_title="ZHVI",
        xaxis_title="",
    )
    return fig


def metro_market_signals(history: pd.DataFrame) -> go.Figure:
    frame = history.copy()
    if frame["inventory"].notna().any():
        inventory_base = frame["inventory"].dropna().iloc[0]
        frame["inventory_index"] = frame["inventory"] / inventory_base * 100
    else:
        frame["inventory_index"] = float("nan")

    if frame["market_heat"].notna().any():
        heat_base = frame["market_heat"].dropna().iloc[0]
        frame["market_heat_indexed"] = frame["market_heat"] / heat_base * 100
    else:
        frame["market_heat_indexed"] = float("nan")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["inventory_index"],
            mode="lines",
            name="Inventory index",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["market_heat_indexed"],
            mode="lines",
            name="Market Heat index",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
        yaxis_title="Indexed to first available = 100",
        xaxis_title="",
    )
    return fig


def forecast_interval_chart(row: pd.Series) -> go.Figure:
    center = float(row["forecast_3m"])
    lower = float(row["forecast_lower"])
    upper = float(row["forecast_upper"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=["3M forecast", "3M forecast"],
            mode="lines",
            line={"width": 10},
            name="Empirical 90% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[center],
            y=["3M forecast"],
            mode="markers",
            marker={"size": 15},
            name="Point forecast",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#98A2B3")
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_tickformat=".1%",
        xaxis_title="Forward ZHVI growth",
        yaxis_title="",
        legend_orientation="h",
        legend_y=-0.35,
    )
    return fig


def model_health_mae() -> go.Figure:
    frame = pd.DataFrame(
        {
            "model": [
                "Elastic Net",
                "Gradient boosting",
                "Price-only Ridge",
                "No change",
                "Trailing momentum",
                "Seasonal naive",
            ],
            "mae_pp": [0.871, 0.907, 1.010, 1.188, 1.307, 1.793],
        }
    )
    fig = px.bar(
        frame,
        x="mae_pp",
        y="model",
        orientation="h",
        text_auto=".3f",
        labels={"mae_pp": "MAE (percentage points)", "model": ""},
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig
