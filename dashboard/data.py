from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.model_table import FULL_FEATURES, TARGET, build_model_table
from src.forecasting.models import elastic_net_model
from src.regimes.early_warning import (
    ALERT_FEATURES,
    _precision_target_threshold,
    add_cooling_entry_target,
)
from src.regimes.early_warning import _model as alert_model
from src.regimes.rules import build_rule_regimes


@dataclass(frozen=True)
class DashboardBundle:
    mart: pd.DataFrame
    model_table: pd.DataFrame
    scored: pd.DataFrame
    latest_month: pd.Timestamp
    forecast_training_end: pd.Timestamp
    alert_training_end: pd.Timestamp


def load_mart(
    path: str | Path = "data/processed/mart_market_monthly.parquet",
) -> pd.DataFrame:
    mart = pd.read_parquet(path)
    mart["month"] = pd.to_datetime(mart["month"])
    return mart.sort_values(["market_id", "month"]).reset_index(drop=True)


def _fit_latest_forecast(table: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    latest_month = table["month"].max()
    labeled = table.loc[table[TARGET].notna()].copy()
    training_end = labeled["month"].max()

    model = elastic_net_model()
    model.fit(labeled[FULL_FEATURES], labeled[TARGET])

    current = table.loc[table["month"].eq(latest_month)].copy()
    current["forecast_3m"] = model.predict(current[FULL_FEATURES])

    residual_prediction = model.predict(labeled[FULL_FEATURES])
    residual = np.abs(labeled[TARGET].to_numpy() - residual_prediction)
    labeled = labeled.assign(abs_residual=residual)

    cohort_radius = (
        labeled.groupby("size_cohort")["abs_residual"]
        .quantile(0.90)
        .to_dict()
    )
    global_radius = float(labeled["abs_residual"].quantile(0.90))

    current["forecast_radius"] = (
        current["size_cohort"].map(cohort_radius).fillna(global_radius)
    )
    current["forecast_lower"] = current["forecast_3m"] - current["forecast_radius"]
    current["forecast_upper"] = current["forecast_3m"] + current["forecast_radius"]
    return training_end, current


def _fit_latest_transition_risk(
    table: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    regime_data = build_rule_regimes(table)
    targeted = add_cooling_entry_target(regime_data)
    training = targeted.loc[
        targeted["eligible_alert_row"]
        & targeted["cooling_entry_next_3m"].notna()
    ].copy()

    training_end = training["month"].max()
    latest_month = targeted["month"].max()
    current = targeted.loc[targeted["month"].eq(latest_month)].copy()

    y = training["cooling_entry_next_3m"].astype(int)
    model = alert_model()
    model.fit(training[ALERT_FEATURES], y)

    train_probability = model.predict_proba(training[ALERT_FEATURES])[:, 1]
    threshold = _precision_target_threshold(
        y,
        train_probability,
        target_precision=0.50,
    )

    current["cooling_risk"] = model.predict_proba(
        current[ALERT_FEATURES]
    )[:, 1]
    current["alert_threshold"] = threshold
    current["high_confidence_alert"] = (
        current["cooling_risk"].ge(threshold)
        & current["eligible_alert_row"]
    )
    return training_end, current


def build_dashboard_bundle(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
) -> DashboardBundle:
    mart = load_mart(mart_path)
    table = build_model_table(mart)
    table = build_rule_regimes(table)

    forecast_training_end, forecast = _fit_latest_forecast(table)
    alert_training_end, alerts = _fit_latest_transition_risk(table)

    alert_cols = [
        "market_id",
        "month",
        "rule_regime",
        "cooling_risk",
        "alert_threshold",
        "high_confidence_alert",
    ]
    forecast_cols = [
        "market_id",
        "month",
        "market_name",
        "state_name",
        "size_rank",
        "size_cohort",
        "zhvi",
        "zhvi_growth_3m",
        "zhvi_growth_12m",
        "zori_growth_12m",
        "inventory_growth_12m",
        "new_listings_growth_12m",
        "price_cut_share",
        "days_to_pending",
        "market_heat",
        "forecast_3m",
        "forecast_lower",
        "forecast_upper",
        "forecast_radius",
    ]

    scored = forecast[forecast_cols].merge(
        alerts[alert_cols],
        on=["market_id", "month"],
        how="left",
        validate="one_to_one",
    )

    scored["attention_score"] = (
        scored["cooling_risk"].fillna(0) * 0.55
        + scored["inventory_growth_12m"].clip(-0.5, 0.5).fillna(0).abs() * 0.15
        + scored["price_cut_share"].fillna(0) * 0.15
        + scored["forecast_radius"].fillna(0) * 3.0 * 0.15
    )
    scored["reliability"] = np.select(
        [
            scored["forecast_radius"].le(
                scored["forecast_radius"].quantile(0.33)
            ),
            scored["forecast_radius"].le(
                scored["forecast_radius"].quantile(0.67)
            ),
        ],
        ["High", "Medium"],
        default="Lower",
    )

    return DashboardBundle(
        mart=mart,
        model_table=table,
        scored=scored.sort_values("attention_score", ascending=False),
        latest_month=pd.Timestamp(scored["month"].max()),
        forecast_training_end=pd.Timestamp(forecast_training_end),
        alert_training_end=pd.Timestamp(alert_training_end),
    )


def metro_history(
    table: pd.DataFrame,
    market_id: int,
) -> pd.DataFrame:
    columns = [
        "market_id",
        "market_name",
        "state_name",
        "month",
        "zhvi",
        "zori",
        "inventory",
        "new_listings",
        "price_cut_share",
        "days_to_pending",
        "market_heat",
        "zhvi_growth_3m",
        "zhvi_growth_12m",
        "inventory_growth_12m",
        "rule_regime",
    ]
    return (
        table.loc[table["market_id"].eq(market_id), columns]
        .sort_values("month")
        .reset_index(drop=True)
    )
