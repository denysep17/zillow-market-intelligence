from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_START = pd.Timestamp("2019-03-31")
TARGET = "zhvi_forward_growth_3m"

PRICE_FEATURES = [
    "zhvi_growth_1m",
    "zhvi_growth_3m",
    "zhvi_growth_6m",
    "zhvi_growth_12m",
    "zhvi_volatility_12m",
    "month_sin",
    "month_cos",
    "log_size_rank",
]

FULL_FEATURES = PRICE_FEATURES + [
    "zori_growth_3m",
    "zori_growth_12m",
    "inventory_growth_3m",
    "inventory_growth_12m",
    "new_listings_growth_3m",
    "new_listings_growth_12m",
    "price_cut_change_3m",
    "price_cut_change_12m",
    "days_pending_change_3m",
    "days_pending_change_12m",
    "market_heat_change_3m",
    "zori_growth_3m_pctile",
    "inventory_growth_3m_pctile",
    "price_cut_change_3m_pctile",
    "days_pending_change_3m_pctile",
    "market_heat_change_3m_pctile",
]


def _pct_change(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    lagged = df.groupby("market_id")[column].shift(periods)
    return df[column] / lagged - 1.0


def _point_change(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    lagged = df.groupby("market_id")[column].shift(periods)
    return df[column] - lagged


def _forward_growth(df: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    future = df.groupby("market_id")[column].shift(-horizon)
    return future / df[column] - 1.0


def _cross_sectional_pctile(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("month")[column].rank(pct=True, method="average")


def _size_cohort(size_rank: pd.Series) -> pd.Series:
    conditions = [
        size_rank.le(100),
        size_rank.between(101, 300),
        size_rank.gt(300),
    ]
    values = ["top_100", "rank_101_300", "rank_301_plus"]
    return pd.Series(
        np.select(conditions, values, default="unknown"),
        index=size_rank.index,
    )


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build features available at month t and a 3-month forward ZHVI target."""
    out = df.sort_values(["market_id", "month"]).copy()
    out["month"] = pd.to_datetime(out["month"])

    for periods in (1, 3, 6, 12):
        out[f"zhvi_growth_{periods}m"] = _pct_change(
            out,
            "zhvi",
            periods,
        )

    out["zori_growth_3m"] = _pct_change(out, "zori", 3)
    out["zori_growth_12m"] = _pct_change(out, "zori", 12)
    out["inventory_growth_3m"] = _pct_change(out, "inventory", 3)
    out["inventory_growth_12m"] = _pct_change(out, "inventory", 12)
    out["new_listings_growth_3m"] = _pct_change(out, "new_listings", 3)
    out["new_listings_growth_12m"] = _pct_change(out, "new_listings", 12)
    out["price_cut_change_3m"] = _point_change(out, "price_cut_share", 3)
    out["price_cut_change_12m"] = _point_change(out, "price_cut_share", 12)
    out["days_pending_change_3m"] = _point_change(
        out,
        "days_to_pending",
        3,
    )
    out["days_pending_change_12m"] = _point_change(
        out,
        "days_to_pending",
        12,
    )
    out["market_heat_change_3m"] = _point_change(out, "market_heat", 3)

    out[TARGET] = _forward_growth(out, "zhvi", 3)

    lag_12 = out.groupby("market_id")["zhvi"].shift(12)
    lag_9 = out.groupby("market_id")["zhvi"].shift(9)
    out["seasonal_naive_3m"] = lag_9 / lag_12 - 1.0

    monthly_return = out["zhvi_growth_1m"]
    out["zhvi_volatility_12m"] = (
        monthly_return.groupby(out["market_id"])
        .rolling(12, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    month_number = out["month"].dt.month.astype(float)
    out["month_sin"] = np.sin(2.0 * np.pi * month_number / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month_number / 12.0)
    out["log_size_rank"] = np.log1p(out["size_rank"].astype(float))
    out["size_cohort"] = _size_cohort(out["size_rank"])

    relative_features = [
        "zori_growth_3m",
        "inventory_growth_3m",
        "price_cut_change_3m",
        "days_pending_change_3m",
        "market_heat_change_3m",
    ]
    for feature in relative_features:
        out[f"{feature}_pctile"] = _cross_sectional_pctile(out, feature)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[out["month"].ge(MODEL_START)].copy()
    return out
