from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_FEATURES = [
    "zhvi_growth_3m",
    "inventory_growth_3m",
    "price_cut_change_3m",
    "days_pending_change_3m",
    "market_heat_change_3m",
]


def _within_month_percentile(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    return df.groupby("month")[column].rank(
        pct=True,
        method="average",
    )


def add_regime_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in REGIME_FEATURES:
        out[f"{feature}_pctile_regime"] = _within_month_percentile(
            out,
            feature,
        )
    return out


def assign_rule_regime(df: pd.DataFrame) -> pd.Series:
    """Assign transparent market-state labels from same-month percentiles.

    States are designed for interpretability, not as ground truth.
    """
    price = df["zhvi_growth_3m_pctile_regime"]
    inventory = df["inventory_growth_3m_pctile_regime"]
    cuts = df["price_cut_change_3m_pctile_regime"]
    pending = df["days_pending_change_3m_pctile_regime"]
    heat = df["market_heat_change_3m_pctile_regime"]

    accelerating = (
        price.ge(0.70)
        & inventory.le(0.45)
        & cuts.le(0.55)
        & heat.ge(0.45)
    )
    cooling = (
        price.le(0.35)
        & inventory.ge(0.55)
        & cuts.ge(0.55)
    )
    tightening = (
        inventory.le(0.35)
        & pending.le(0.45)
        & heat.ge(0.60)
        & ~accelerating
    )
    loosening = (
        inventory.ge(0.65)
        & pending.ge(0.55)
        & heat.le(0.45)
        & ~cooling
    )

    labels = np.select(
        [accelerating, cooling, tightening, loosening],
        ["accelerating", "cooling", "tightening", "loosening"],
        default="balanced",
    )
    return pd.Series(labels, index=df.index, dtype="object")


def build_rule_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = add_regime_percentiles(df)
    out["rule_regime"] = assign_rule_regime(out)
    return out
