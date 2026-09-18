from __future__ import annotations

import pandas as pd


def add_forward_growth_target(
    df: pd.DataFrame,
    value_col: str = "zhvi",
    horizon: int = 3,
) -> pd.DataFrame:
    """Create a forward growth target within each market.

    Assumes one row per market-month and ascending monthly order after sorting.
    """
    out = df.sort_values(["market_id", "month"]).copy()
    future = out.groupby("market_id")[value_col].shift(-horizon)
    out[f"{value_col}_forward_growth_{horizon}m"] = future / out[value_col] - 1.0
    return out
