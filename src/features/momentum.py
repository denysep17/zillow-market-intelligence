from __future__ import annotations

import pandas as pd


def add_lagged_growth_features(
    df: pd.DataFrame,
    value_col: str,
    windows: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    out = df.sort_values(["market_id", "month"]).copy()
    grouped = out.groupby("market_id")[value_col]

    for window in windows:
        lagged = grouped.shift(window)
        out[f"{value_col}_growth_{window}m"] = out[value_col] / lagged - 1.0

    return out
