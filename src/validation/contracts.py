from __future__ import annotations

import pandas as pd


def assert_unique_market_month(df: pd.DataFrame) -> None:
    required = {"market_id", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicates = df.duplicated(["market_id", "month"]).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate market-month rows")


def assert_positive_metric(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")
    if (df[column].dropna() <= 0).any():
        raise ValueError(f"Column {column} contains non-positive values")
