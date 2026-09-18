from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_ID_COLUMNS = [
    "RegionID",
    "SizeRank",
    "RegionName",
    "RegionType",
    "StateName",
]


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that parse cleanly as YYYY-MM-DD dates."""
    out: list[str] = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(col, format="%Y-%m-%d", errors="raise")
        except (ValueError, TypeError):
            continue
        if not pd.isna(parsed):
            out.append(col)
    return out


def normalize_zillow_wide(
    df: pd.DataFrame,
    metric: str,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert a Zillow wide metro time series to one row per market-month."""
    date_columns = detect_date_columns(df)
    if not date_columns:
        raise ValueError("No YYYY-MM-DD date columns were detected")

    preferred_ids = id_columns or DEFAULT_ID_COLUMNS
    ids = [col for col in preferred_ids if col in df.columns]
    if "RegionID" not in ids or "RegionName" not in ids:
        raise ValueError("Expected Zillow columns RegionID and RegionName")

    long_df = df.melt(
        id_vars=ids,
        value_vars=date_columns,
        var_name="month",
        value_name=metric,
    )
    long_df["month"] = pd.to_datetime(long_df["month"], format="%Y-%m-%d")
    long_df = long_df.rename(
        columns={
            "RegionID": "market_id",
            "SizeRank": "size_rank",
            "RegionName": "market_name",
            "RegionType": "region_type",
            "StateName": "state_name",
        }
    )

    return long_df.sort_values(["market_id", "month"]).reset_index(drop=True)


def normalize_csv(path: str | Path, metric: str) -> pd.DataFrame:
    return normalize_zillow_wide(pd.read_csv(path), metric=metric)
