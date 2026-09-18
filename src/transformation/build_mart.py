from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import pandas as pd

from src.ingestion.catalog import load_source_catalog
from src.validation.contracts import assert_unique_market_month

JOIN_KEYS = ["market_id", "month"]
DESCRIPTIVE_COLUMNS = ["market_name", "region_type", "state_name", "size_rank"]


def build_market_month_mart(
    interim_dir: str | Path = "data/interim",
    output_path: str | Path = "data/processed/mart_market_monthly.parquet",
    catalog_path: str | Path = "config/data_sources.yaml",
) -> pd.DataFrame:
    catalog = load_source_catalog(catalog_path)
    interim = Path(interim_dir)

    frames: list[pd.DataFrame] = []
    for idx, (key, source) in enumerate(catalog["sources"].items()):
        frame = pd.read_parquet(interim / f"{key}.parquet")
        metric = source["metric"]

        keep = JOIN_KEYS + [metric]
        if idx == 0:
            keep += [c for c in DESCRIPTIVE_COLUMNS if c in frame.columns]
        frames.append(frame[keep])

    mart = reduce(
        lambda left, right: left.merge(
            right,
            on=JOIN_KEYS,
            how="outer",
            validate="one_to_one",
        ),
        frames,
    )
    # Zillow "Metro" files also include a national United States aggregate row.
    # The canonical modeling mart is metro-only; keep national benchmarks separate.
    if "region_type" in mart.columns:
        mart = mart.loc[mart["region_type"].eq("msa")].copy()

    mart = mart.sort_values(JOIN_KEYS).reset_index(drop=True)
    assert_unique_market_month(mart)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(target, index=False)
    return mart


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical market-month mart.")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument(
        "--output",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--catalog", default="config/data_sources.yaml")
    args = parser.parse_args()

    mart = build_market_month_mart(args.interim_dir, args.output, args.catalog)
    print(
        f"mart_market_monthly: {len(mart):,} rows, "
        f"{mart['market_id'].nunique():,} markets, "
        f"{mart['month'].min().date()} to {mart['month'].max().date()}"
    )


if __name__ == "__main__":
    main()
