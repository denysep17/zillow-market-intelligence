from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CORE_METRICS = [
    "zhvi",
    "zori",
    "inventory",
    "new_listings",
    "price_cut_share",
    "days_to_pending",
    "market_heat",
]


def longest_missing_run(values: pd.Series) -> int:
    """Return the longest consecutive run of missing observations."""
    is_missing = values.isna()
    if not is_missing.any():
        return 0
    groups = (is_missing != is_missing.shift()).cumsum()
    runs = is_missing.groupby(groups).sum()
    return int(runs.max())


def metric_audit(df: pd.DataFrame, metric: str) -> dict[str, object]:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column: {metric}")

    observed = df.loc[df[metric].notna(), ["market_id", "month", metric]].copy()
    observed["month"] = pd.to_datetime(observed["month"])

    if observed.empty:
        return {
            "metric": metric,
            "observations": 0,
            "markets": 0,
            "first_month": None,
            "last_month": None,
            "null_rate": 1.0,
            "duplicate_market_months": 0,
            "min": None,
            "median": None,
            "max": None,
        }

    duplicates = observed.duplicated(["market_id", "month"]).sum()

    return {
        "metric": metric,
        "observations": int(len(observed)),
        "markets": int(observed["market_id"].nunique()),
        "first_month": observed["month"].min().date().isoformat(),
        "last_month": observed["month"].max().date().isoformat(),
        "null_rate": float(df[metric].isna().mean()),
        "duplicate_market_months": int(duplicates),
        "min": float(observed[metric].min()),
        "median": float(observed[metric].median()),
        "max": float(observed[metric].max()),
    }


def coverage_by_year(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    data = df.copy()
    data["year"] = pd.to_datetime(data["month"]).dt.year
    rows: list[dict[str, object]] = []

    for year, frame in data.groupby("year"):
        denominator = max(
            frame[["market_id", "month"]].drop_duplicates().shape[0],
            1,
        )
        for metric in metrics:
            observed = frame.loc[frame[metric].notna()]
            numerator = observed[["market_id", "month"]].drop_duplicates().shape[0]
            rows.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "observations": int(len(observed)),
                    "markets": int(observed["market_id"].nunique()),
                    "coverage_rate": float(numerator / denominator),
                }
            )

    return pd.DataFrame(rows)


def build_audit(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs/reports",
) -> None:
    mart = pd.read_parquet(mart_path)
    metrics = [metric for metric in CORE_METRICS if metric in mart.columns]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(mart)),
        "markets": int(mart["market_id"].nunique()),
        "first_month": pd.to_datetime(mart["month"]).min().date().isoformat(),
        "last_month": pd.to_datetime(mart["month"]).max().date().isoformat(),
        "metrics": [metric_audit(mart, metric) for metric in metrics],
    }

    (output / "data_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    coverage_by_year(mart, metrics).to_csv(
        output / "coverage_by_year.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical market-month coverage.")
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()
    build_audit(args.mart, args.output_dir)


if __name__ == "__main__":
    main()
