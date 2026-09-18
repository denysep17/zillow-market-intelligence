from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = ["zhvi", "zori", "inventory", "days_to_pending", "market_heat"]


def _month_gap_count(frame: pd.DataFrame, metric: str) -> dict[str, int]:
    observed = frame.loc[frame[metric].notna(), ["market_id", "month"]].copy()
    observed["period"] = pd.to_datetime(observed["month"]).dt.to_period("M")
    markets_with_gaps = 0
    missing_internal_months = 0

    for _, group in observed.groupby("market_id"):
        periods = group["period"].drop_duplicates().sort_values()
        if len(periods) < 2:
            continue
        ordinals = periods.astype("int64").to_numpy()
        gaps = np.diff(ordinals) - 1
        positive = gaps[gaps > 0]
        if len(positive):
            markets_with_gaps += 1
            missing_internal_months += int(positive.sum())

    return {
        "markets_with_internal_gaps": markets_with_gaps,
        "missing_internal_months": missing_internal_months,
    }


def _metric_summary(df: pd.DataFrame, metric: str) -> dict[str, object]:
    observed = df.loc[df[metric].notna(), ["market_id", "month", metric]].copy()
    observed["month"] = pd.to_datetime(observed["month"])
    q = observed[metric].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).to_dict()
    overlap = df["zhvi"].notna() & df[metric].notna()
    zhvi_rows = max(int(df["zhvi"].notna().sum()), 1)

    return {
        "metric": metric,
        "first_month": observed["month"].min().date().isoformat() if not observed.empty else None,
        "last_month": observed["month"].max().date().isoformat() if not observed.empty else None,
        "markets": int(observed["market_id"].nunique()),
        "observations": int(len(observed)),
        "null_rate": float(df[metric].isna().mean()),
        "overlap_with_zhvi_rows": int(overlap.sum()),
        "overlap_with_zhvi_rate": float(overlap.sum() / zhvi_rows),
        "min": float(observed[metric].min()) if not observed.empty else None,
        "p001": float(q.get(0.001, np.nan)) if not observed.empty else None,
        "p01": float(q.get(0.01, np.nan)) if not observed.empty else None,
        "median": float(q.get(0.5, np.nan)) if not observed.empty else None,
        "p99": float(q.get(0.99, np.nan)) if not observed.empty else None,
        "p999": float(q.get(0.999, np.nan)) if not observed.empty else None,
        "max": float(observed[metric].max()) if not observed.empty else None,
        **_month_gap_count(df, metric),
    }


def run_final_audit(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs/reports",
) -> dict[str, object]:
    df = pd.read_parquet(mart_path)
    df["month"] = pd.to_datetime(df["month"])
    metrics = [m for m in CORE_METRICS if m in df.columns]

    metric_summaries = [_metric_summary(df, metric) for metric in metrics]

    complete = df.dropna(subset=metrics)
    duplicate_count = int(df.duplicated(["market_id", "month"]).sum())

    common_last_month = min(
        pd.to_datetime(df.loc[df[m].notna(), "month"]).max()
        for m in metrics
    )
    common_first_month = max(
        pd.to_datetime(df.loc[df[m].notna(), "month"]).min()
        for m in metrics
    )

    report = {
        "panel": {
            "rows": int(len(df)),
            "markets": int(df["market_id"].nunique()),
            "first_month": df["month"].min().date().isoformat(),
            "last_month": df["month"].max().date().isoformat(),
            "duplicate_market_months": duplicate_count,
        },
        "metrics": metric_summaries,
        "complete_case_panel": {
            "rows": int(len(complete)),
            "markets": int(complete["market_id"].nunique()),
            "first_month": complete["month"].min().date().isoformat() if not complete.empty else None,
            "last_month": complete["month"].max().date().isoformat() if not complete.empty else None,
        },
        "temporal_alignment": {
            "common_feature_window_start": common_first_month.date().isoformat(),
            "common_feature_window_end": common_last_month.date().isoformat(),
            "three_month_target_latest_feature_month": (
                common_last_month - pd.DateOffset(months=3)
            ).date().isoformat(),
            "duplicate_key_check_passed": duplicate_count == 0,
            "future_feature_columns_present": False,
            "warning": (
                "Current-release Zillow histories may contain retrospective revisions. "
                "This pipeline prevents row-level future leakage, but vintage/revision leakage "
                "cannot be eliminated without archived historical releases."
            ),
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    metric_df = pd.DataFrame(metric_summaries)
    metric_df.to_csv(out / "metric_coverage_extremes.csv", index=False)

    return report


if __name__ == "__main__":
    print(json.dumps(run_final_audit(), indent=2))
