from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.model_table import (
    FULL_FEATURES,
    LIQUIDITY_FEATURES,
    PRICE_FEATURES,
    RENT_FEATURES,
    SUPPLY_FEATURES,
    TARGET,
    build_model_table,
)
from src.forecasting.evaluation import (
    ForecastFold,
    purged_rolling_origin_folds,
    regression_metrics,
)
from src.forecasting.experiment import _fold_predictions
from src.forecasting.models import elastic_net_model
from src.regimes.early_warning import run_early_warning
from src.regimes.rules import build_rule_regimes

NON_PRICE_FEATURES = (
    RENT_FEATURES + SUPPLY_FEATURES + LIQUIDITY_FEATURES
)
STATIC_FEATURES = {"month_sin", "month_cos", "log_size_rank"}
DYNAMIC_FEATURES = [
    feature for feature in FULL_FEATURES if feature not in STATIC_FEATURES
]


def apply_feature_lag(
    table: pd.DataFrame,
    features: list[str],
    months: int,
) -> pd.DataFrame:
    """Delay selected features within each market by a fixed number of months."""
    if months < 0:
        raise ValueError("months must be non-negative")

    out = table.sort_values(["market_id", "month"]).copy()
    if months == 0:
        return out

    for feature in features:
        out[feature] = out.groupby("market_id")[feature].shift(months)
    return out


def _elastic_predictions(
    table: pd.DataFrame,
    folds: list[ForecastFold],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for fold in folds:
        train = table.loc[
            table["month"].le(fold.train_end) & table[TARGET].notna()
        ].copy()
        validation = table.loc[
            table["month"].between(
                fold.validation_start,
                fold.validation_end,
            )
            & table[TARGET].notna()
        ].copy()

        model = elastic_net_model()
        model.fit(train[FULL_FEATURES], train[TARGET])

        frame = validation[
            [
                "market_id",
                "market_name",
                "state_name",
                "size_cohort",
                "month",
                TARGET,
            ]
        ].copy()
        frame["fold"] = fold.fold
        frame["prediction"] = model.predict(validation[FULL_FEATURES])
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def publication_lag_stress_test(
    table: pd.DataFrame,
    folds: list[ForecastFold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate scenario-based feature availability stress tests.

    These are operational stress scenarios, not claims about Zillow's actual
    publication SLA.
    """
    scenarios: list[tuple[str, list[str], int]] = [
        ("contemporaneous", [], 0),
        ("non_price_lag_1m", NON_PRICE_FEATURES, 1),
        ("non_price_lag_2m", NON_PRICE_FEATURES, 2),
        ("all_dynamic_lag_1m", DYNAMIC_FEATURES, 1),
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for name, features, months in scenarios:
        scenario_table = apply_feature_lag(table, features, months)
        predictions = _elastic_predictions(scenario_table, folds)
        predictions["scenario"] = name
        prediction_frames.append(predictions)

        metrics = regression_metrics(
            predictions[TARGET],
            predictions["prediction"],
        )
        metric_rows.append(
            {
                "scenario": name,
                "lagged_feature_count": len(features),
                "lag_months": months,
                **metrics,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    baseline_mae = float(
        metrics.loc[
            metrics["scenario"].eq("contemporaneous"),
            "mae",
        ].iloc[0]
    )
    metrics["mae_degradation_vs_current"] = (
        metrics["mae"] / baseline_mae - 1.0
    )
    return metrics, pd.concat(prediction_frames, ignore_index=True)


def _group_regression_metrics(
    frame: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, part in frame.groupby(group_col, dropna=False):
        metrics = regression_metrics(part[TARGET], part["prediction"])
        rows.append({group_col: str(group), **metrics})
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def forecast_failure_analysis(
    table: pd.DataFrame,
    folds: list[ForecastFold],
) -> dict[str, pd.DataFrame]:
    prediction_frames = [_fold_predictions(table, fold) for fold in folds]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.rename(
        columns={"elastic_net_full": "prediction"}
    )

    metadata = build_rule_regimes(table)[
        [
            "market_id",
            "month",
            "rule_regime",
            "zhvi_volatility_12m",
        ]
    ].copy()

    frame = predictions[
        [
            "market_id",
            "market_name",
            "state_name",
            "size_cohort",
            "month",
            "fold",
            TARGET,
            "prediction",
        ]
    ].merge(
        metadata,
        on=["market_id", "month"],
        how="left",
        validate="many_to_one",
    )
    frame["error"] = frame["prediction"] - frame[TARGET]
    frame["abs_error"] = frame["error"].abs()

    valid_vol = frame["zhvi_volatility_12m"].dropna()
    if valid_vol.nunique() >= 4:
        frame["volatility_quartile"] = pd.qcut(
            frame["zhvi_volatility_12m"],
            q=4,
            labels=["Q1 low", "Q2", "Q3", "Q4 high"],
            duplicates="drop",
        ).astype("string")
    else:
        frame["volatility_quartile"] = "unknown"

    by_fold = _group_regression_metrics(frame, "fold")
    by_cohort = _group_regression_metrics(frame, "size_cohort")
    by_regime = _group_regression_metrics(frame, "rule_regime")
    by_volatility = _group_regression_metrics(
        frame,
        "volatility_quartile",
    )

    market_rows: list[dict[str, object]] = []
    for market_id, part in frame.groupby("market_id"):
        if len(part) < 20:
            continue
        metrics = regression_metrics(part[TARGET], part["prediction"])
        market_rows.append(
            {
                "market_id": int(market_id),
                "market_name": str(part["market_name"].iloc[0]),
                "state_name": str(part["state_name"].iloc[0]),
                **metrics,
            }
        )
    by_market = (
        pd.DataFrame(market_rows)
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )

    extreme_errors = (
        frame.nlargest(50, "abs_error")
        .sort_values("abs_error", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "predictions": frame,
        "by_fold": by_fold,
        "by_cohort": by_cohort,
        "by_regime": by_regime,
        "by_volatility": by_volatility,
        "by_market": by_market,
        "extreme_errors": extreme_errors,
    }


def early_warning_failure_analysis(
    table: pd.DataFrame,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        run_early_warning(table, temp_dir)
        predictions = pd.read_parquet(
            Path(temp_dir) / "early_warning_predictions.parquet"
        )

    actual = predictions["actual"].astype(int)
    alert = predictions["alert_precision50"].astype(bool)

    predictions["error_type"] = np.select(
        [
            alert & actual.eq(1),
            alert & actual.eq(0),
            ~alert & actual.eq(1),
        ],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )

    counts = (
        predictions["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="rows")
    )

    false_positive = (
        predictions.loc[
            predictions["error_type"].eq("false_positive")
        ]
        .sort_values("probability", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
    false_negative = (
        predictions.loc[
            predictions["error_type"].eq("false_negative")
        ]
        .sort_values("probability", ascending=True)
        .head(50)
        .reset_index(drop=True)
    )

    event_rows = int(actual.sum())
    missed_events = int(
        predictions["error_type"].eq("false_negative").sum()
    )
    alert_rows = int(alert.sum())
    false_alerts = int(
        predictions["error_type"].eq("false_positive").sum()
    )

    summary = {
        "rows": float(len(predictions)),
        "event_rows": float(event_rows),
        "alert_rows": float(alert_rows),
        "missed_event_share": (
            float(missed_events / event_rows) if event_rows else np.nan
        ),
        "false_alert_share": (
            float(false_alerts / alert_rows) if alert_rows else np.nan
        ),
    }

    return {
        "predictions": predictions,
        "counts": counts,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "summary": summary,
    }


def _pp(value: float) -> str:
    return f"{value * 100:.3f} pp"


def _pct(value: float) -> str:
    return f"{value:.1%}"


def build_report(
    lag_metrics: pd.DataFrame,
    forecast_failures: dict[str, pd.DataFrame],
    alert_failures: dict[str, pd.DataFrame | dict[str, float]],
) -> str:
    baseline = lag_metrics.loc[
        lag_metrics["scenario"].eq("contemporaneous")
    ].iloc[0]
    worst_lag = lag_metrics.sort_values(
        "mae_degradation_vs_current",
        ascending=False,
    ).iloc[0]

    worst_regime = forecast_failures["by_regime"].iloc[0]
    worst_cohort = forecast_failures["by_cohort"].iloc[0]
    worst_volatility = forecast_failures["by_volatility"].iloc[0]
    worst_fold = forecast_failures["by_fold"].iloc[0]
    alert_summary = alert_failures["summary"]
    assert isinstance(alert_summary, dict)

    lines = [
        "# v1.0 Release Hardening — Publication Lag & Failure Analysis",
        "",
        "## Publication-lag stress test",
        "",
        (
            "The lag scenarios below are operational stress tests. They do not "
            "claim a specific Zillow publication SLA."
        ),
        "",
        "| Scenario | MAE | Directional accuracy | MAE degradation |",
        "|---|---:|---:|---:|",
    ]

    for _, row in lag_metrics.iterrows():
        lines.append(
            (
                f"| {row['scenario']} | {_pp(float(row['mae']))} | "
                f"{_pct(float(row['directional_accuracy']))} | "
                f"{_pct(float(row['mae_degradation_vs_current']))} |"
            )
        )

    lines.extend(
        [
            "",
            (
                f"Current contemporaneous-feature MAE is "
                f"**{_pp(float(baseline['mae']))}**. The harshest tested "
                f"scenario is **{worst_lag['scenario']}**, with "
                f"**{_pct(float(worst_lag['mae_degradation_vs_current']))}** "
                "MAE degradation."
            ),
            "",
            "## Forecast failure analysis",
            "",
            (
                f"Worst validation fold: **{worst_fold['fold']}** "
                f"({_pp(float(worst_fold['mae']))} MAE)."
            ),
            (
                f"Highest-error market-size cohort: "
                f"**{worst_cohort['size_cohort']}** "
                f"({_pp(float(worst_cohort['mae']))} MAE)."
            ),
            (
                f"Highest-error regime: **{worst_regime['rule_regime']}** "
                f"({_pp(float(worst_regime['mae']))} MAE)."
            ),
            (
                f"Highest-error volatility bucket: "
                f"**{worst_volatility['volatility_quartile']}** "
                f"({_pp(float(worst_volatility['mae']))} MAE)."
            ),
            "",
            "### Worst metros with at least 20 validation observations",
            "",
            "| Metro | State | N | MAE | Bias |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for _, row in forecast_failures["by_market"].head(10).iterrows():
        lines.append(
            (
                f"| {row['market_name']} | {row['state_name']} | "
                f"{int(row['n'])} | {_pp(float(row['mae']))} | "
                f"{_pp(float(row['bias']))} |"
            )
        )

    lines.extend(
        [
            "",
            "## Early-warning failure analysis",
            "",
            (
                f"Precision-oriented alerts miss "
                f"**{_pct(float(alert_summary['missed_event_share']))}** of "
                "realized cooling-entry events."
            ),
            (
                f"Among emitted precision-oriented alerts, "
                f"**{_pct(float(alert_summary['false_alert_share']))}** are "
                "false positives."
            ),
            "",
            (
                "These are not hidden as model defects: the dashboard should "
                "treat transition probability as a prioritization signal and "
                "show the threshold policy explicitly."
            ),
            "",
            "## Release implications",
            "",
            "- Keep publication-lag sensitivity visible in Model Health.",
            "- Keep uncertainty wider for smaller markets.",
            "- Treat high-volatility periods as lower-reliability contexts.",
            "- Do not present transition alerts as deterministic predictions.",
            "- Preserve current-release historical-vintage caveat.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_release_analysis(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs/release",
) -> dict[str, object]:
    mart = pd.read_parquet(mart_path)
    table = build_model_table(mart)
    labeled = table.loc[table[TARGET].notna()].copy()

    folds = purged_rolling_origin_folds(
        labeled["month"],
        horizon_months=3,
        min_train_months=36,
        validation_months=6,
    )

    lag_metrics, lag_predictions = publication_lag_stress_test(
        labeled,
        folds,
    )
    forecast_failures = forecast_failure_analysis(labeled, folds)
    alert_failures = early_warning_failure_analysis(table)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    lag_metrics.to_csv(
        output / "publication_lag_metrics.csv",
        index=False,
    )
    lag_predictions.to_parquet(
        output / "publication_lag_predictions.parquet",
        index=False,
    )

    for key in [
        "by_fold",
        "by_cohort",
        "by_regime",
        "by_volatility",
        "by_market",
        "extreme_errors",
    ]:
        forecast_failures[key].to_csv(
            output / f"forecast_failure_{key}.csv",
            index=False,
        )

    alert_failures["counts"].to_csv(
        output / "alert_failure_counts.csv",
        index=False,
    )
    alert_failures["false_positive"].to_csv(
        output / "alert_false_positives.csv",
        index=False,
    )
    alert_failures["false_negative"].to_csv(
        output / "alert_false_negatives.csv",
        index=False,
    )

    summary = {
        "publication_lag": lag_metrics.to_dict(orient="records"),
        "worst_forecast_cohort": (
            forecast_failures["by_cohort"].iloc[0].to_dict()
        ),
        "worst_forecast_regime": (
            forecast_failures["by_regime"].iloc[0].to_dict()
        ),
        "worst_forecast_volatility": (
            forecast_failures["by_volatility"].iloc[0].to_dict()
        ),
        "alert_failure_summary": alert_failures["summary"],
    }
    (output / "release_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output / "release_report.md").write_text(
        build_report(
            lag_metrics,
            forecast_failures,
            alert_failures,
        ),
        encoding="utf-8",
    )

    print((output / "release_report.md").read_text(encoding="utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v1.0 publication-lag and failure analysis."
    )
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/release")
    args = parser.parse_args()
    run_release_analysis(args.mart, args.output_dir)


if __name__ == "__main__":
    main()
