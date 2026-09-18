from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.model_table import (
    FULL_FEATURES,
    PRICE_FEATURES,
    TARGET,
    build_model_table,
)
from src.forecasting.ablation import run_feature_ablation
from src.forecasting.evaluation import (
    ForecastFold,
    purged_rolling_origin_folds,
    regression_metrics,
)
from src.forecasting.models import (
    elastic_net_model,
    gradient_boosting_model,
    ridge_price_model,
)
from src.forecasting.uncertainty import (
    interval_summary,
    rolling_conformal_predictions,
)

MODEL_ORDER = [
    "no_change",
    "seasonal_naive",
    "trailing_momentum",
    "ridge_price",
    "elastic_net_full",
    "gradient_boosting_full",
]


def _fit_predict(
    model: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    model.fit(train[features], train[TARGET])
    return model.predict(validation[features])


def _fold_predictions(
    table: pd.DataFrame,
    fold: ForecastFold,
) -> pd.DataFrame:
    train = table.loc[
        table["month"].le(fold.train_end) & table[TARGET].notna()
    ].copy()
    validation = table.loc[
        table["month"].between(fold.validation_start, fold.validation_end)
        & table[TARGET].notna()
    ].copy()

    base_columns = [
        "market_id",
        "market_name",
        "state_name",
        "size_rank",
        "size_cohort",
        "month",
        TARGET,
    ]
    output = validation[base_columns].copy()
    output["fold"] = fold.fold

    output["no_change"] = 0.0
    output["seasonal_naive"] = validation["seasonal_naive_3m"].to_numpy()
    output["trailing_momentum"] = validation["zhvi_growth_3m"].to_numpy()

    output["ridge_price"] = _fit_predict(
        ridge_price_model(),
        train,
        validation,
        PRICE_FEATURES,
    )
    output["elastic_net_full"] = _fit_predict(
        elastic_net_model(),
        train,
        validation,
        FULL_FEATURES,
    )
    output["gradient_boosting_full"] = _fit_predict(
        gradient_boosting_model(),
        train,
        validation,
        FULL_FEATURES,
    )
    return output


def _metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        metrics = regression_metrics(predictions[TARGET], predictions[model])
        rows.append({"model": model, **metrics})
    return pd.DataFrame(rows)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, frame in predictions.groupby("fold"):
        for model in MODEL_ORDER:
            metrics = regression_metrics(frame[TARGET], frame[model])
            rows.append({"fold": int(fold), "model": model, **metrics})
    return pd.DataFrame(rows)


def _cohort_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, frame in predictions.groupby("size_cohort"):
        for model in MODEL_ORDER:
            metrics = regression_metrics(frame[TARGET], frame[model])
            rows.append(
                {
                    "size_cohort": str(cohort),
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _fold_winners(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    winners = (
        fold_metrics.sort_values(["fold", "mae", "model"])
        .groupby("fold", as_index=False)
        .first()
    )
    return winners[["fold", "model", "mae", "directional_accuracy"]]


def _best_model(metrics: pd.DataFrame) -> str:
    eligible = metrics.loc[metrics["model"].ne("no_change")].copy()
    return str(eligible.sort_values("mae").iloc[0]["model"])


def _relative_improvement(
    metrics: pd.DataFrame,
    model: str,
    baseline: str,
) -> float:
    index = metrics.set_index("model")
    baseline_mae = float(index.loc[baseline, "mae"])
    model_mae = float(index.loc[model, "mae"])
    return 1.0 - model_mae / baseline_mae


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _pp(value: float) -> str:
    return f"{value * 100:.3f} pp"


def build_markdown_report(
    summary: dict[str, object],
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    fold_winners: pd.DataFrame,
    ablation: pd.DataFrame,
    conformal_summary: dict[str, float],
    conformal_by_cohort: pd.DataFrame,
) -> str:
    best = str(summary["best_model"])
    index = metrics.set_index("model")

    lines = [
        "# Forecasting Experiment — v0.3",
        "",
        "## Experimental design",
        "",
        (
            f"Target: **3-month forward ZHVI growth**. Modeling window: "
            f"**{summary['modeling_window']['start']} to "
            f"{summary['modeling_window']['end']}**."
        ),
        "",
        (
            f"Validation uses **{summary['folds']} non-overlapping rolling-origin "
            "folds** with a 3-month purge, so a training row is used only when "
            "its target would have been observable before the validation block."
        ),
        "",
        "No random train/test split is used.",
        "",
        "## Aggregate out-of-time results",
        "",
        "| Model | MAE | RMSE | Bias | Directional accuracy |",
        "|---|---:|---:|---:|---:|",
    ]

    for model in MODEL_ORDER:
        row = index.loc[model]
        lines.append(
            (
                f"| {model} | {_pp(float(row['mae']))} | "
                f"{_pp(float(row['rmse']))} | {_pp(float(row['bias']))} | "
                f"{_pct(float(row['directional_accuracy']))} |"
            )
        )

    lines.extend(
        [
            "",
            (
                f"**Lowest aggregate MAE:** `{best}`. Its MAE improvement "
                f"versus trailing momentum is "
                f"**{summary['improvement_vs_trailing']:.1%}**, and versus the "
                "price-only Ridge benchmark is "
                f"**{summary['improvement_vs_price_ridge']:.1%}**."
            ),
            "",
            "## Complexity check",
            "",
        ]
    )

    if best == "gradient_boosting_full":
        lines.append(
            "The nonlinear full-feature model earns additional complexity on "
            "aggregate MAE. Stability and uncertainty remain the next gates."
        )
    elif best == "elastic_net_full":
        lines.append(
            "The regularized linear full-feature model is sufficient at this "
            "stage; gradient boosting does not improve aggregate MAE enough to "
            "justify the added complexity."
        )
    elif best == "ridge_price":
        lines.append(
            "The price-only autoregressive benchmark is not beaten. Additional "
            "market signals are not promoted until they demonstrate genuine "
            "out-of-time lift."
        )
    else:
        lines.append(
            "A simple benchmark remains strongest. Additional model complexity "
            "is not currently justified."
        )

    lines.extend(
        [
            "",
            "## Fold stability",
            "",
            "| Fold | Lowest-MAE model | MAE | Directional accuracy |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in fold_winners.iterrows():
        lines.append(
            (
                f"| {int(row['fold'])} | {row['model']} | "
                f"{_pp(float(row['mae']))} | "
                f"{_pct(float(row['directional_accuracy']))} |"
            )
        )

    winner_counts = fold_winners["model"].value_counts()
    winner_text = ", ".join(
        f"{model}: {int(count)}"
        for model, count in winner_counts.items()
    )
    lines.extend(
        [
            "",
            (
                f"Fold winners are not stable across time ({winner_text}). "
                "Aggregate performance therefore does not imply one model is "
                "uniformly strongest across market regimes."
            ),
            "",
            "## Feature-family ablation",
            "",
            "| Feature set | Features | MAE | Directional accuracy |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, row in ablation.iterrows():
        lines.append(
            (
                f"| {row['feature_set']} | {int(row['feature_count'])} | "
                f"{_pp(float(row['mae']))} | "
                f"{_pct(float(row['directional_accuracy']))} |"
            )
        )

    best_ablation = ablation.iloc[0]
    lines.extend(
        [
            "",
            (
                f"Lowest ablation MAE is **{best_ablation['feature_set']}** "
                f"({_pp(float(best_ablation['mae']))}). This test asks whether "
                "rent, supply, or liquidity families provide incremental value "
                "once evaluated out of time rather than selected by correlation."
            ),
            "",
            "## Empirical prediction intervals",
            "",
            (
                "A time-ordered split-conformal-style calibration layer is "
                "evaluated around the Elastic Net point forecast."
            ),
            "",
            "| Scope | Empirical coverage | Mean interval width |",
            "|---|---:|---:|",
            (
                f"| Overall | "
                f"{_pct(float(conformal_summary['empirical_coverage']))} | "
                f"{_pp(float(conformal_summary['mean_interval_width']))} |"
            ),
        ]
    )

    for _, row in conformal_by_cohort.iterrows():
        lines.append(
            (
                f"| {row['size_cohort']} | "
                f"{_pct(float(row['empirical_coverage']))} | "
                f"{_pp(float(row['mean_interval_width']))} |"
            )
        )

    lines.extend(
        [
            "",
            (
                "The nominal target is 90% coverage. Because metro-month "
                "observations are dependent across both geography and time, "
                "coverage is reported empirically; this is not presented as a "
                "formal iid conformal guarantee."
            ),
            "",
            "## Performance by Zillow size-rank cohort",
            "",
            "| Cohort | Model | MAE | Directional accuracy |",
            "|---|---|---:|---:|",
        ]
    )

    cohort_order = ["top_100", "rank_101_300", "rank_301_plus"]
    for cohort in cohort_order:
        frame = cohort_metrics.loc[cohort_metrics["size_cohort"].eq(cohort)]
        if frame.empty:
            continue
        selected = frame.loc[frame["model"].isin([best, "ridge_price"])]
        for _, row in selected.iterrows():
            lines.append(
                (
                    f"| {cohort} | {row['model']} | "
                    f"{_pp(float(row['mae']))} | "
                    f"{_pct(float(row['directional_accuracy']))} |"
                )
            )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- All feature transformations are backward-looking at month t.",
            (
                "- The rolling-origin split includes a target-horizon purge to "
                "avoid training on labels that would not yet exist."
            ),
            (
                "- Hyperparameters are intentionally fixed in this experiment; "
                "there is no full-sample tuning."
            ),
            (
                "- Missing features are imputed from the training fold only, "
                "with missingness indicators added by the sklearn pipeline."
            ),
            (
                "- Historical Zillow revisions remain a vintage-data caveat "
                "for retrospective evaluation."
            ),
            (
                "- Same-month market features assume the public monthly signal "
                "is available by the scoring cutoff; publication-lag sensitivity "
                "should be tested before a production claim."
            ),
            "",
            "## Next gate",
            "",
            (
                "Use fold instability, ablation results, and uncertainty "
                "coverage to define the regime-aware early-warning layer. "
                "Do not add model complexity unless it solves an observed "
                "failure mode."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def _write_mae_svg(metrics: pd.DataFrame, path: Path) -> None:
    data = metrics.sort_values("mae").reset_index(drop=True)
    width, height = 1100, 560
    left, right = 250, 1030
    top, row_gap = 120, 62
    max_mae = float(data["mae"].max()) * 1.10

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#F7F8FA"/>',
        (
            '<text x="55" y="52" font-family="Inter,Arial,sans-serif" '
            'font-size="29" font-weight="700" fill="#111827">'
            "Rolling out-of-time forecast error</text>"
        ),
        (
            '<text x="55" y="80" font-family="Inter,Arial,sans-serif" '
            'font-size="14" fill="#667085">'
            "Mean absolute error for 3-month forward ZHVI growth</text>"
        ),
    ]

    for idx, row in data.iterrows():
        y = top + idx * row_gap
        bar_width = float(row["mae"]) / max_mae * (right - left)
        color = "#006AFF" if idx == 0 else "#98A2B3"
        lines.append(
            f'<text x="55" y="{y + 6}" font-family="Inter,Arial,sans-serif" '
            f'font-size="14" fill="#344054">{row["model"]}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y - 14}" width="{bar_width:.1f}" '
            f'height="28" rx="5" fill="{color}"/>'
        )
        value = float(row["mae"]) * 100
        lines.append(
            f'<text x="{left + bar_width + 10:.1f}" y="{y + 6}" '
            'font-family="Inter,Arial,sans-serif" font-size="13" '
            f'font-weight="600" fill="#101828">{value:.3f} pp</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs",
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
    if not folds:
        raise ValueError("No rolling-origin folds could be constructed")

    prediction_frames = [_fold_predictions(labeled, fold) for fold in folds]
    predictions = pd.concat(prediction_frames, ignore_index=True)

    metrics = _metrics_from_predictions(predictions)
    fold_metrics = _fold_metrics(predictions)
    cohort_metrics = _cohort_metrics(predictions)
    fold_winners = _fold_winners(fold_metrics)

    ablation = run_feature_ablation(labeled, folds)

    conformal_predictions = rolling_conformal_predictions(
        labeled,
        folds,
        confidence=0.90,
        calibration_months=6,
        horizon_months=3,
    )
    conformal_summary, conformal_by_cohort = interval_summary(
        conformal_predictions
    )

    best = _best_model(metrics)
    summary: dict[str, object] = {
        "target": TARGET,
        "modeling_window": {
            "start": labeled["month"].min().date().isoformat(),
            "end": labeled["month"].max().date().isoformat(),
        },
        "rows": int(len(labeled)),
        "markets": int(labeled["market_id"].nunique()),
        "folds": len(folds),
        "validation_rows": int(len(predictions)),
        "best_model": best,
        "improvement_vs_trailing": _relative_improvement(
            metrics,
            best,
            "trailing_momentum",
        ),
        "improvement_vs_price_ridge": _relative_improvement(
            metrics,
            best,
            "ridge_price",
        ),
        "ablation_best_feature_set": str(ablation.iloc[0]["feature_set"]),
        "ablation_best_mae": float(ablation.iloc[0]["mae"]),
        "conformal": conformal_summary,
        "fold_winner_counts": {
            str(model): int(count)
            for model, count in fold_winners["model"].value_counts().items()
        },
        "fold_definitions": [
            {
                "fold": fold.fold,
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
            }
            for fold in folds
        ],
    }

    output = Path(output_dir)
    reports = output / "reports"
    figures = output / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(reports / "forecast_metrics.csv", index=False)
    fold_metrics.to_csv(reports / "forecast_metrics_by_fold.csv", index=False)
    cohort_metrics.to_csv(
        reports / "forecast_metrics_by_cohort.csv",
        index=False,
    )
    fold_winners.to_csv(
        reports / "forecast_fold_winners.csv",
        index=False,
    )
    ablation.to_csv(
        reports / "forecast_ablation.csv",
        index=False,
    )
    predictions.to_parquet(
        reports / "forecast_predictions.parquet",
        index=False,
    )
    conformal_predictions.to_parquet(
        reports / "forecast_conformal_predictions.parquet",
        index=False,
    )
    conformal_by_cohort.to_csv(
        reports / "forecast_conformal_by_cohort.csv",
        index=False,
    )
    (reports / "forecast_conformal_summary.json").write_text(
        json.dumps(conformal_summary, indent=2),
        encoding="utf-8",
    )
    (reports / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (reports / "forecast_report.md").write_text(
        build_markdown_report(
            summary,
            metrics,
            fold_metrics,
            cohort_metrics,
            fold_winners,
            ablation,
            conformal_summary,
            conformal_by_cohort,
        ),
        encoding="utf-8",
    )
    _write_mae_svg(metrics, figures / "forecast_mae.svg")

    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print("\nFeature-family ablation:")
    print(ablation.to_string(index=False))
    print("\nConformal interval summary:")
    print(json.dumps(conformal_summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run purged rolling-origin Zillow forecast experiment."
    )
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_experiment(args.mart, args.output_dir)


if __name__ == "__main__":
    main()
