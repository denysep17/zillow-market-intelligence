from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.model_table import (
    FULL_FEATURES,
    PRICE_FEATURES,
    TARGET,
    build_model_table,
)
from src.forecasting.evaluation import (
    ForecastFold,
    purged_rolling_origin_folds,
    regression_metrics,
)

MODEL_ORDER = [
    "no_change",
    "seasonal_naive",
    "trailing_momentum",
    "ridge_price",
    "elastic_net_full",
    "gradient_boosting_full",
]


def _ridge_price_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _elastic_net_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.0005,
                    l1_ratio=0.20,
                    max_iter=20_000,
                    random_state=42,
                ),
            ),
        ]
    )


def _gradient_boosting_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


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
        _ridge_price_model(),
        train,
        validation,
        PRICE_FEATURES,
    )
    output["elastic_net_full"] = _fit_predict(
        _elastic_net_model(),
        train,
        validation,
        FULL_FEATURES,
    )
    output["gradient_boosting_full"] = _fit_predict(
        _gradient_boosting_model(),
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
                f"price-only Ridge benchmark is "
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
            "aggregate MAE. The next gate is stability by fold and cohort, not "
            "further model complexity."
        )
    elif best == "elastic_net_full":
        lines.append(
            "The regularized linear full-feature model is sufficient at this "
            "stage; a more complex nonlinear model does not improve aggregate "
            "MAE enough to justify itself."
        )
    elif best == "ridge_price":
        lines.append(
            "The price-only autoregressive benchmark is not beaten. Additional "
            "market signals should not be claimed as useful until feature or "
            "regime design produces genuine out-of-time lift."
        )
    else:
        lines.append(
            "A simple benchmark remains strongest. Model complexity is not "
            "currently justified."
        )

    lines.extend(
        [
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
            "## Fold stability",
            "",
            (
                "A model is not promoted based only on pooled performance. "
                "The fold-level output is retained to identify periods where "
                "the ranking reverses or errors spike."
            ),
            "",
            "## Guardrails",
            "",
            "- All feature transformations are backward-looking at month t.",
            (
                "- The rolling-origin split includes a target-horizon purge to "
                "avoid training on labels that would not yet exist."
            ),
            (
                "- Hyperparameters are intentionally fixed in this first "
                "experiment; there is no full-sample tuning."
            ),
            (
                "- Missing non-price features are imputed from the training "
                "fold only, with missingness indicators added by the pipeline."
            ),
            (
                "- Zillow historical revisions remain a vintage-data caveat "
                "for retrospective evaluation."
            ),
            "",
            "## Next gate",
            "",
            (
                "Inspect fold-level and cohort-level failure modes, then add "
                "prediction intervals and feature-ablation tests before "
                "promoting a model into the regime / early-warning layer."
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
        lines.append(
            f'<text x="{left + bar_width + 10:.1f}" y="{y + 6}" '
            'font-family="Inter,Arial,sans-serif" font-size="13" '
            f'font-weight="600" fill="#101828">{float(row["mae"]) * 100:.3f} pp</text>'
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
    predictions.to_parquet(
        reports / "forecast_predictions.parquet",
        index=False,
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
        ),
        encoding="utf-8",
    )
    _write_mae_svg(metrics, figures / "forecast_mae.svg")

    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
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
