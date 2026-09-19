from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.model_table import FULL_FEATURES, build_model_table
from src.forecasting.evaluation import purged_rolling_origin_folds
from src.regimes.rules import build_rule_regimes

ALERT_FEATURES = FULL_FEATURES + [
    "zhvi_growth_3m_pctile_regime",
    "inventory_growth_3m_pctile_regime",
    "price_cut_change_3m_pctile_regime",
    "days_pending_change_3m_pctile_regime",
    "market_heat_change_3m_pctile_regime",
]


def add_cooling_entry_target(
    df: pd.DataFrame,
    horizon_months: int = 3,
) -> pd.DataFrame:
    out = df.sort_values(["market_id", "month"]).copy()
    future_cooling = []

    for step in range(1, horizon_months + 1):
        future_state = out.groupby("market_id")["rule_regime"].shift(-step)
        future_cooling.append(future_state.eq("cooling"))

    out["cooling_entry_next_3m"] = np.logical_or.reduce(future_cooling)
    out["eligible_alert_row"] = ~out["rule_regime"].eq("cooling")

    future_available = out.groupby("market_id")["rule_regime"].shift(
        -horizon_months
    ).notna()
    out.loc[~future_available, "cooling_entry_next_3m"] = np.nan
    return out


def _model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _threshold_from_training(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:
    candidates = np.linspace(0.20, 0.80, 61)
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in candidates:
        predicted = probability >= threshold
        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            predicted,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["actual"].astype(int)
    predicted = frame["alert"].astype(int)
    probability = frame["probability"].astype(float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predicted,
        average="binary",
        zero_division=0,
    )

    alerts = int(predicted.sum())
    false_alerts = int(((predicted == 1) & (y == 0)).sum())

    return {
        "n": float(len(frame)),
        "event_rate": float(y.mean()),
        "alerts": float(alerts),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alert_share": (
            float(false_alerts / alerts) if alerts else np.nan
        ),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def run_early_warning(
    table: pd.DataFrame,
    output_dir: str | Path = "outputs/reports",
) -> dict[str, float]:
    data = build_rule_regimes(table)
    data = add_cooling_entry_target(data)
    data = data.loc[
        data["eligible_alert_row"]
        & data["cooling_entry_next_3m"].notna()
    ].copy()

    folds = purged_rolling_origin_folds(
        data["month"],
        horizon_months=3,
        min_train_months=36,
        validation_months=6,
    )

    predictions: list[pd.DataFrame] = []

    for fold in folds:
        train = data.loc[
            data["month"].le(fold.train_end)
        ].copy()
        validation = data.loc[
            data["month"].between(
                fold.validation_start,
                fold.validation_end,
            )
        ].copy()

        if train.empty or validation.empty:
            continue

        model = _model()
        y_train = train["cooling_entry_next_3m"].astype(int)
        model.fit(train[ALERT_FEATURES], y_train)

        train_probability = model.predict_proba(
            train[ALERT_FEATURES]
        )[:, 1]
        threshold = _threshold_from_training(y_train, train_probability)

        probability = model.predict_proba(
            validation[ALERT_FEATURES]
        )[:, 1]

        frame = validation[
            [
                "market_id",
                "market_name",
                "state_name",
                "size_cohort",
                "month",
                "rule_regime",
            ]
        ].copy()
        frame["fold"] = fold.fold
        frame["actual"] = validation["cooling_entry_next_3m"].astype(int)
        frame["probability"] = probability
        frame["threshold"] = threshold
        frame["alert"] = probability >= threshold
        predictions.append(frame)

    if not predictions:
        raise ValueError("No early-warning predictions were generated")

    result = pd.concat(predictions, ignore_index=True)
    summary = _metrics(result)

    fold_rows: list[dict[str, float | int]] = []
    for fold, frame in result.groupby("fold"):
        fold_rows.append({"fold": int(fold), **_metrics(frame)})
    fold_metrics = pd.DataFrame(fold_rows)

    cohort_rows: list[dict[str, float | str]] = []
    for cohort, frame in result.groupby("size_cohort"):
        cohort_rows.append(
            {
                "size_cohort": str(cohort),
                **_metrics(frame),
            }
        )
    cohort_metrics = pd.DataFrame(cohort_rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        out / "early_warning_predictions.parquet",
        index=False,
    )
    fold_metrics.to_csv(
        out / "early_warning_by_fold.csv",
        index=False,
    )
    cohort_metrics.to_csv(
        out / "early_warning_by_cohort.csv",
        index=False,
    )
    (out / "early_warning_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cooling-transition early-warning evaluation."
    )
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()

    mart = pd.read_parquet(args.mart)
    table = build_model_table(mart)
    run_early_warning(table, args.output_dir)


if __name__ == "__main__":
    main()
