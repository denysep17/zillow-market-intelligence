from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.features.model_table import FULL_FEATURES, TARGET
from src.forecasting.evaluation import ForecastFold, regression_metrics
from src.forecasting.models import elastic_net_model


def conformal_quantile(
    absolute_residuals: np.ndarray,
    confidence: float,
) -> float:
    scores = np.asarray(absolute_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("Calibration residuals are empty")

    level = math.ceil((len(scores) + 1) * confidence) / len(scores)
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def rolling_conformal_predictions(
    table: pd.DataFrame,
    folds: list[ForecastFold],
    *,
    confidence: float = 0.90,
    calibration_months: int = 6,
    horizon_months: int = 3,
) -> pd.DataFrame:
    """Generate split-conformal-style intervals for each validation fold.

    Calibration is time-ordered and separated from proper training by the
    target horizon. Because metro-month observations are dependent, empirical
    coverage is reported rather than claiming exchangeability-based guarantees.
    """
    frames: list[pd.DataFrame] = []

    for fold in folds:
        calibration_end = fold.train_end.to_period("M")
        calibration_start = calibration_end - (calibration_months - 1)
        proper_train_end = calibration_start - (horizon_months + 1)

        proper_train = table.loc[
            table["month"].dt.to_period("M").le(proper_train_end)
            & table[TARGET].notna()
        ].copy()
        calibration = table.loc[
            table["month"].dt.to_period("M").between(
                calibration_start,
                calibration_end,
            )
            & table[TARGET].notna()
        ].copy()
        validation = table.loc[
            table["month"].between(
                fold.validation_start,
                fold.validation_end,
            )
            & table[TARGET].notna()
        ].copy()

        if proper_train.empty or calibration.empty or validation.empty:
            continue

        model = elastic_net_model()
        model.fit(proper_train[FULL_FEATURES], proper_train[TARGET])

        calibration_prediction = model.predict(
            calibration[FULL_FEATURES]
        )
        residuals = np.abs(
            calibration[TARGET].to_numpy() - calibration_prediction
        )
        radius = conformal_quantile(residuals, confidence)

        prediction = model.predict(validation[FULL_FEATURES])
        output = validation[
            [
                "market_id",
                "market_name",
                "state_name",
                "size_rank",
                "size_cohort",
                "month",
                TARGET,
            ]
        ].copy()
        output["fold"] = fold.fold
        output["prediction"] = prediction
        output["lower"] = prediction - radius
        output["upper"] = prediction + radius
        output["interval_radius"] = radius
        output["covered"] = (
            output[TARGET].ge(output["lower"])
            & output[TARGET].le(output["upper"])
        )
        frames.append(output)

    if not frames:
        raise ValueError("No conformal validation predictions were produced")

    return pd.concat(frames, ignore_index=True)


def interval_summary(
    predictions: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    point = regression_metrics(
        predictions[TARGET],
        predictions["prediction"],
    )
    summary = {
        "n": float(len(predictions)),
        "empirical_coverage": float(predictions["covered"].mean()),
        "mean_interval_width": float(
            (predictions["upper"] - predictions["lower"]).mean()
        ),
        "median_interval_width": float(
            (predictions["upper"] - predictions["lower"]).median()
        ),
        "point_mae": float(point["mae"]),
    }

    rows: list[dict[str, object]] = []
    for cohort, frame in predictions.groupby("size_cohort"):
        rows.append(
            {
                "size_cohort": str(cohort),
                "n": int(len(frame)),
                "empirical_coverage": float(frame["covered"].mean()),
                "mean_interval_width": float(
                    (frame["upper"] - frame["lower"]).mean()
                ),
            }
        )

    return summary, pd.DataFrame(rows)
