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


def _cohort_radii(
    calibration: pd.DataFrame,
    residuals: np.ndarray,
    confidence: float,
    *,
    min_group_size: int = 50,
) -> tuple[float, dict[str, float]]:
    global_radius = conformal_quantile(residuals, confidence)
    calibration_scores = calibration[["size_cohort"]].copy()
    calibration_scores["score"] = residuals

    radii: dict[str, float] = {}
    for cohort, frame in calibration_scores.groupby("size_cohort"):
        scores = frame["score"].to_numpy()
        if len(scores) < min_group_size:
            radii[str(cohort)] = global_radius
        else:
            radii[str(cohort)] = conformal_quantile(scores, confidence)

    return global_radius, radii


def rolling_conformal_predictions(
    table: pd.DataFrame,
    folds: list[ForecastFold],
    *,
    confidence: float = 0.90,
    calibration_months: int = 6,
    horizon_months: int = 3,
    conditional_on_cohort: bool = False,
) -> pd.DataFrame:
    """Generate time-ordered empirical conformal-style forecast intervals.

    Calibration is separated from proper training by the target horizon. If
    conditional_on_cohort is true, interval radii are estimated separately for
    Zillow size-rank cohorts, with a global fallback for small groups.

    Metro-month observations are dependent across geography and time, so the
    project reports empirical coverage rather than claiming an iid conformal
    coverage guarantee.
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

        calibration_prediction = model.predict(calibration[FULL_FEATURES])
        residuals = np.abs(
            calibration[TARGET].to_numpy() - calibration_prediction
        )
        global_radius, cohort_radii = _cohort_radii(
            calibration,
            residuals,
            confidence,
        )

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

        if conditional_on_cohort:
            output["interval_radius"] = (
                output["size_cohort"]
                .astype(str)
                .map(cohort_radii)
                .fillna(global_radius)
            )
            output["calibration_scope"] = "size_cohort"
        else:
            output["interval_radius"] = global_radius
            output["calibration_scope"] = "global"

        output["lower"] = output["prediction"] - output["interval_radius"]
        output["upper"] = output["prediction"] + output["interval_radius"]
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
    width = predictions["upper"] - predictions["lower"]
    summary = {
        "n": float(len(predictions)),
        "empirical_coverage": float(predictions["covered"].mean()),
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(width.median()),
        "point_mae": float(point["mae"]),
    }

    rows: list[dict[str, object]] = []
    for cohort, frame in predictions.groupby("size_cohort"):
        cohort_width = frame["upper"] - frame["lower"]
        rows.append(
            {
                "size_cohort": str(cohort),
                "n": int(len(frame)),
                "empirical_coverage": float(frame["covered"].mean()),
                "mean_interval_width": float(cohort_width.mean()),
            }
        )

    return summary, pd.DataFrame(rows)
