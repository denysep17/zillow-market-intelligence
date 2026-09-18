from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastFold:
    fold: int
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def purged_rolling_origin_folds(
    months: pd.Series,
    *,
    horizon_months: int = 3,
    min_train_months: int = 36,
    validation_months: int = 6,
) -> list[ForecastFold]:
    """Create non-overlapping validation blocks with a target-horizon purge.

    If validation begins in month V, training feature months end early enough
    that their forward target timestamp is strictly before V.
    """
    periods = pd.PeriodIndex(
        pd.to_datetime(months).dropna().dt.to_period("M").unique()
    ).sort_values()

    first_validation = min_train_months + horizon_months
    folds: list[ForecastFold] = []
    fold_number = 1

    start = first_validation
    while start < len(periods):
        train_end_index = start - horizon_months - 1
        if train_end_index < 0:
            break

        end = min(start + validation_months, len(periods)) - 1
        folds.append(
            ForecastFold(
                fold=fold_number,
                train_end=periods[train_end_index].to_timestamp("M"),
                validation_start=periods[start].to_timestamp("M"),
                validation_end=periods[end].to_timestamp("M"),
            )
        )
        fold_number += 1
        start += validation_months

    return folds


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]

    if len(y) == 0:
        return {
            "n": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "directional_accuracy": np.nan,
        }

    error = pred - y
    return {
        "n": float(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
    }
