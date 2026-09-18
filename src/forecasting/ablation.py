from __future__ import annotations

import pandas as pd

from src.features.model_table import (
    FULL_FEATURES,
    LIQUIDITY_FEATURES,
    PRICE_FEATURES,
    RENT_FEATURES,
    SUPPLY_FEATURES,
    TARGET,
)
from src.forecasting.evaluation import ForecastFold, regression_metrics
from src.forecasting.models import elastic_net_model


def feature_sets() -> dict[str, list[str]]:
    return {
        "price_only_elastic": list(PRICE_FEATURES),
        "full": list(FULL_FEATURES),
        "full_without_rent": [
            feature for feature in FULL_FEATURES if feature not in RENT_FEATURES
        ],
        "full_without_supply": [
            feature for feature in FULL_FEATURES if feature not in SUPPLY_FEATURES
        ],
        "full_without_liquidity": [
            feature
            for feature in FULL_FEATURES
            if feature not in LIQUIDITY_FEATURES
        ],
    }


def run_feature_ablation(
    table: pd.DataFrame,
    folds: list[ForecastFold],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for name, features in feature_sets().items():
        actual_parts: list[pd.Series] = []
        prediction_parts: list[pd.Series] = []

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
            model.fit(train[features], train[TARGET])
            prediction = model.predict(validation[features])

            actual_parts.append(validation[TARGET])
            prediction_parts.append(
                pd.Series(prediction, index=validation.index)
            )

        actual = pd.concat(actual_parts).sort_index()
        predicted = pd.concat(prediction_parts).sort_index()
        metrics = regression_metrics(actual, predicted)

        rows.append(
            {
                "feature_set": name,
                "feature_count": len(features),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
