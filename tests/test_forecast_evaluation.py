import pandas as pd

from src.forecasting.evaluation import (
    purged_rolling_origin_folds,
    regression_metrics,
)


def test_purged_fold_leaves_target_horizon_gap():
    months = pd.Series(pd.date_range("2019-01-31", periods=60, freq="ME"))
    folds = purged_rolling_origin_folds(
        months,
        horizon_months=3,
        min_train_months=36,
        validation_months=6,
    )

    first = folds[0]
    train_period = first.train_end.to_period("M")
    validation_period = first.validation_start.to_period("M")

    assert validation_period.ordinal - train_period.ordinal == 4


def test_regression_metrics_are_exact_for_simple_case():
    result = regression_metrics([0.01, -0.01], [0.02, -0.02])

    assert result["mae"] == 0.01
    assert result["directional_accuracy"] == 1.0
