import pandas as pd
import pytest

from src.validation.contracts import assert_positive_metric, assert_unique_market_month


def test_unique_market_month_passes():
    df = pd.DataFrame(
        {
            "market_id": [1, 1],
            "month": ["2026-01-01", "2026-02-01"],
        }
    )
    assert_unique_market_month(df)


def test_unique_market_month_raises_on_duplicates():
    df = pd.DataFrame(
        {
            "market_id": [1, 1],
            "month": ["2026-01-01", "2026-01-01"],
        }
    )
    with pytest.raises(ValueError):
        assert_unique_market_month(df)


def test_positive_metric_raises_on_non_positive_values():
    df = pd.DataFrame({"zhvi": [100.0, 0.0]})
    with pytest.raises(ValueError):
        assert_positive_metric(df, "zhvi")
