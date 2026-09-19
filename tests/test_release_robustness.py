import pandas as pd

from src.release.robustness import apply_feature_lag


def test_apply_feature_lag_stays_within_market():
    frame = pd.DataFrame(
        {
            "market_id": [1, 1, 2, 2],
            "month": pd.to_datetime(
                ["2025-01-31", "2025-02-28"] * 2
            ),
            "signal": [10.0, 20.0, 100.0, 200.0],
        }
    )

    result = apply_feature_lag(frame, ["signal"], 1)

    assert pd.isna(result.loc[result["market_id"].eq(1), "signal"].iloc[0])
    assert result.loc[result["market_id"].eq(1), "signal"].iloc[1] == 10.0
    assert pd.isna(result.loc[result["market_id"].eq(2), "signal"].iloc[0])
    assert result.loc[result["market_id"].eq(2), "signal"].iloc[1] == 100.0


def test_zero_month_lag_preserves_signal():
    frame = pd.DataFrame(
        {
            "market_id": [1, 1],
            "month": pd.to_datetime(["2025-01-31", "2025-02-28"]),
            "signal": [10.0, 20.0],
        }
    )

    result = apply_feature_lag(frame, ["signal"], 0)

    assert result["signal"].tolist() == [10.0, 20.0]
