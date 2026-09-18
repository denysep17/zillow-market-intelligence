import pandas as pd

from src.features.targets import add_forward_growth_target


def test_forward_growth_is_computed_within_market():
    df = pd.DataFrame(
        {
            "market_id": [1, 1, 1, 1],
            "month": pd.date_range("2026-01-01", periods=4, freq="MS"),
            "zhvi": [100.0, 101.0, 102.0, 110.0],
        }
    )
    out = add_forward_growth_target(df, horizon=3)
    assert round(out.loc[0, "zhvi_forward_growth_3m"], 6) == 0.1
