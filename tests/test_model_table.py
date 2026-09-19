import pandas as pd

from src.features.model_table import TARGET, build_model_table


def test_model_table_forward_target_and_seasonal_naive():
    months = pd.date_range("2018-01-31", periods=24, freq="ME")
    df = pd.DataFrame(
        {
            "market_id": [1] * 24,
            "month": months,
            "zhvi": [100.0 + i for i in range(24)],
            "zori": [10.0 + i * 0.1 for i in range(24)],
            "inventory": [50.0 + i for i in range(24)],
            "new_listings": [20.0 + i for i in range(24)],
            "price_cut_share": [0.10 + i * 0.001 for i in range(24)],
            "days_to_pending": [30.0 + i for i in range(24)],
            "market_heat": [70.0 - i for i in range(24)],
            "size_rank": [10] * 24,
        }
    )

    out = build_model_table(df)
    row = out.iloc[0]

    assert TARGET in out.columns
    assert row["seasonal_naive_3m"] > 0
    assert row["zhvi_growth_12m"] > 0
