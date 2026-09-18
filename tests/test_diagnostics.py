import pandas as pd

from src.diagnostics.market_diagnostics import (
    add_research_features,
    lead_lag_table,
)


def _example_panel() -> pd.DataFrame:
    months = pd.date_range("2024-01-31", periods=18, freq="ME")
    rows = []
    for market_id, scale in [(1, 1.0), (2, 1.2)]:
        for idx, month in enumerate(months):
            rows.append(
                {
                    "market_id": market_id,
                    "month": month,
                    "zhvi": 100_000 * scale * (1 + 0.01 * idx),
                    "zori": 1_000 * scale * (1 + 0.008 * idx),
                    "inventory": 100 * scale * (1 + 0.02 * idx),
                    "new_listings": 50 * scale * (1 + 0.01 * idx),
                    "price_cut_share": 0.10 + 0.002 * idx,
                    "days_to_pending": 30 + idx,
                    "market_heat": 70 - idx,
                }
            )
    return pd.DataFrame(rows)


def test_feature_engineering_uses_market_local_lags():
    df = add_research_features(_example_panel())

    first_market = df.loc[df["market_id"].eq(1)].reset_index(drop=True)
    assert pd.isna(first_market.loc[0, "zhvi_growth_3m"])
    assert first_market.loc[3, "zhvi_growth_3m"] > 0
    assert first_market.loc[0, "zhvi_forward_growth_3m"] > 0


def test_lead_lag_table_has_three_horizons():
    df = add_research_features(_example_panel())
    rows = lead_lag_table(df)
    horizons = {row["horizon_months"] for row in rows}
    assert horizons == {1, 3, 6}
