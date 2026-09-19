import pandas as pd

from src.regimes.early_warning import add_cooling_entry_target


def test_cooling_entry_target_flags_next_three_months():
    df = pd.DataFrame(
        {
            "market_id": [1] * 5,
            "month": pd.date_range("2025-01-31", periods=5, freq="ME"),
            "rule_regime": [
                "balanced",
                "balanced",
                "cooling",
                "cooling",
                "balanced",
            ],
        }
    )

    out = add_cooling_entry_target(df)

    assert bool(out.loc[0, "cooling_entry_next_3m"])
    assert bool(out.loc[1, "cooling_entry_next_3m"])
