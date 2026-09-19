import pandas as pd

from src.regimes.rules import assign_rule_regime


def test_rule_regimes_assign_expected_states():
    df = pd.DataFrame(
        {
            "zhvi_growth_3m_pctile_regime": [0.90, 0.10, 0.50],
            "inventory_growth_3m_pctile_regime": [0.20, 0.90, 0.50],
            "price_cut_change_3m_pctile_regime": [0.20, 0.90, 0.50],
            "days_pending_change_3m_pctile_regime": [0.40, 0.80, 0.50],
            "market_heat_change_3m_pctile_regime": [0.80, 0.20, 0.50],
        }
    )

    result = assign_rule_regime(df)

    assert result.iloc[0] == "accelerating"
    assert result.iloc[1] == "cooling"
    assert result.iloc[2] == "balanced"
