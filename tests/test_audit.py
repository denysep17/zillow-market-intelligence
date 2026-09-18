import pandas as pd

from src.validation.audit import metric_audit


def test_metric_audit_reports_temporal_coverage():
    df = pd.DataFrame(
        {
            "market_id": [1, 1, 2, 2],
            "month": pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"]
            ),
            "zhvi": [100.0, 101.0, 200.0, None],
        }
    )

    result = metric_audit(df, "zhvi")

    assert result["observations"] == 3
    assert result["markets"] == 2
    assert result["first_month"] == "2025-01-01"
    assert result["last_month"] == "2025-02-01"
    assert result["null_rate"] == 0.25
