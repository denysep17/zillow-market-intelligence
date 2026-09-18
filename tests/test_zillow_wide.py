import pandas as pd

from src.transformation.zillow_wide import normalize_zillow_wide


def test_normalize_zillow_wide():
    source = pd.DataFrame(
        {
            "RegionID": [101],
            "SizeRank": [1],
            "RegionName": ["Example Metro"],
            "RegionType": ["msa"],
            "StateName": ["EX"],
            "2026-01-31": [100.0],
            "2026-02-28": [102.0],
        }
    )

    result = normalize_zillow_wide(source, metric="zhvi")

    assert list(result["market_id"]) == [101, 101]
    assert list(result["zhvi"]) == [100.0, 102.0]
    assert result["month"].dtype.kind == "M"
