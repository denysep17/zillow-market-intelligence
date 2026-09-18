from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.transformation.zillow_wide import normalize_zillow_wide


CANDIDATES = {
    "price_cut_share": {
        "url": "https://files.zillowstatic.com/research/public_csvs/perc_listings_price_cut/Metro_perc_listings_price_cut_uc_sfrcondo_sm_month.csv",
        "role": "seller_pressure",
    },
    "new_listings": {
        "url": "https://files.zillowstatic.com/research/public_csvs/new_listings/Metro_new_listings_uc_sfrcondo_sm_month.csv",
        "role": "supply_flow",
    },
    "sale_to_list": {
        "url": "https://files.zillowstatic.com/research/public_csvs/mean_sale_to_list/Metro_mean_sale_to_list_uc_sfrcondo_sm_month.csv",
        "role": "pricing_competition",
    },
    "affordability_income_needed": {
        "url": "https://files.zillowstatic.com/research/public_csvs/new_homeowner_income_needed/Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "role": "affordability_macro_context",
    },
}


def _download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    from io import BytesIO
    return pd.read_csv(BytesIO(response.content))


def audit_candidates(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs/reports",
) -> list[dict[str, object]]:
    mart = pd.read_parquet(mart_path)
    mart["month"] = pd.to_datetime(mart["month"])
    zhvi_keys = mart.loc[mart["zhvi"].notna(), ["market_id", "month"]].drop_duplicates()
    results: list[dict[str, object]] = []

    for metric, spec in CANDIDATES.items():
        try:
            raw = _download_csv(spec["url"])
            long = normalize_zillow_wide(raw, metric=metric)
            observed = long.loc[long[metric].notna(), ["market_id", "month", metric]]
            overlap = zhvi_keys.merge(
                observed[["market_id", "month"]],
                on=["market_id", "month"],
                how="inner",
            )
            result = {
                "metric": metric,
                "role": spec["role"],
                "download_ok": True,
                "first_month": observed["month"].min().date().isoformat(),
                "last_month": observed["month"].max().date().isoformat(),
                "markets": int(observed["market_id"].nunique()),
                "observations": int(len(observed)),
                "overlap_with_zhvi_rows": int(len(overlap)),
                "overlap_with_zhvi_rate": float(len(overlap) / max(len(zhvi_keys), 1)),
                "null_rate_within_source": float(long[metric].isna().mean()),
            }
        except Exception as exc:
            result = {
                "metric": metric,
                "role": spec["role"],
                "download_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_signal_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(out / "candidate_signal_audit.csv", index=False)
    return results


if __name__ == "__main__":
    print(json.dumps(audit_candidates(), indent=2))
