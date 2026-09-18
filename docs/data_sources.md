# v0.1 data source contract

The first build deliberately starts with five metro-level monthly Zillow Research series that cover the minimum viable signal families required for the early-warning system.

| Source | Analytical role |
|---|---|
| ZHVI | Outcome and price momentum |
| ZORI | Rental demand / rent momentum |
| For-Sale Inventory | Supply |
| Mean Days to Pending | Market velocity |
| Market Heat Index | Composite supply-demand balance |

## Why start with five sources?

The initial goal is not to maximize feature count. It is to create a reproducible market-month data foundation, establish historical coverage, quantify missingness, and determine whether the candidate signals have sufficient overlap for leakage-safe modeling.

Additional Zillow series such as new listings, share of listings with a price cut, sale-to-list ratio, affordability, and sales nowcast should be added only after the v0.1 coverage audit confirms the canonical join strategy.

## Canonical grain

`market_id × month`

## Reproducibility

URLs and semantic roles are versioned in `config/data_sources.yaml`. Each download produces a local manifest containing its timestamp, byte size, URL, and SHA-256 hash.

Raw CSVs are immutable and excluded from Git. Normalized interim datasets are written as Parquet.

## Required audit before modeling

For each metric:
- first and last available month
- number of markets
- null rate
- duplicate market-month records
- percentage of rows overlapping the ZHVI universe
- metro coverage by year
- longest missing run per market
- distribution and extreme values

No model development should begin until this audit is complete.
