# Zillow Market Intelligence — Portfolio Case Study

## Problem

Headline home-value metrics can confirm a housing-market transition only after supply, liquidity, and demand conditions have already shifted. The project asks whether those earlier signals can improve near-term forecasting and surface markets that deserve analyst attention before the change is obvious in price history alone.

## Product decision

Build a market-level decision-support system rather than a single prediction model.

`Monitor → Detect → Diagnose → Forecast → Prioritize → Act → Measure`

The product combines a 3-month ZHVI forecast, empirical uncertainty interval, current market regime, cooling-transition probability, local predictive drivers, and reliability context.

## Data

Public Zillow Research metro-level monthly datasets:

- ZHVI
- ZORI
- for-sale inventory
- new listings
- price-cut share
- mean days to pending
- Market Heat Index

The canonical analytical mart contains 286,080 metro-month rows across 894 metros, spanning 2000-01 through 2026-08 in the current release.

## Forecasting design

Primary target:

`ZHVI(t+3) / ZHVI(t) - 1`

Validation uses 8 rolling-origin folds, 6-month validation blocks, and a 3-month target-horizon purge. No random split is used.

Models compared:

- no-change baseline
- seasonal naive
- trailing momentum
- price-only Ridge
- full Elastic Net
- histogram gradient boosting

## Forecast result

Elastic Net is the strongest aggregate model:

| Model | MAE | Directional accuracy |
|---|---:|---:|
| Elastic Net | **0.871 pp** | **77.36%** |
| Gradient boosting | 0.907 pp | 76.72% |
| Price-only Ridge | 1.010 pp | 76.44% |
| Trailing momentum | 1.307 pp | 69.77% |

Elastic Net improves MAE by 13.8% versus the price-only Ridge benchmark and by 33.4% versus trailing momentum. The more complex nonlinear challenger is not promoted because it does not improve aggregate error.

## What adds predictive value

Feature-family ablation shows that supply and liquidity signals produce most of the incremental lift beyond price history. Rent momentum is descriptively strong but largely redundant in the full predictive specification.

## Uncertainty

A time-ordered empirical calibration layer produces approximately 90% interval coverage. Size-cohort conditional intervals are preferred because smaller metros have materially wider forecast error.

## Regimes and early warning

A transparent five-state rule system classifies markets as accelerating, tightening, balanced, loosening, or cooling.

The early-warning model predicts entry into a cooling state within 1–3 months. Ranking performance is useful (ROC AUC 0.804; average precision 0.389 against a 12.9% event rate), but thresholding creates a real product trade-off.

Precision-oriented policy:

- precision: 51.4%
- recall: 19.8%
- median true-alert lead: 1 month

This is intentionally framed as prioritization, not deterministic prediction.

## Failure analysis

The global MAE is not treated as universal. Error is higher in:

- high-volatility quartile: 1.276 pp MAE
- cooling regime: 0.989 pp MAE
- rank-301+ metros: 0.952 pp MAE
- worst validation fold: 1.347 pp MAE

Publication-lag stress testing also shows that stale information matters:

- non-price signals lagged 1 month: +6.6% MAE degradation
- non-price signals lagged 2 months: +13.5%
- all dynamic signals lagged 1 month: +31.2%

## Product surface

The deployed Plotly + Streamlit dashboard exposes five workflows:

1. Executive Monitor
2. Market Explorer
3. Metro Deep Dive
4. Alerts
5. Model Health

Live dashboard: https://dashboard-production-b1bb.up.railway.app

## Engineering and reproducibility

- reusable Python modules instead of notebook-only logic
- DuckDB-oriented analytical marts
- pytest + Ruff
- GitHub Actions for data audit, forecasting, regimes, dashboard smoke tests, and release hardening
- Railway deployment with health checking and automatic mart rebuild from current public Zillow releases

## Scientific guardrails

- descriptive, predictive, and causal claims are kept separate
- local feature contributions explain the fitted predictive model only
- current public Zillow history may contain revisions
- publication-lag stress tests are scenarios rather than source-specific SLA claims
- transition alerts are ranking/policy tools, not certainty

## Outcome

The final artifact is not a leaderboard model. It is a production-minded analytical system that makes model performance, uncertainty, operating trade-offs, and failure modes visible to the user.