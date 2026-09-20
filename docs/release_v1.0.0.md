# v1.0.0 — Zillow Market Intelligence

## Release summary

This release packages the project as a complete end-to-end housing-market data science product using public Zillow Research data.

Live dashboard: https://dashboard-production-b1bb.up.railway.app

## Included

- reproducible Zillow Research ingestion and data-quality pipeline
- canonical metro-month analytical mart across 894 metros
- market diagnostics and lead-lag analysis
- leakage-safe 3-month ZHVI forecasting
- Elastic Net primary model with baseline and nonlinear challenger comparison
- empirical forecast uncertainty with size-cohort calibration
- interpretable housing-market regime detection
- 1–3 month cooling-transition early-warning model
- explicit alert policy precision/recall trade-offs
- publication-lag stress testing
- forecast and alert failure analysis
- Plotly + Streamlit decision-support dashboard
- Railway live deployment with health check
- CI workflows for validation, forecasting, regimes, dashboard smoke tests, and release hardening
- portfolio case study and 60–90 second demo script

## Headline results

- Elastic Net forecast MAE: **0.871 pp**
- directional accuracy: **77.36%**
- MAE improvement vs. trailing momentum: **33.4%**
- MAE improvement vs. price-only Ridge: **13.8%**
- empirical interval coverage: **90.53%** with size-cohort conditional calibration
- cooling-entry ROC AUC: **0.804**
- cooling-entry average precision: **0.389** vs. **12.9%** event prevalence

## Operating constraints

The release intentionally exposes failure modes rather than hiding them.

- high-volatility quartile forecast MAE: **1.276 pp**
- cooling-regime forecast MAE: **0.989 pp**
- rank-301+ metro forecast MAE: **0.952 pp**
- one-month lag across all dynamic features increases MAE by **31.2%**
- precision-oriented alert policy has **51.4% precision** and **19.8% recall**

## Intended use

This is a research and portfolio decision-support system using public aggregate market data. It is not a Zillow internal product, does not use private Zillow telemetry, and is not intended for individualized financial or real-estate advice.

## Links

- Live dashboard: https://dashboard-production-b1bb.up.railway.app
- Case study: docs/portfolio_case_study.md
- Model card: docs/model_card.md
- Release hardening: docs/release_hardening_2026-09-19.md
- Demo script: docs/demo_script.md
