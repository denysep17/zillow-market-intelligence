# Zillow Market Intelligence

An early-warning housing-market intelligence system designed to detect market regime shifts, forecast near-term momentum, explain predictive signals, and surface markets that deserve analyst attention.

> **Research question:** Can leading housing-market signals identify turning points before headline home-value measures fully reflect them?

## Product

This project is built as a decision-support product rather than a standalone ML notebook. It combines market monitoring, forecasting, regime detection, interpretability, anomaly detection, and model-health monitoring into one workflow.

**Decision loop**

`Monitor → Detect → Diagnose → Forecast → Prioritize → Act → Measure`

Primary users:
- Product and strategy teams monitoring market conditions
- Data scientists and analysts investigating market changes
- Business stakeholders who need concise, interpretable market signals

## Why this matters

Housing-market conditions are distributed across multiple signals: prices, rents, inventory, price cuts, listing velocity, and other market-activity metrics. Looking at one metric in isolation can make turning points visible only after they are already underway.

The project tests whether leading supply and liquidity indicators add incremental predictive information beyond historical price momentum.

## Core hypothesis

`H1: Supply and liquidity indicators improve out-of-time prediction of future market momentum beyond autoregressive price-only baselines.`

Primary target:

`3-month forward ZHVI growth = ZHVI(t+3) / ZHVI(t) - 1`

Secondary target:

`P(next market regime | current regime, current market signals)`

## Architecture

```text
Zillow Research data
        ↓
Raw ingestion
        ↓
Schema + quality validation
        ↓
Canonical market/month tables
        ↓
Feature engineering
        ↓
┌────────────────────┬────────────────────┐
│ Market diagnostics │ Forecasting        │
│ Regime detection   │ Uncertainty        │
└────────────────────┴────────────────────┘
        ↓
Interpretation + anomaly detection
        ↓
Decision / attention layer
        ↓
Interactive dashboard
        ↓
Model monitoring
```

## Analytical stack

- **Python:** pandas, NumPy, scikit-learn, statsmodels
- **SQL:** DuckDB for reproducible local analytical marts
- **Visualization:** Plotly / Streamlit
- **Modeling:** regularized linear baselines + gradient boosting
- **Validation:** rolling-origin / walk-forward evaluation
- **Testing:** pytest + data-quality assertions
- **CI:** GitHub Actions

## Planned data

The project will use publicly available Zillow Research datasets at compatible geographic and temporal grains, including where available:

- Zillow Home Value Index (ZHVI)
- Zillow Observed Rent Index (ZORI)
- Inventory
- New listings
- Price cuts
- Days to pending
- Sale-to-list ratio
- Other market activity indicators

Raw source files are intentionally excluded from Git history. See [data/README.md](data/README.md).

## Repository structure

```text
zillow-market-intelligence/
├── config/             # Project configuration
├── data/               # Raw/interim/processed data conventions
├── dashboard/          # Interactive product UI
├── docs/               # Methodology, architecture, model card
├── models/             # Serialized model artifacts (ignored by default)
├── notebooks/          # Research notebooks, not production logic
├── outputs/            # Figures and reports
├── sql/                # Staging, mart, and quality SQL
├── src/                # Reusable production code
├── tests/              # Unit and data-contract tests
└── .github/workflows/  # CI
```

## Project milestones

### v0.1 — Data foundation
- [x] Lock initial Zillow Research source contract
- [x] Build reproducible ingestion pipeline + SHA-256 download manifest
- [x] Implement wide-to-long metro-month normalization
- [x] Add schema and data-quality validation
- [x] Implement canonical analytical mart builder
- [x] Add coverage-audit report generator
- [x] Execute the full build against current Zillow releases and review coverage
- [x] Promote price-cut share and new listings after live coverage audit
- [x] Keep sale-to-list and affordability as optional/contextual signals
- [x] Validate temporal alignment and document revision-vintage leakage
- [x] Publish final v0.1 data-audit report

### v0.2 — Market diagnostics
- [x] Cross-market and metro EDA
- [x] Seasonality and synchronized-shift analysis
- [x] Pooled + decomposed lead-lag exploration
- [x] Zillow size-rank cohort analysis
- [x] Top-100 metro market matrix
- [x] Lock feature hypotheses for forecasting

### v0.3 — Forecasting
- [ ] Define leakage-safe feature set
- [ ] Build naive and autoregressive baselines
- [ ] Add Elastic Net benchmark
- [ ] Add gradient-boosting model
- [ ] Implement rolling-origin evaluation
- [ ] Add uncertainty intervals

### v0.4 — Regimes & early warning
- [ ] Define interpretable market states
- [ ] Compare rule-based vs unsupervised regimes
- [ ] Model regime transitions
- [ ] Measure alert precision, false positives, and lead time

### v0.5 — Product dashboard
- [ ] Executive market monitor
- [ ] Market explorer
- [ ] Metro deep dive
- [ ] Drivers / explanations
- [ ] Alerts
- [ ] Model health

### v1.0 — Portfolio release
- [ ] Final model card
- [ ] Failure analysis
- [ ] Limitations
- [ ] Deployment
- [ ] Portfolio case study
- [ ] 60–90 second product demo

## Scientific guardrails

This project deliberately separates three different claims:

1. **Descriptive:** what changed in the observed data?
2. **Predictive:** which signals improve out-of-time forecasts?
3. **Causal:** did a specific external intervention cause a change?

Feature importance is not treated as causal evidence. Causal analysis, if added, will use a separate identification strategy with explicit assumptions and robustness checks.

## Evaluation principles

A model is useful only if it beats reasonable alternatives. All models will be compared with simple baselines such as persistence, trailing momentum, and seasonal naive forecasts.

Random train/test splitting is not used for forecasting. Evaluation is temporal and out-of-time.

Performance will be analyzed globally and by:
- metro
- region
- volatility cohort
- market regime
- historical period

## Intended use

This is a research and portfolio project using public aggregate market data. It is not a Zillow internal system, does not use private Zillow product telemetry, and should not be interpreted as individualized financial or real-estate advice.

## Reproduce the v0.1 data foundation

```bash
pip install -r requirements.txt
python -m src.ingestion.download
python -m src.transformation.build_interim
python -m src.transformation.build_mart
python -m src.validation.audit
```

Or run:

```bash
bash scripts/build_v01_data.sh
```

The raw downloads are not committed to Git. Each run records source URL, timestamp, file size, and SHA-256 hash in the local download manifest.

## Current status

**v0.1 Data Foundation: COMPLETE.**  
**v0.2 Market Diagnostics: COMPLETE.**

The live market-diagnostics pipeline now covers cross-market trends, seasonality, Zillow size-rank cohorts, synchronized-shift candidates, pooled lead-lag screening, and a stronger within-month / within-market correlation decomposition.

Key empirical result: recent ZHVI momentum is the dominant baseline, while ZORI momentum is the strongest non-price signal that remains meaningful when metros are compared within the same month. Inventory, price cuts, and Market Heat appear more regime-sensitive than purely cross-sectional.

See [the final data audit](docs/data_audit_2026-09-18.md) and [market diagnostics findings](docs/market_diagnostics_2026-09-18.md).

Next: **v0.3 Forecasting** — build the leakage-safe feature table, establish naive and autoregressive baselines, then compare Elastic Net and gradient boosting under rolling-origin validation.
