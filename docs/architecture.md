# Architecture

```text
Public Zillow Research datasets
            |
            v
      Raw ingestion
            |
            v
   Schema / QA validation
            |
            v
 Canonical market-month tables
            |
            v
    Feature engineering
            |
       +----+----+
       |         |
       v         v
Diagnostics   Forecasting
Regimes       Uncertainty
       |         |
       +----+----+
            |
            v
Interpretation + anomaly detection
            |
            v
   Analyst attention layer
            |
            v
 Interactive dashboard
            |
            v
     Model monitoring
```

## Design principles

1. Raw data is immutable.
2. Production logic lives in `src/`, not notebooks.
3. Features are leakage-safe by construction.
4. Forecast evaluation is strictly temporal.
5. Descriptive, predictive, and causal claims are separated.
6. Model complexity must outperform simple baselines.
7. Dashboard outputs expose uncertainty and failure modes.
