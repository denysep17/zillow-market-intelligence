# Methodology

## Research question

Can leading housing-market signals identify turning points before headline home-value measures fully reflect them?

## Primary target

Three-month forward ZHVI growth:

`target_t = ZHVI_(t+3) / ZHVI_t - 1`

## Baselines

Before any complex ML model, evaluate:
- persistence / no-change
- trailing momentum
- seasonal naive
- autoregressive linear baseline

## Candidate features

- 1/3/6/12 month ZHVI momentum
- ZORI momentum
- inventory growth and acceleration
- price-cut changes
- days-to-pending changes
- sale-to-list dynamics
- volatility
- seasonal deviation
- market-relative z-scores / percentiles

## Validation

Use expanding-window or rolling-origin validation. No random train/test split.

## Evaluation

- MAE
- RMSE
- directional accuracy
- bias
- performance by metro
- performance by regime
- stability across historical periods

## Interpretation

Feature importance is predictive evidence only. It is not treated as causal evidence.

## Causal extension

Any causal analysis must define:
- treatment
- comparison group
- identifying assumption
- pre-trend diagnostics
- placebo / robustness checks
