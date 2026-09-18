# v0.3 Forecasting Plan

## Target

Three-month forward metro-level ZHVI growth:

`ZHVI(t+3) / ZHVI(t) - 1`

## Core question

Do the additional Zillow market signals improve genuinely out-of-time forecasts beyond recent home-value momentum?

## Models

The first experiment deliberately limits model complexity:

1. no-change baseline
2. seasonal naive baseline
3. trailing 3-month momentum baseline
4. price-only Ridge autoregression
5. full-feature Elastic Net
6. full-feature histogram gradient boosting

No model is promoted merely because it is more complex.

## Validation design

Use expanding rolling-origin validation with:

- 36-month minimum training history
- 6-month non-overlapping validation blocks
- 3-month target-horizon purge between the training feature window and each validation block

The purge is important. Without it, a historical backtest could train on rows whose three-month outcome would not yet have been observable at the simulated prediction date.

## Feature families

### Price-only baseline

- 1M / 3M / 6M / 12M ZHVI momentum
- 12M rolling ZHVI volatility
- calendar seasonality
- Zillow size-rank proxy

### Incremental market signals

- ZORI momentum
- inventory growth
- new-listing growth
- price-cut change
- days-to-pending change
- Market Heat change
- same-month cross-sectional percentile features

## Missing data

Non-price features are imputed from the **training fold only**. Missingness indicators are added by the sklearn pipeline. This keeps validation information out of preprocessing while retaining metros with incomplete non-price coverage.

## Evaluation

Primary:
- MAE

Secondary:
- RMSE
- bias
- directional accuracy
- fold stability
- Zillow size-rank cohort performance

## Decision rule

A full-feature model must demonstrate out-of-time improvement over both:

- trailing momentum
- price-only Ridge

If it does not, the additional signals are not yet justified as predictive inputs.

## Next after the first run

- inspect fold reversals and stress periods
- feature ablation
- prediction intervals / conformal coverage
- error analysis by market regime
- only then select the model that feeds the early-warning product
