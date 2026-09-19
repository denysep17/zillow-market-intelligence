# Model card

## Status

**Trained and evaluated — v0.3 forecasting candidate**

Primary point-model candidate: **Elastic Net**

Challenger: histogram gradient boosting.

The primary model is not yet considered production-ready because regime-aware behavior, publication-lag sensitivity, and monitoring thresholds still need to be added.

## Intended use

Support aggregate U.S. housing-market monitoring and research by estimating near-term metro-level home-value momentum.

The model is designed for a decision-support product that combines:

`forecast + uncertainty + market regime + driver explanation + reliability flag`

## Not intended for

- individualized investment advice
- mortgage underwriting
- property-level valuation
- causal claims
- deterministic market-timing decisions
- production deployment without publication-lag and vintage-data controls

## Target

Three-month forward metro-level ZHVI growth:

`ZHVI(t+3) / ZHVI(t) - 1`

## Training data

Public Zillow Research aggregate metro-level monthly data.

Core signal families:

- ZHVI
- ZORI
- for-sale inventory
- new listings
- price-cut share
- mean days to pending
- Market Heat Index

Modeling window: **2019-03-31 to 2026-05-31**

Evaluation universe:

- 894 metros
- 77,752 labeled market-month rows
- 42,910 out-of-time validation predictions

## Feature families

### Price / autoregressive

- 1M, 3M, 6M, 12M ZHVI momentum
- 12M rolling ZHVI volatility
- calendar seasonality
- Zillow size-rank proxy

### Incremental market signals

- ZORI momentum
- inventory growth
- new-listing growth
- price-cut changes
- days-to-pending changes
- Market Heat changes
- same-month cross-sectional percentile features

## Validation

Purged rolling-origin evaluation:

- 36-month minimum historical training window
- 6-month non-overlapping validation blocks
- 3-month target-horizon purge
- 8 validation folds
- no random train/test split

The purge ensures that historical training rows are only used when their forward outcome would have been observable before the simulated validation period.

Preprocessing and imputation are fit on the training fold only.

## Aggregate results

| Model | MAE | RMSE | Directional accuracy |
|---|---:|---:|---:|
| Seasonal naive | 1.793 pp | 2.553 pp | 63.23% |
| Trailing momentum | 1.307 pp | 1.831 pp | 69.77% |
| Price-only Ridge | 1.010 pp | 1.376 pp | 76.44% |
| Full Elastic Net | **0.871 pp** | **1.182 pp** | **77.36%** |
| Gradient boosting | 0.907 pp | 1.275 pp | 76.72% |

Elastic Net reduces MAE by approximately:

- **33.4% vs. trailing momentum**
- **13.8% vs. price-only Ridge**

Gradient boosting does not outperform Elastic Net, so additional nonlinear complexity is not justified by aggregate error.

## Feature-ablation result

The strongest feature-family result is that **supply and liquidity features contribute most of the incremental out-of-time lift**.

Rent momentum was the strongest non-price cross-sectional relationship in EDA, but removing rent from the full Elastic Net does not materially degrade MAE. This means the rent signal is largely redundant in the current predictive specification.

Predictive contribution is not interpreted as causal contribution.

## Temporal stability

The lowest-MAE model varies materially across validation periods:

- Ridge wins 3 folds
- no-change baseline wins 2 folds
- Elastic Net wins 2 folds
- gradient boosting wins 1 fold

This instability motivates the next regime-aware modeling layer.

## Uncertainty

A time-ordered empirical conformal-style calibration layer is evaluated around Elastic Net.

Global 90% interval:

- empirical coverage: **90.78%**
- mean width: **3.844 pp**

Global intervals overcover larger metros and under-cover rank-301+ metros.

Size-cohort conditional calibration:

| Cohort | Empirical coverage | Mean interval width |
|---|---:|---:|
| Top 100 | 89.15% | 3.182 pp |
| Ranks 101–300 | 90.17% | 3.091 pp |
| Rank 301+ | 90.86% | 4.245 pp |

Because metro-month observations are dependent across time and geography, these are empirical coverage measurements rather than claims of formal iid conformal guarantees.

## Known limitations

- Zillow may revise historical observations, creating vintage-data differences in retrospective evaluation.
- Same-month features assume the signal is available by the scoring cutoff; publication-lag sensitivity still needs explicit evaluation.
- Metro-month observations are cross-sectionally and temporally dependent.
- Smaller Zillow-ranked metros have materially higher forecast error.
- Performance varies across historical periods.
- No causal interpretation should be made from coefficients, importance, or ablation.
- Hyperparameters were intentionally fixed for the first benchmark experiment rather than tuned on the full sample.

## Promotion criteria for the next version

Before the forecasting model is used in the early-warning product layer:

1. define interpretable housing-market regimes;
2. evaluate error and interval coverage by regime;
3. test publication-lag sensitivity;
4. establish alert thresholds and reliability rules;
5. quantify false alerts and lead time;
6. document model-health monitoring behavior.
