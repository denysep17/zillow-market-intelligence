# Model card

## Status

**v1.0 analytical release candidate — evaluated, stress-tested, and integrated into the dashboard**

Primary point-model candidate: **Elastic Net**

Challenger: histogram gradient boosting.

The system is a research and portfolio decision-support prototype, not a production Zillow system. Temporal validation, uncertainty calibration, regime analysis, alert-policy evaluation, publication-lag stress testing, and formal failure analysis are complete.

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

## Publication-lag stress test

The original forecasting experiment assumes same-month features are available by the scoring cutoff. v1.0 explicitly stress-tests that assumption.

These scenarios are sensitivity tests; they do not claim Zillow's actual publication SLA.

| Scenario | MAE | Directional accuracy | MAE degradation vs current |
|---|---:|---:|---:|
| Contemporaneous | 0.871 pp | 77.4% | 0.0% |
| Non-price signals lagged 1M | 0.928 pp | 76.5% | +6.6% |
| Non-price signals lagged 2M | 0.989 pp | 76.2% | +13.5% |
| All dynamic market signals lagged 1M | 1.143 pp | 70.4% | +31.2% |

The system is reasonably robust to a one-month delay in non-price signals, but forecast quality degrades materially when the full dynamic information set is one month stale.

## Forecast failure analysis

Observed high-error contexts:

- **high-volatility quartile:** 1.276 pp MAE
- **cooling regime:** 0.989 pp MAE
- **rank-301+ metros:** 0.952 pp MAE
- **worst validation fold:** 1.347 pp MAE

Worst metros with at least 20 validation observations include Greenville, MS; Clarksdale, MS; Murray, KY; Bennettsville, SC; and Indianola, MS. Several small markets have MAE above 2.5 pp, materially above the global result.

## Early-warning failure analysis

For the precision-oriented policy:

- **80.2%** of realized cooling-entry events are missed
- **48.6%** of emitted alerts are false positives

The policy is therefore suitable for selective prioritization only. The transition probability should not be interpreted as a deterministic warning.

## Known limitations

- Zillow may revise historical observations, creating vintage-data differences in retrospective evaluation.
- Publication-lag sensitivity is scenario-tested, but source-specific availability timestamps and SLAs are not modeled.
- Metro-month observations are cross-sectionally and temporally dependent.
- Smaller Zillow-ranked metros have materially higher forecast error.
- Performance varies across historical periods.
- No causal interpretation should be made from coefficients, importance, or ablation.
- Hyperparameters were intentionally fixed for the first benchmark experiment rather than tuned on the full sample.

## Release position

The project is suitable as a reproducible research and portfolio decision-support system.

A true production implementation would still require:

1. archived source vintages or point-in-time snapshots;
2. explicit publication timestamps and source-level availability SLAs;
3. automated feature/data drift monitoring;
4. model-version and threshold-version tracking;
5. operational ownership for alert review and threshold changes.
