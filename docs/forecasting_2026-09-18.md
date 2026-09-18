# Forecasting Findings — 2026-09-18

## Executive result

The first leakage-safe forecasting experiment is complete.

Using **42,910 out-of-time metro-month predictions** across **8 purged rolling-origin folds**, the regularized full-feature Elastic Net achieves the lowest aggregate error:

- **MAE: 0.871 percentage points**
- **RMSE: 1.182 percentage points**
- **Directional accuracy: 77.36%**
- **Bias: +0.263 percentage points**

Relative to the baselines, this is:

- **33.4% lower MAE than trailing 3-month momentum**
- **13.8% lower MAE than the price-only Ridge model**

The more complex histogram gradient-boosting model does **not** improve on Elastic Net, so nonlinear complexity is not justified at this stage.

## Experimental design

Target:

`ZHVI(t+3) / ZHVI(t) - 1`

Modeling window:

**2019-03-31 to 2026-05-31**

Universe:

- 77,752 labeled metro-month rows
- 894 metros
- 42,910 validation predictions

Validation:

- 8 non-overlapping rolling-origin folds
- 36-month minimum historical training window
- 6-month validation blocks
- 3-month target-horizon purge
- no random shuffle
- preprocessing fitted only on each training fold

The purge matters because a training observation at month t cannot be used if its t+3 outcome would not yet have been known at the simulated validation date.

## Aggregate benchmark comparison

| Model | MAE | RMSE | Bias | Directional accuracy |
|---|---:|---:|---:|---:|
| No change | 1.188 pp | 1.581 pp | -0.498 pp | — |
| Seasonal naive | 1.793 pp | 2.553 pp | +0.626 pp | 63.23% |
| Trailing momentum | 1.307 pp | 1.831 pp | +0.170 pp | 69.77% |
| Price-only Ridge | 1.010 pp | 1.376 pp | +0.613 pp | 76.44% |
| Full Elastic Net | **0.871 pp** | **1.182 pp** | +0.263 pp | **77.36%** |
| Full gradient boosting | 0.907 pp | 1.275 pp | +0.347 pp | 76.72% |

The main result is not that a more complicated algorithm wins. It is the opposite: **a regularized linear model with well-designed market features beats the nonlinear alternative**.

## Feature-family ablation

The out-of-time ablation is more informative than the earlier correlation screen.

| Feature set | MAE | Directional accuracy |
|---|---:|---:|
| Full without rent | **0.871 pp** | 77.37% |
| Full | 0.871 pp | 77.36% |
| Full without supply | 0.895 pp | 77.53% |
| Price + liquidity | 0.895 pp | 77.51% |
| Price + supply | 0.909 pp | 76.51% |
| Full without liquidity | 0.911 pp | 76.52% |
| Price + rent | 1.005 pp | 76.46% |
| Price only | 1.009 pp | 76.32% |

### Interpretation

The earlier EDA found rent momentum to be the strongest non-price **cross-sectional correlation** with future home-value momentum. The forecasting experiment shows that this does not translate into meaningful incremental out-of-time lift once price, supply, and liquidity information is already present.

That distinction is important:

- **Rent:** informative descriptively, but largely redundant in the current predictive specification.
- **Liquidity features:** strongest direct incremental family over price-only.
- **Supply features:** also materially improve the price-only model.
- **Full model:** gains additional lift by combining supply and liquidity.
- **New listings / individual variables:** should still be evaluated later within the family rather than assumed useful.

This is exactly why feature selection should be driven by out-of-time evidence rather than full-sample correlation.

## Temporal instability

No model wins every historical validation block.

| Fold | Validation period | Lowest-MAE model | MAE |
|---:|---|---|---:|
| 1 | 2022-06 to 2022-11 | No change | 1.232 pp |
| 2 | 2022-12 to 2023-05 | Price-only Ridge | 0.979 pp |
| 3 | 2023-06 to 2023-11 | Full Elastic Net | 0.807 pp |
| 4 | 2023-12 to 2024-05 | Full Elastic Net | 0.740 pp |
| 5 | 2024-06 to 2024-11 | Price-only Ridge | 0.624 pp |
| 6 | 2024-12 to 2025-05 | No change | 0.957 pp |
| 7 | 2025-06 to 2025-11 | Price-only Ridge | 0.577 pp |
| 8 | 2025-12 to 2026-05 | Gradient boosting | 0.574 pp |

Winner counts:

- Ridge: 3 folds
- No-change baseline: 2 folds
- Elastic Net: 2 folds
- Gradient boosting: 1 fold

This is a strong empirical argument for the next project layer: **market regime detection**. The system should not assume that one forecasting rule dominates under every housing-market state.

## Error heterogeneity by metro size

| Zillow size cohort | Ridge MAE | Elastic Net MAE | Elastic directional accuracy |
|---|---:|---:|---:|
| Top 100 | 0.934 pp | **0.724 pp** | 78.00% |
| Ranks 101–300 | 0.858 pp | **0.691 pp** | 79.18% |
| Rank 301+ | 1.071 pp | **0.952 pp** | 76.67% |

Smaller Zillow-ranked metros are harder to forecast in this setup. A single global error metric therefore understates product uncertainty for that cohort.

## Prediction intervals

A time-ordered empirical conformal-style layer was evaluated around the Elastic Net point forecast.

### Global interval calibration

Nominal target: 90%

| Scope | Coverage | Mean width |
|---|---:|---:|
| Overall | 90.78% | 3.844 pp |
| Top 100 | 96.28% | 3.844 pp |
| Ranks 101–300 | 96.62% | 3.844 pp |
| Rank 301+ | 88.01% | 3.844 pp |

The global interval has the right aggregate coverage but allocates uncertainty poorly: it is too wide for larger markets and too narrow for smaller markets.

### Size-cohort conditional calibration

| Scope | Coverage | Mean width |
|---|---:|---:|
| Overall | 90.53% | 3.879 pp |
| Top 100 | 89.15% | 3.182 pp |
| Ranks 101–300 | 90.17% | 3.091 pp |
| Rank 301+ | 90.86% | 4.245 pp |

The cohort-conditional version adapts interval width to observed error heterogeneity. Smaller metros receive wider intervals; larger metros receive narrower intervals.

This is materially more useful for a decision-support product than presenting one confidence band for every market.

These are empirical coverage results, not a formal iid conformal guarantee, because metro-month observations are dependent across both geography and time.

## Product implication

The product should not display a forecast as a single deterministic number.

A metro-level output should eventually look conceptually like:

`3M forecast + uncertainty interval + current regime + driver explanation + reliability flag`

The forecasting layer now provides the first two components.

## Model decision

For the next stage:

- **Elastic Net remains the primary point-model candidate.**
- **Gradient boosting stays as a challenger, not the default.**
- **Cohort-conditional empirical intervals are preferred to one global interval.**
- **Regime-awareness is required before production-style model promotion.**

The purpose of v0.4 is therefore not to add a more complicated forecaster. It is to explain **when different market dynamics apply** and use those states for early-warning detection.

## Remaining limitations

- Zillow historical revisions can create vintage-data differences in retrospective evaluation.
- Same-month features assume they are available by the scoring cutoff; true publication-lag sensitivity still needs an explicit test.
- Metro observations are cross-sectionally dependent.
- The feature-family ablation identifies predictive contribution, not causal effect.
- The first experiment uses fixed hyperparameters intentionally; no nested tuning claim is made.

## v0.3 conclusion

**PASS.**

The forecasting system demonstrates genuine out-of-time lift over price-only and naive baselines while exposing meaningful temporal and geographic instability.

That instability is now evidence for, rather than an excuse to skip, **v0.4 Regimes & Early Warning**.
