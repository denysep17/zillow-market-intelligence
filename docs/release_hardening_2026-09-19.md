# Release Hardening Findings — 2026-09-19

## Executive result

The v1.0 analytical release gate passes, with explicit operating constraints.

The selected Elastic Net forecast remains strongest under the original contemporaneous-feature assumption, but performance degrades as the information set becomes stale.

## Publication-lag sensitivity

| Scenario | MAE | Directional accuracy | MAE degradation |
|---|---:|---:|---:|
| Contemporaneous | 0.871 pp | 77.4% | 0.0% |
| Non-price lag 1M | 0.928 pp | 76.5% | +6.6% |
| Non-price lag 2M | 0.989 pp | 76.2% | +13.5% |
| All dynamic lag 1M | 1.143 pp | 70.4% | +31.2% |

These are scenario-based sensitivity tests, not claims about Zillow's actual publication SLA.

The model is reasonably robust to a one-month delay in non-price signals, but materially weaker when the full dynamic information set is one month stale.

## Forecast failure modes

Error is not uniform.

- highest-volatility quartile: **1.276 pp MAE**
- cooling regime: **0.989 pp MAE**
- rank-301+ metros: **0.952 pp MAE**
- worst validation fold: **1.347 pp MAE**

Worst metros with at least 20 validation observations:

| Metro | State | MAE | Bias |
|---|---|---:|---:|
| Greenville | MS | 3.518 pp | +1.501 pp |
| Clarksdale | MS | 3.455 pp | +0.942 pp |
| Murray | KY | 3.050 pp | +1.105 pp |
| Bennettsville | SC | 2.756 pp | +1.301 pp |
| Indianola | MS | 2.605 pp | +0.797 pp |

The global 0.871 pp MAE should not be presented as a universal market-level error rate.

## Early-warning failure modes

For the precision-oriented threshold policy:

- precision: **51.4%**
- recall: **19.8%**
- realized cooling-entry events missed: **80.2%**
- false-alert share: **48.6%**

The model has useful ranking signal, but the thresholded policy is intentionally selective and should only be used to prioritize analyst attention.

## Release decision

**PASS for research / portfolio deployment.**

Not approved for a production claim.

Before a true production implementation, the system would need:

- point-in-time historical source vintages
- source-specific publication timestamps
- drift monitoring
- model and threshold versioning
- operational alert ownership