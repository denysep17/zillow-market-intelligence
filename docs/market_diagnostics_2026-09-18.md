# Market Diagnostics Findings — 2026-09-18

## Scope

The live v0.2 diagnostics run uses the metro-only Zillow panel from **2018-03-31 through 2026-08-31**, covering **894 metros**. The purpose of this phase is descriptive and predictive screening before any model is trained.

## Latest cross-market snapshot

As of August 2026, the median across available metros is:

| Metric | Median |
|---|---:|
| 3-month ZHVI growth | 0.24% |
| 12-month ZHVI growth | 2.61% |
| 12-month ZORI growth | 3.36% |
| 12-month inventory growth | 5.82% |
| 12-month new-listing growth | 3.46% |
| Price-cut share | 24.97% |
| Days to pending | 52 days |
| Market Heat Index | 45 |

The current market is therefore not well summarized by one headline metric: home values and rents remain positive year over year while inventory has expanded and roughly one quarter of listings show a price cut.

## Metro-size heterogeneity

Latest median 12-month ZHVI growth differs materially by Zillow size rank:

| Cohort | Metros | Median 12M ZHVI | Median 12M inventory |
|---|---:|---:|---:|
| Top 100 | 97 | 1.52% | 4.67% |
| Ranks 101–300 | 194 | 2.17% | 5.29% |
| Rank 301+ | 603 | 3.07% | 5.98% |

This is enough heterogeneity to justify cohort-level error analysis later. A single global MAE could hide systematic failure in larger or smaller markets.

## Lead-lag result

The strongest pooled association with forward 3-month ZHVI growth is recent price momentum itself:

- 3-month ZHVI momentum: pooled Spearman rho = **+0.526**
- n = **88,437** market-month observations

That makes an autoregressive price-only model the required benchmark for v0.3.

However, pooled panel correlations can mix three effects:
1. persistent differences between metros;
2. shared macro time effects;
3. genuine within-market predictive signal.

To separate these, the diagnostic layer now reports pooled, within-month cross-sectional, and within-market correlations.

### Three-month horizon decomposition

| Signal | Pooled rho | Median monthly cross-sectional rho | Median within-market rho |
|---|---:|---:|---:|
| Price momentum | +0.526 | +0.490 | +0.459 |
| Rent momentum | +0.216 | +0.206 | +0.142 |
| Days-pending change | -0.137 | -0.090 | -0.145 |
| Inventory growth | -0.241 | -0.037 | -0.284 |
| New-listing growth | -0.036 | +0.026 | -0.038 |
| Market-heat change | +0.174 | +0.001 | +0.207 |
| Price-cut change | -0.186 | -0.001 | -0.209 |

The important DS conclusion is that **rent momentum is the strongest non-price signal that remains meaningful after comparing metros within the same month**.

Inventory growth, price-cut changes, and Market Heat show a different pattern: they are much stronger within the same metro over time than across metros in the same month. This suggests they are better candidates for **regime / state-transition features** than for simple cross-sectional ranking.

New-listing growth is weak as a standalone linear signal. It stays in the research feature set, but it must earn its place through interactions, regime detection, or actual out-of-time lift.

## Horizon stability

The same broad pattern persists across 1-, 3-, and 6-month horizons:

- Recent price momentum remains dominant.
- Rent momentum remains the most stable non-price cross-sectional signal.
- Inventory growth is consistently negative within markets.
- Price-cut changes and Market Heat behave more like market-state variables than clean cross-sectional predictors.
- Days to pending is consistently negative but weaker.
- New listings is weak standalone.

## Seasonality

The listing-side metrics show strong calendar structure:

- New listings typically rise sharply in spring and contract into year-end.
- Inventory tends to rise from spring into summer and decline late in the year.
- Price-cut share tends to build through summer and fall before declining around year-end.

This means the modeling table should not rely on naive month-over-month levels. Candidate treatments include:

- year-over-year changes;
- calendar-month controls;
- market-relative seasonal deviations;
- seasonally normalized features.

## Candidate inflection windows

The synchronized-shift diagnostic flags the following high-movement periods for later regime analysis:

| Month | Composite shift | ZHVI move | Inventory move | New-listing move | Market Heat move |
|---|---:|---:|---:|---:|---:|
| 2022-01 | 2.02 | +1.19% | -11.21% | -16.48% | +8.5 |
| 2022-05 | 1.75 | +1.23% | +11.68% | +15.89% | -7.0 |
| 2023-01 | 1.66 | -0.22% | -10.97% | -11.76% | +6.0 |
| 2022-02 | 1.65 | +1.24% | -9.98% | -5.26% | +6.5 |
| 2021-05 | 1.64 | +1.51% | +9.40% | +19.61% | -4.0 |

These are **candidate synchronized shifts**, not pre-labeled structural breaks. They will be useful later for regime discovery and model stress tests.

## Current major-metro contrast

Among the top 100 Zillow metros by size rank in August 2026, recent 3-month ZHVI momentum is strongest in markets including Syracuse, Bridgeport, Hartford, Rochester, and Chicago, while weaker examples include Las Vegas, Durham, Lakeland, Austin, and Seattle.

This is descriptive only; the point is the cross-market dispersion and the need for metro-level diagnostics, not a ranking of market quality.

## Feature hypotheses locked for v0.3

1. **Price momentum is the mandatory baseline.** ML must beat a price-only autoregressive model out of time.
2. **Rent momentum is the highest-priority incremental cross-sectional feature.**
3. **Inventory, price cuts, and Market Heat are regime-sensitive features.** Test changes, lags, interactions, and market-relative normalization.
4. **Days to pending is a secondary liquidity feature.**
5. **New listings remains conditional** until it demonstrates incremental lift or regime value.

## Statistical guardrails

- These are descriptive and predictive diagnostics, not causal estimates.
- Feature selection will be based on rolling out-of-time performance, not full-sample correlation.
- Historical Zillow revisions remain a vintage-data limitation.
- Model evaluation must be segmented by metro-size cohort and later by market regime.
- The three-month forward target must only use feature information available at or before month t.

## v0.2 conclusion

**PASS.**

The project now has enough empirical structure to move into **v0.3 Forecasting**.

The next experiment should compare:

`seasonal naive / persistence / trailing momentum / autoregressive price-only / Elastic Net / gradient boosting`

using rolling-origin validation for the 3-month forward ZHVI target.
