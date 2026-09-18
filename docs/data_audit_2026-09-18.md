# Final Data Audit — 2026-09-18

## Executive summary

The v0.1 data foundation has been executed end-to-end against the live Zillow Research CSV endpoints.

The canonical modeling panel is now explicitly **metro-only**. Zillow's Metro files also contain a U.S. aggregate row; that row is excluded from the modeling mart and should be retained separately only for benchmark visualizations.

Final canonical mart:

- **286,080 market-month rows**
- **894 metro areas**
- **2000-01-31 to 2026-08-31** overall
- **0 duplicate market-month keys**
- **37,172 complete seven-signal rows**
- **683 metros represented in the complete-case panel**
- **2018-03-31 to 2026-08-31** common seven-signal window

The seven-signal core is:

1. ZHVI
2. ZORI
3. For-sale inventory
4. New listings
5. Share of listings with a price cut
6. Mean days to pending
7. Market Heat Index

## Source coverage

| Metric | First month | Last month | Metros | Observations | ZHVI overlap |
|---|---:|---:|---:|---:|---:|
| ZHVI | 2000-01 | 2026-08 | 894 | 236,814 | 100.0% |
| ZORI | 2015-01 | 2026-08 | 733 | 52,444 | 22.1% |
| Inventory | 2018-03 | 2026-08 | 894 | 91,040 | 38.4% |
| New listings | 2018-03 | 2026-08 | 891 | 88,164 | 37.2% |
| Price-cut share | 2018-03 | 2026-08 | 894 | 90,592 | 38.2% |
| Days to pending | 2018-03 | 2026-08 | 756 | 49,649 | 21.0% |
| Market Heat Index | 2018-01 | 2026-08 | 894 | 91,759 | 38.7% |

The null rates in the union mart are intentionally not interpreted as source-quality failure: the mart begins in 2000 because of ZHVI, while most operational market signals begin in 2018. Coverage should therefore be evaluated inside each metric's active historical window.

## Discontinuities

Internal missing-month diagnostics:

| Metric | Metros with internal gaps | Missing internal months |
|---|---:|---:|
| ZHVI | 169 | 432 |
| ZORI | 127 | 169 |
| Inventory | 35 | 65 |
| New listings | 52 | 131 |
| Price-cut share | 3 | 3 |
| Days to pending | 4 | 9 |
| Market Heat Index | 200 | 894 |

Implication: price-cut share and days-to-pending are particularly contiguous once available. Market Heat Index and ZHVI require explicit missing-period handling and sensitivity checks.

## Extreme-value audit

The initial run exposed an important schema issue: the Zillow Metro files contain a national aggregate row. Once the mart was restricted to `region_type == "msa"`, inventory and new-listing extremes became economically plausible for metro-level data.

Selected final ranges:

- ZHVI: $48,084 to $1,601,701
- ZORI: $526 to $4,228
- Inventory: 5 to 97,260
- New listings: 3 to 25,154
- Price-cut share: 0.9% to 48.2%
- Days to pending: 7 to 258 days
- Market Heat Index: -120 to 279

These are screening ranges, not automatic outlier deletions. Large metros can legitimately occupy distribution tails.

## Signal decision

### Promote to core: price-cut share

**Decision: add.**

Reasons:
- 894 metros
- current through 2026-08
- almost full continuity once available
- distinct behavioral meaning: seller pressure
- directly aligned with the turning-point hypothesis
- minimal complete-case cost when added to the original five-signal panel

### Promote to core: new listings

**Decision: add.**

Reasons:
- 891 metros
- current through 2026-08
- distinct from inventory stock because it measures supply flow
- useful for detecting changes in seller participation before price indices move
- broad coverage and minimal incremental panel loss

### Keep optional: sale-to-list ratio

**Decision: do not make core yet.**

Observed:
- 684 metros
- 2018-03 to 2026-07
- 44,933 observations
- 18.7% overlap with the full ZHVI history
- 35.0% missingness within its source-shaped panel

It is analytically relevant, but it narrows geography and is one month less current than the core signals. It should be tested in an ablation/secondary model rather than defining the primary modeling universe.

### Keep contextual: affordability income needed

**Decision: do not make core predictive input yet.**

Observed:
- 390 metros
- 2012-01 to 2026-08
- 68,446 observations
- excellent continuity within its available source
- materially narrower geography than the operational listing signals

Affordability is useful for segmentation, explanatory context, and robustness analysis. Because it is partly constructed from home values, interest rates, taxes, insurance, and income assumptions, it should not be added mechanically to the main predictive model without checking target-proximity and interpretation.

## Is the original five-signal set enough?

Yes for an initial model, but not for the strongest version of the product.

The live audit shows that **price-cut share and new listings add two economically distinct leading-signal families with almost no complete-case penalty**. They therefore improve the product's ability to distinguish:

- supply stock vs. supply flow
- seller pressure vs. simple inventory growth
- liquidity slowdown vs. price movement

The canonical core is therefore expanded from five to seven signals.

## Temporal alignment and leakage

### Passed

- zero duplicate `market_id × month` keys
- no explicitly future-dated feature columns in the mart
- all core features share a common available window beginning 2018-03
- for a three-month forward target, the latest fully labeled feature month is **2026-05-31** when the latest source month is 2026-08-31

### Modeling rule

For every observation at month `t`:

- features must be computed only from values dated `<= t`
- target is `ZHVI(t+3) / ZHVI(t) - 1`
- rolling features must be backward-looking
- validation must be rolling-origin / expanding-window, never random shuffle

### Remaining leakage risk: historical revisions

The current Zillow download gives the latest revised historical series. Zillow may revise historical values over time. A backtest run on today's full history can therefore use values that were not exactly the same as the values available to an analyst at the original historical prediction date.

This is **vintage/revision leakage**, not row-level leakage.

For the portfolio version:
- disclose it explicitly
- keep the backtest leakage-safe at the row/timestamp level
- optionally archive future monthly source snapshots to support true vintage evaluation going forward

## Final v0.1 gate

**Status: PASS**

Data Foundation is complete enough to move into v0.2 Market Diagnostics / EDA.

The next phase should operate on the 2018-03 onward seven-signal research window while retaining the longer ZHVI/ZORI history for context, seasonality, and baseline analysis.
