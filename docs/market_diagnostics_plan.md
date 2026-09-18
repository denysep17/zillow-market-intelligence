# v0.2 Market Diagnostics

## Objective

Before model training, establish how the Zillow market signals behave across time,
metro size, and market conditions.

The diagnostic layer answers four questions:

1. What changed?
2. Which signals move together?
3. Which signals appear to precede future ZHVI momentum?
4. Where are relationships unstable enough to require cohort- or regime-aware modeling?

## Research window

The primary multi-signal research window begins in **March 2018**, when the
core supply, seller-pressure, and velocity series overlap.

Longer ZHVI and ZORI history may still be used for context and baseline work.

## Core diagnostics

- cross-market median signal trajectories
- latest metro snapshot
- Zillow size-rank cohorts
- seasonality
- internal volatility
- exploratory lead-lag associations at 1, 3, and 6 months
- candidate market-wide inflection periods
- top-100 metro matrix

## Statistical guardrails

- pooled correlations are exploratory, not causal
- no random train/test splitting
- no feature is accepted for modeling only because its correlation is large
- market-size heterogeneity is surfaced rather than averaged away
- extreme values are inspected before any trimming
- historical Zillow revisions remain a documented vintage-data limitation

## Visual design

The first portfolio-quality figures are generated as lightweight SVG assets so
they render directly on GitHub and can later be reused in the portfolio site.

Planned assets:

- cross-market indexed signal trajectories
- 3-month lead-lag association screen
- major-metro inventory-vs-price-momentum matrix

## Exit criteria

v0.2 is complete when:

- the live diagnostics workflow passes
- the key findings are documented from the real mart
- the initial feature hypotheses are locked
- the baseline modeling experiment is specified
- the visuals are promoted into the README / case study
