# v0.4 Regimes & Early Warning

## Why this phase exists

v0.3 showed that no forecasting model wins every historical validation period.

That instability is a product signal: the system should identify **which market state is active** before treating one forecasting rule as universally reliable.

## Stage A — regime discovery

Compare:

1. transparent rule-based market states
2. an unsupervised KMeans comparator

Candidate interpretable states:

- accelerating
- balanced
- tightening
- loosening
- cooling

The rule system uses same-month market-relative percentiles of:

- 3M ZHVI momentum
- inventory growth
- price-cut change
- days-to-pending change
- Market Heat change

## Evaluation criteria

A regime system is useful only if it has:

- interpretable economic meaning
- enough observations per state
- non-trivial persistence
- distinct forward outcome distributions
- understandable transition paths
- practical usefulness for alerts

A high silhouette score alone is not sufficient.

## Stage B — early warning

After state definitions are validated, construct transition targets such as:

`Balanced/Tightening at t → Cooling within the next 1–3 months`

Evaluate:

- alert precision
- recall
- false-alert rate
- lead time
- calibration
- performance by metro-size cohort
- performance through historical stress periods

## Product output

The eventual market card should include:

`current regime + transition risk + forecast + interval + drivers + reliability`

This phase supplies the regime and transition-risk components.
