# Data layer

This project uses public, aggregate Zillow Research datasets.

## Directory contract

- `raw/` — immutable source downloads exactly as published.
- `interim/` — normalized long-form tables and intermediate joins.
- `processed/` — leakage-safe analytical marts and model-ready feature tables.

Large/raw data files are intentionally excluded from Git.

## Required source metadata

Every ingested dataset should record:
- source name
- source URL
- download timestamp
- geography
- temporal frequency
- metric definition
- earliest/latest available period
- known methodology notes

## Canonical grain

The primary analytical grain is:

`market_id × month`

The final feature mart should contain one row per market-month with only information available at that prediction timestamp.
