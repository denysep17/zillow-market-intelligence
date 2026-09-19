# Product Dashboard

The Streamlit application is the interactive decision-support layer for the Zillow Market Intelligence project.

## Views

1. Executive Monitor — current regime mix, priority metros, forward momentum, and transition risk.
2. Market Explorer — filtered metro comparison across supply, forecast, and risk.
3. Metro Deep Dive — one-market diagnostic view with forecast interval and historical signals.
4. Alerts — precision-oriented cooling-transition alerts and historical policy performance.
5. Model Health — model benchmarks, validation design, uncertainty, and known limitations.

## Run locally

Build the current Zillow mart first.

    python -m src.ingestion.download
    python -m src.transformation.build_interim
    python -m src.transformation.build_mart

Then launch.

    streamlit run dashboard/app.py

## Product principles

- decision-support rather than decorative analytics
- no fake values in current market views
- uncertainty shown alongside point forecasts
- alert thresholds treated as product policy
- model limitations visible in the interface
- reusable Plotly components rather than notebook-only charts
