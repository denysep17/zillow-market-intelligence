# Live dashboard deployment

The production-style portfolio deployment uses Railway.

## Service command

Railway runs:

    bash scripts/start_dashboard.sh

The startup script:

1. checks whether the canonical metro-month mart exists;
2. downloads the current public Zillow Research source files when needed;
3. builds interim normalized tables;
4. builds the canonical metro-only mart;
5. starts Streamlit on Railway's assigned PORT.

## Health check

Streamlit health endpoint:

    /_stcore/health

## Deployment source

Repository:

    denysep17/zillow-market-intelligence

Branch:

    main

## Important behavior

The deployment intentionally rebuilds from current public Zillow releases when the processed mart is absent. This keeps the portfolio dashboard reproducible and avoids committing large raw datasets to Git.

Because Zillow can revise historical series, a redeployment may reflect a newer public data vintage than an earlier run.
