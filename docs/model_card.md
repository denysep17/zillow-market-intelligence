# Model card

## Status
Not trained yet.

## Intended use
Support aggregate U.S. housing-market monitoring and research.

## Not intended for
- individualized investment advice
- mortgage underwriting
- property-level valuation
- causal claims without a separate identification strategy

## Target
Three-month forward metro-level ZHVI growth.

## Training data
Public Zillow Research aggregate market data. Exact datasets and date ranges will be versioned here once locked.

## Validation
Rolling-origin / expanding-window out-of-time validation.

## Metrics
MAE, RMSE, directional accuracy, bias, regime-specific performance.

## Known risks to evaluate
- data revisions
- publication lag
- missingness
- structural breaks
- geographic heterogeneity
- extreme low-liquidity markets
- drift after macroeconomic regime changes
