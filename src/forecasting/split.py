from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    months: pd.Series,
    min_train_months: int = 36,
    validation_months: int = 12,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield train-end, validation-start, validation-end timestamps.

    This split generator is deterministic and preserves temporal ordering.
    """
    unique_months = pd.Index(pd.to_datetime(months).dropna().unique()).sort_values()
    start = min_train_months

    while start + validation_months <= len(unique_months):
        train_end = unique_months[start - 1]
        validation_start = unique_months[start]
        validation_end = unique_months[start + validation_months - 1]
        yield train_end, validation_start, validation_end
        start += validation_months
