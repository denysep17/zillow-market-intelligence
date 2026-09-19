from __future__ import annotations

import numpy as np
import pandas as pd


def regime_distribution(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    counts = (
        df.groupby(label_col)
        .size()
        .rename("rows")
        .reset_index()
    )
    counts["share"] = counts["rows"] / counts["rows"].sum()
    return counts.sort_values("share", ascending=False).reset_index(drop=True)


def transition_table(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    ordered = df.sort_values(["market_id", "month"]).copy()
    ordered["next_regime"] = ordered.groupby("market_id")[label_col].shift(-1)
    transitions = ordered.dropna(subset=[label_col, "next_regime"]).copy()

    table = pd.crosstab(
        transitions[label_col],
        transitions["next_regime"],
        normalize="index",
    )
    return table


def persistence_metrics(
    df: pd.DataFrame,
    label_col: str,
) -> dict[str, float]:
    ordered = df.sort_values(["market_id", "month"]).copy()
    ordered["next_regime"] = ordered.groupby("market_id")[label_col].shift(-1)
    valid = ordered.dropna(subset=[label_col, "next_regime"]).copy()

    same = valid[label_col].eq(valid["next_regime"])
    run_lengths: list[int] = []

    for _, group in ordered.groupby("market_id", sort=False):
        labels = group[label_col].astype(str).to_numpy()
        if len(labels) == 0:
            continue
        run = 1
        for idx in range(1, len(labels)):
            if labels[idx] == labels[idx - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

    return {
        "one_month_persistence": float(same.mean()) if len(valid) else np.nan,
        "median_run_months": float(np.median(run_lengths)) if run_lengths else np.nan,
        "p90_run_months": float(np.quantile(run_lengths, 0.90)) if run_lengths else np.nan,
    }


def transition_entropy(
    transition_matrix: pd.DataFrame,
) -> pd.Series:
    rows: dict[str, float] = {}
    for state, row in transition_matrix.iterrows():
        p = row.to_numpy(dtype=float)
        p = p[p > 0]
        rows[str(state)] = float(-(p * np.log2(p)).sum())
    return pd.Series(rows, name="transition_entropy_bits")
