from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.model_table import TARGET, build_model_table
from src.regimes.diagnostics import (
    persistence_metrics,
    regime_distribution,
    transition_entropy,
    transition_table,
)
from src.regimes.rules import build_rule_regimes
from src.regimes.unsupervised import fit_unsupervised_regimes


def _forward_outcomes(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in df.groupby(label_col):
        target = frame[TARGET].dropna()
        rows.append(
            {
                "regime": str(label),
                "rows": int(len(frame)),
                "target_rows": int(len(target)),
                "mean_forward_3m_zhvi": float(target.mean()),
                "median_forward_3m_zhvi": float(target.median()),
                "negative_forward_share": float((target < 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_forward_3m_zhvi",
        ascending=False,
    )


def _latest_distribution(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    latest = df["month"].max()
    current = df.loc[df["month"].eq(latest)]
    out = regime_distribution(current, label_col)
    out.insert(0, "month", latest.date().isoformat())
    return out


def build_report(
    rule_distribution: pd.DataFrame,
    rule_latest: pd.DataFrame,
    rule_outcomes: pd.DataFrame,
    rule_persistence: dict[str, float],
    cluster_diagnostics: dict[str, float],
    cluster_persistence: dict[str, float],
) -> str:
    lines = [
        "# Regime Discovery — v0.4",
        "",
        "## Objective",
        "",
        (
            "Test whether interpretable market states are stable enough to "
            "support an early-warning layer, and compare them with a simple "
            "unsupervised alternative."
        ),
        "",
        "## Rule-based state distribution",
        "",
        "| State | Rows | Share |",
        "|---|---:|---:|",
    ]

    for _, row in rule_distribution.iterrows():
        lines.append(
            f"| {row['rule_regime']} | {int(row['rows']):,} | {row['share']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## State persistence",
            "",
            (
                f"Rule-based one-month persistence: "
                f"**{rule_persistence['one_month_persistence']:.1%}**."
            ),
            (
                f"Median rule-state run length: "
                f"**{rule_persistence['median_run_months']:.0f} months**."
            ),
            "",
            "## Forward outcomes by rule state",
            "",
            "| State | Mean forward 3M ZHVI | Median | Negative-forward share |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, row in rule_outcomes.iterrows():
        lines.append(
            (
                f"| {row['regime']} | {row['mean_forward_3m_zhvi']:.2%} | "
                f"{row['median_forward_3m_zhvi']:.2%} | "
                f"{row['negative_forward_share']:.1%} |"
            )
        )

    lines.extend(
        [
            "",
            "## Latest state mix",
            "",
            "| Month | State | Share |",
            "|---|---|---:|",
        ]
    )
    for _, row in rule_latest.iterrows():
        lines.append(
            f"| {row['month']} | {row['rule_regime']} | {row['share']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Unsupervised comparator",
            "",
            (
                f"KMeans silhouette score on a 20k-row sample: "
                f"**{cluster_diagnostics['silhouette_score_sample']:.3f}**."
            ),
            (
                f"Cluster one-month persistence: "
                f"**{cluster_persistence['one_month_persistence']:.1%}**."
            ),
            (
                f"Median cluster run length: "
                f"**{cluster_persistence['median_run_months']:.0f} months**."
            ),
            "",
            (
                "The unsupervised solution is a comparator, not ground truth. "
                "Promotion should depend on interpretability, persistence, "
                "forward-outcome separation, and usefulness for alerts."
            ),
            "",
            "## Next gate",
            "",
            (
                "Define transition targets such as Balanced/Tightening → Cooling, "
                "then measure whether current signals detect those transitions "
                "with useful precision, false-alert rate, and lead time."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_regime_experiment(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    mart = pd.read_parquet(mart_path)
    table = build_model_table(mart)

    regime_data = build_rule_regimes(table)
    regime_data, cluster_diagnostics = fit_unsupervised_regimes(regime_data)

    rule_distribution = regime_distribution(regime_data, "rule_regime")
    rule_latest = _latest_distribution(regime_data, "rule_regime")
    rule_outcomes = _forward_outcomes(regime_data, "rule_regime")
    rule_transition = transition_table(regime_data, "rule_regime")
    rule_persistence = persistence_metrics(regime_data, "rule_regime")
    rule_entropy = transition_entropy(rule_transition)

    cluster_distribution = regime_distribution(
        regime_data,
        "cluster_regime",
    )
    cluster_transition = transition_table(
        regime_data,
        "cluster_regime",
    )
    cluster_persistence = persistence_metrics(
        regime_data,
        "cluster_regime",
    )

    summary = {
        "rows": int(len(regime_data)),
        "markets": int(regime_data["market_id"].nunique()),
        "first_month": regime_data["month"].min().date().isoformat(),
        "last_month": regime_data["month"].max().date().isoformat(),
        "rule_persistence": rule_persistence,
        "rule_transition_entropy_bits": {
            str(index): float(value)
            for index, value in rule_entropy.items()
        },
        "cluster_diagnostics": cluster_diagnostics,
        "cluster_persistence": cluster_persistence,
    }

    output = Path(output_dir)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    regime_data[
        [
            "market_id",
            "market_name",
            "state_name",
            "month",
            TARGET,
            "rule_regime",
            "cluster_regime",
        ]
    ].to_parquet(reports / "regime_assignments.parquet", index=False)

    rule_distribution.to_csv(
        reports / "rule_regime_distribution.csv",
        index=False,
    )
    rule_latest.to_csv(
        reports / "rule_regime_latest.csv",
        index=False,
    )
    rule_outcomes.to_csv(
        reports / "rule_regime_forward_outcomes.csv",
        index=False,
    )
    rule_transition.to_csv(
        reports / "rule_regime_transition_matrix.csv",
    )
    cluster_distribution.to_csv(
        reports / "cluster_regime_distribution.csv",
        index=False,
    )
    cluster_transition.to_csv(
        reports / "cluster_regime_transition_matrix.csv",
    )
    (reports / "regime_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (reports / "regime_report.md").write_text(
        build_report(
            rule_distribution,
            rule_latest,
            rule_outcomes,
            rule_persistence,
            cluster_diagnostics,
            cluster_persistence,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(rule_outcomes.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Zillow market-regime discovery."
    )
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_regime_experiment(args.mart, args.output_dir)


if __name__ == "__main__":
    main()
