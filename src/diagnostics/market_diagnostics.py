from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = [
    "zhvi",
    "zori",
    "inventory",
    "new_listings",
    "price_cut_share",
    "days_to_pending",
    "market_heat",
]

RESEARCH_START = pd.Timestamp("2018-03-31")


def _pct_change(
    df: pd.DataFrame,
    column: str,
    periods: int,
) -> pd.Series:
    lagged = df.groupby("market_id")[column].shift(periods)
    return df[column] / lagged - 1.0


def _point_change(
    df: pd.DataFrame,
    column: str,
    periods: int,
) -> pd.Series:
    lagged = df.groupby("market_id")[column].shift(periods)
    return df[column] - lagged


def _forward_growth(
    df: pd.DataFrame,
    column: str,
    horizon: int,
) -> pd.Series:
    future = df.groupby("market_id")[column].shift(-horizon)
    return future / df[column] - 1.0


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["market_id", "month"]).copy()

    for periods in (1, 3, 12):
        out[f"zhvi_growth_{periods}m"] = _pct_change(out, "zhvi", periods)

    out["zori_growth_3m"] = _pct_change(out, "zori", 3)
    out["zori_growth_12m"] = _pct_change(out, "zori", 12)
    out["inventory_growth_3m"] = _pct_change(out, "inventory", 3)
    out["inventory_growth_12m"] = _pct_change(out, "inventory", 12)
    out["new_listings_growth_3m"] = _pct_change(out, "new_listings", 3)
    out["new_listings_growth_12m"] = _pct_change(out, "new_listings", 12)
    out["price_cut_change_3m"] = _point_change(out, "price_cut_share", 3)
    out["price_cut_change_12m"] = _point_change(out, "price_cut_share", 12)
    out["days_pending_change_3m"] = _point_change(out, "days_to_pending", 3)
    out["days_pending_change_12m"] = _point_change(out, "days_to_pending", 12)
    out["market_heat_change_3m"] = _point_change(out, "market_heat", 3)

    for horizon in (1, 3, 6):
        out[f"zhvi_forward_growth_{horizon}m"] = _forward_growth(
            out,
            "zhvi",
            horizon,
        )

    monthly_return = out["zhvi_growth_1m"]
    out["zhvi_volatility_12m"] = (
        monthly_return.groupby(out["market_id"])
        .rolling(12, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    return out


def _spearman_pair(frame: pd.DataFrame, x: str, y: str) -> dict[str, object]:
    pair = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 30:
        return {"feature": x, "n": int(len(pair)), "rho": None}

    ranked_x = pair[x].rank(method="average")
    ranked_y = pair[y].rank(method="average")
    rho = ranked_x.corr(ranked_y)
    return {"feature": x, "n": int(len(pair)), "rho": float(rho)}


def lead_lag_table(df: pd.DataFrame) -> list[dict[str, object]]:
    features = [
        "zhvi_growth_3m",
        "zori_growth_3m",
        "inventory_growth_3m",
        "new_listings_growth_3m",
        "price_cut_change_3m",
        "days_pending_change_3m",
        "market_heat_change_3m",
    ]
    rows: list[dict[str, object]] = []

    for horizon in (1, 3, 6):
        target = f"zhvi_forward_growth_{horizon}m"
        for feature in features:
            result = _spearman_pair(df, feature, target)
            result["horizon_months"] = horizon
            rows.append(result)

    return rows


def _size_cohort(size_rank: pd.Series) -> pd.Series:
    conditions = [
        size_rank.le(100),
        size_rank.between(101, 300),
        size_rank.gt(300),
    ]
    values = ["top_100", "rank_101_300", "rank_301_plus"]
    return pd.Series(
        np.select(conditions, values, default="unknown"),
        index=size_rank.index,
    )


def cohort_snapshot(df: pd.DataFrame, latest_month: pd.Timestamp) -> list[dict[str, object]]:
    snap = df.loc[df["month"].eq(latest_month)].copy()
    snap["size_cohort"] = _size_cohort(snap["size_rank"])

    metrics = [
        "zhvi_growth_3m",
        "zhvi_growth_12m",
        "inventory_growth_12m",
        "new_listings_growth_12m",
        "price_cut_change_12m",
        "days_pending_change_12m",
    ]
    rows: list[dict[str, object]] = []

    for cohort, frame in snap.groupby("size_cohort"):
        row: dict[str, object] = {
            "cohort": str(cohort),
            "markets": int(frame["market_id"].nunique()),
        }
        for metric in metrics:
            row[f"median_{metric}"] = _safe_median(frame[metric])
        rows.append(row)

    return rows


def _safe_median(series: pd.Series) -> float | None:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return float(clean.median())


def latest_snapshot(df: pd.DataFrame) -> dict[str, object]:
    latest_month = pd.to_datetime(df["month"]).max()
    snap = df.loc[df["month"].eq(latest_month)].copy()

    return {
        "month": latest_month.date().isoformat(),
        "markets": int(snap["market_id"].nunique()),
        "median_zhvi_growth_3m": _safe_median(snap["zhvi_growth_3m"]),
        "median_zhvi_growth_12m": _safe_median(snap["zhvi_growth_12m"]),
        "median_zori_growth_12m": _safe_median(snap["zori_growth_12m"]),
        "median_inventory_growth_12m": _safe_median(
            snap["inventory_growth_12m"]
        ),
        "median_new_listings_growth_12m": _safe_median(
            snap["new_listings_growth_12m"]
        ),
        "median_price_cut_share": _safe_median(snap["price_cut_share"]),
        "median_price_cut_change_12m": _safe_median(
            snap["price_cut_change_12m"]
        ),
        "median_days_to_pending": _safe_median(snap["days_to_pending"]),
        "median_days_pending_change_12m": _safe_median(
            snap["days_pending_change_12m"]
        ),
        "median_market_heat": _safe_median(snap["market_heat"]),
    }


def _market_wide_monthly(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zhvi",
        "zori",
        "inventory",
        "new_listings",
        "price_cut_share",
        "days_to_pending",
        "market_heat",
    ]
    monthly = (
        df.groupby("month")[columns]
        .median(numeric_only=True)
        .sort_index()
        .reset_index()
    )
    return monthly


def seasonality_table(df: pd.DataFrame) -> list[dict[str, object]]:
    seasonal = df.loc[df["month"].dt.year >= 2019].copy()
    seasonal["calendar_month"] = seasonal["month"].dt.month

    seasonal["inventory_growth_1m"] = _pct_change(
        seasonal,
        "inventory",
        1,
    )
    seasonal["new_listings_growth_1m"] = _pct_change(
        seasonal,
        "new_listings",
        1,
    )
    seasonal["price_cut_change_1m"] = _point_change(
        seasonal,
        "price_cut_share",
        1,
    )

    metrics = [
        "inventory_growth_1m",
        "new_listings_growth_1m",
        "price_cut_change_1m",
    ]
    rows: list[dict[str, object]] = []

    for month, frame in seasonal.groupby("calendar_month"):
        row: dict[str, object] = {"calendar_month": int(month)}
        for metric in metrics:
            row[f"median_{metric}"] = _safe_median(frame[metric])
        rows.append(row)

    return rows


def inflection_candidates(df: pd.DataFrame) -> list[dict[str, object]]:
    monthly = _market_wide_monthly(df)
    monthly = monthly.loc[monthly["month"].ge(RESEARCH_START)].copy()

    for metric in ["zhvi", "inventory", "new_listings", "market_heat"]:
        previous = monthly[metric].shift(1)
        if metric == "market_heat":
            monthly[f"{metric}_move"] = monthly[metric] - previous
        else:
            monthly[f"{metric}_move"] = monthly[metric] / previous - 1.0

        move = monthly[f"{metric}_move"]
        med = move.median()
        mad = (move - med).abs().median()
        scale = 1.4826 * mad if mad and not np.isnan(mad) else move.std()
        monthly[f"{metric}_robust_z"] = (move - med) / scale

    z_columns = [
        "zhvi_robust_z",
        "inventory_robust_z",
        "new_listings_robust_z",
        "market_heat_robust_z",
    ]
    monthly["composite_shift"] = monthly[z_columns].abs().mean(axis=1)
    top = monthly.nlargest(8, "composite_shift")

    rows: list[dict[str, object]] = []
    for _, row in top.iterrows():
        rows.append(
            {
                "month": pd.Timestamp(row["month"]).date().isoformat(),
                "composite_shift": float(row["composite_shift"]),
                "zhvi_move": _finite_or_none(row["zhvi_move"]),
                "inventory_move": _finite_or_none(row["inventory_move"]),
                "new_listings_move": _finite_or_none(
                    row["new_listings_move"]
                ),
                "market_heat_move": _finite_or_none(
                    row["market_heat_move"]
                ),
            }
        )
    return rows


def _finite_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _top_100_snapshot(
    df: pd.DataFrame,
    latest_month: pd.Timestamp,
) -> pd.DataFrame:
    snap = df.loc[
        df["month"].eq(latest_month) & df["size_rank"].le(100)
    ].copy()
    required = [
        "market_name",
        "state_name",
        "inventory_growth_12m",
        "zhvi_growth_3m",
        "price_cut_share",
    ]
    return snap.dropna(subset=required).sort_values("size_rank")


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    escaped_title = escape(title)
    escaped_subtitle = escape(subtitle)
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#F7F8FA"/>',
        (
            '<text x="60" y="55" font-family="Inter,Arial,sans-serif" '
            'font-size="30" font-weight="700" fill="#111827">'
            f"{escaped_title}</text>"
        ),
        (
            '<text x="60" y="84" font-family="Inter,Arial,sans-serif" '
            'font-size="15" fill="#667085">'
            f"{escaped_subtitle}</text>"
        ),
    ]


def _svg_footer(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines)


def _scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) * (b - a) / (hi - lo)


def write_signal_overview_svg(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    monthly = _market_wide_monthly(df)
    monthly = monthly.loc[monthly["month"].ge(RESEARCH_START)].copy()
    series = {
        "ZHVI": ("zhvi", "#006AFF"),
        "ZORI": ("zori", "#7A5AF8"),
        "Inventory": ("inventory", "#00A699"),
        "New listings": ("new_listings", "#F79009"),
    }

    indexed: dict[str, pd.Series] = {}
    for label, (column, _) in series.items():
        values = monthly[column]
        first_valid = values.first_valid_index()
        if first_valid is None:
            continue
        base = values.loc[first_valid]
        indexed[label] = values / base * 100.0

    width, height = 1200, 660
    left, right = 80, 1140
    top, bottom = 125, 575

    values = pd.concat(indexed.values()).dropna()
    y_lo = float(values.quantile(0.01))
    y_hi = float(values.quantile(0.99))
    padding = max((y_hi - y_lo) * 0.08, 5)
    y_lo -= padding
    y_hi += padding

    lines = _svg_header(
        width,
        height,
        "Cross-market signal trajectories",
        "Median across U.S. metros, indexed to 100 at each series' first "
        "available month in the research window.",
    )

    for grid in np.linspace(y_lo, y_hi, 5):
        y = _scale(float(grid), y_lo, y_hi, bottom, top)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#E4E7EC" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            'font-family="Inter,Arial,sans-serif" font-size="12" '
            f'fill="#667085">{grid:.0f}</text>'
        )

    n = len(monthly)
    for label, (_, color) in series.items():
        if label not in indexed:
            continue
        points: list[str] = []
        for idx, value in enumerate(indexed[label]):
            if pd.isna(value):
                continue
            x = _scale(idx, 0, max(n - 1, 1), left, right)
            y = _scale(float(value), y_lo, y_hi, bottom, top)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    legend_x = 90
    for label, (_, color) in series.items():
        lines.append(
            f'<circle cx="{legend_x}" cy="615" r="5" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 12}" y="620" '
            'font-family="Inter,Arial,sans-serif" font-size="13" '
            f'fill="#344054">{escape(label)}</text>'
        )
        legend_x += 170

    start_label = monthly["month"].min().strftime("%Y")
    end_label = monthly["month"].max().strftime("%Y")
    lines.append(
        f'<text x="{left}" y="598" font-family="Inter,Arial,sans-serif" '
        f'font-size="12" fill="#667085">{start_label}</text>'
    )
    lines.append(
        f'<text x="{right}" y="598" text-anchor="end" '
        'font-family="Inter,Arial,sans-serif" font-size="12" '
        f'fill="#667085">{end_label}</text>'
    )

    Path(path).write_text(_svg_footer(lines), encoding="utf-8")


def write_lead_lag_svg(
    lead_lag: list[dict[str, object]],
    path: str | Path,
) -> None:
    rows = [
        row
        for row in lead_lag
        if row["horizon_months"] == 3 and row["rho"] is not None
    ]
    rows = sorted(rows, key=lambda row: abs(float(row["rho"])), reverse=True)

    labels = {
        "zhvi_growth_3m": "Price momentum",
        "zori_growth_3m": "Rent momentum",
        "inventory_growth_3m": "Inventory growth",
        "new_listings_growth_3m": "New-listing growth",
        "price_cut_change_3m": "Price-cut change",
        "days_pending_change_3m": "Days-pending change",
        "market_heat_change_3m": "Market-heat change",
    }

    width, height = 1200, 650
    center, max_bar = 690, 360
    top, row_gap = 145, 62
    max_abs = max(abs(float(row["rho"])) for row in rows)
    max_abs = max(max_abs, 0.05)

    lines = _svg_header(
        width,
        height,
        "Which signals move with future 3-month home-value momentum?",
        "Exploratory pooled Spearman association. Predictive association is "
        "not causal evidence.",
    )
    lines.append(
        f'<line x1="{center}" y1="125" x2="{center}" y2="590" '
        'stroke="#98A2B3" stroke-width="1.5"/>'
    )

    for idx, row in enumerate(rows):
        rho = float(row["rho"])
        y = top + idx * row_gap
        length = abs(rho) / max_abs * max_bar
        x = center if rho >= 0 else center - length
        color = "#006AFF" if rho >= 0 else "#D92D20"
        label = labels.get(str(row["feature"]), str(row["feature"]))

        lines.append(
            f'<text x="60" y="{y + 5}" font-family="Inter,Arial,sans-serif" '
            f'font-size="15" fill="#344054">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{x:.1f}" y="{y - 13}" width="{length:.1f}" '
            f'height="26" rx="5" fill="{color}" opacity="0.88"/>'
        )
        value_x = center + 8 if rho >= 0 else center - 8
        anchor = "start" if rho >= 0 else "end"
        lines.append(
            f'<text x="{value_x}" y="{y + 5}" text-anchor="{anchor}" '
            'font-family="Inter,Arial,sans-serif" font-size="13" '
            f'font-weight="600" fill="#101828">{rho:+.3f}</text>'
        )

    lines.append(
        '<text x="330" y="620" font-family="Inter,Arial,sans-serif" '
        'font-size="12" fill="#667085">Negative association</text>'
    )
    lines.append(
        '<text x="850" y="620" font-family="Inter,Arial,sans-serif" '
        'font-size="12" fill="#667085">Positive association</text>'
    )
    Path(path).write_text(_svg_footer(lines), encoding="utf-8")


def write_market_matrix_svg(
    df: pd.DataFrame,
    latest_month: pd.Timestamp,
    path: str | Path,
) -> None:
    snap = _top_100_snapshot(df, latest_month)
    if snap.empty:
        return

    x_col = "inventory_growth_12m"
    y_col = "zhvi_growth_3m"
    x_low, x_high = snap[x_col].quantile([0.02, 0.98])
    y_low, y_high = snap[y_col].quantile([0.02, 0.98])

    width, height = 1200, 700
    left, right = 110, 1130
    top, bottom = 135, 610

    lines = _svg_header(
        width,
        height,
        "Top-100 metro market matrix",
        f"Inventory YoY vs. 3-month ZHVI momentum, {latest_month:%B %Y}. "
        "Points are major Zillow metros by size rank.",
    )

    zero_x = _scale(0, float(x_low), float(x_high), left, right)
    zero_y = _scale(0, float(y_low), float(y_high), bottom, top)
    lines.append(
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" '
        f'y2="{bottom}" stroke="#D0D5DD" stroke-dasharray="5 5"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{right}" '
        f'y2="{zero_y:.1f}" stroke="#D0D5DD" stroke-dasharray="5 5"/>'
    )

    clipped = snap.loc[
        snap[x_col].between(x_low, x_high)
        & snap[y_col].between(y_low, y_high)
    ].copy()

    x_center = clipped[x_col].median()
    y_center = clipped[y_col].median()
    x_scale = max(clipped[x_col].std(), 1e-9)
    y_scale = max(clipped[y_col].std(), 1e-9)
    clipped["label_score"] = np.sqrt(
        ((clipped[x_col] - x_center) / x_scale) ** 2
        + ((clipped[y_col] - y_center) / y_scale) ** 2
    )
    label_ids = set(clipped.nlargest(8, "label_score").index)

    for idx, row in clipped.iterrows():
        x = _scale(float(row[x_col]), float(x_low), float(x_high), left, right)
        y = _scale(float(row[y_col]), float(y_low), float(y_high), bottom, top)
        share = float(row["price_cut_share"])
        radius = 5 + min(max(share, 0.0), 0.4) * 12
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            'fill="#006AFF" opacity="0.42" stroke="#0056CC" '
            'stroke-width="0.8"/>'
        )

        if idx in label_ids:
            label = escape(str(row["market_name"]).split(",")[0])
            lines.append(
                f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" '
                'font-family="Inter,Arial,sans-serif" font-size="12" '
                f'font-weight="600" fill="#344054">{label}</text>'
            )

    lines.append(
        f'<text x="{(left + right) / 2:.1f}" y="670" text-anchor="middle" '
        'font-family="Inter,Arial,sans-serif" font-size="13" '
        'fill="#475467">Inventory growth, 12 months →</text>'
    )
    lines.append(
        '<text x="25" y="380" transform="rotate(-90 25 380)" '
        'font-family="Inter,Arial,sans-serif" font-size="13" '
        'fill="#475467">3-month ZHVI momentum →</text>'
    )

    Path(path).write_text(_svg_footer(lines), encoding="utf-8")


def _market_extremes(
    df: pd.DataFrame,
    latest_month: pd.Timestamp,
) -> dict[str, list[dict[str, object]]]:
    snap = _top_100_snapshot(df, latest_month)
    columns = [
        "market_name",
        "state_name",
        "size_rank",
        "zhvi_growth_3m",
        "inventory_growth_12m",
        "price_cut_share",
    ]
    hottest = snap.nlargest(5, "zhvi_growth_3m")[columns]
    coolest = snap.nsmallest(5, "zhvi_growth_3m")[columns]

    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            out.append(
                {
                    "market_name": str(row["market_name"]),
                    "state_name": str(row["state_name"]),
                    "size_rank": int(row["size_rank"]),
                    "zhvi_growth_3m": float(row["zhvi_growth_3m"]),
                    "inventory_growth_12m": float(
                        row["inventory_growth_12m"]
                    ),
                    "price_cut_share": float(row["price_cut_share"]),
                }
            )
        return out

    return {"strongest_3m": records(hottest), "weakest_3m": records(coolest)}


def build_markdown_report(report: dict[str, object]) -> str:
    snapshot = report["latest_snapshot"]
    lead = pd.DataFrame(report["lead_lag"])
    lead_3m = lead.loc[lead["horizon_months"].eq(3)].dropna(subset=["rho"])
    lead_3m["abs_rho"] = lead_3m["rho"].abs()
    strongest = lead_3m.sort_values("abs_rho", ascending=False).iloc[0]

    lines = [
        "# Market Diagnostics — live Zillow data",
        "",
        "## Scope",
        "",
        (
            f"Research window: **{report['research_window']['start']} to "
            f"{report['research_window']['end']}**. The canonical mart remains "
            "metro-only."
        ),
        "",
        "## Latest cross-market snapshot",
        "",
        (
            f"As of **{snapshot['month']}**, median 12-month ZHVI growth across "
            f"available metros is **{snapshot['median_zhvi_growth_12m']:.1%}**; "
            f"median 12-month inventory growth is "
            f"**{snapshot['median_inventory_growth_12m']:.1%}**."
        ),
        "",
        (
            f"Median price-cut share is **{snapshot['median_price_cut_share']:.1%}** "
            f"and median days to pending is "
            f"**{snapshot['median_days_to_pending']:.0f} days**."
        ),
        "",
        "## Lead-lag screen",
        "",
        (
            "The strongest pooled exploratory association with forward "
            f"3-month ZHVI growth is **{strongest['feature']}** "
            f"(Spearman rho = **{strongest['rho']:+.3f}**, "
            f"n = {int(strongest['n']):,})."
        ),
        "",
        (
            "These correlations are used to prioritize hypotheses for formal "
            "out-of-time modeling. They are not causal estimates."
        ),
        "",
        "## Inflection candidates",
        "",
        (
            "The report also flags months with unusually large synchronized "
            "moves across cross-market median ZHVI, inventory, new listings, "
            "and Market Heat. These are candidate regime-change periods for "
            "deeper structural-break analysis, not pre-labeled events."
        ),
        "",
        "## Next analytical gate",
        "",
        (
            "Use these diagnostics to lock the feature hypotheses, define metro "
            "cohorts, and specify the baseline forecasting experiment before "
            "training any nonlinear model."
        ),
    ]
    return "\n".join(lines) + "\n"


def run_market_diagnostics(
    mart_path: str | Path = "data/processed/mart_market_monthly.parquet",
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    df = pd.read_parquet(mart_path)
    df["month"] = pd.to_datetime(df["month"])
    df = add_research_features(df)

    research = df.loc[df["month"].ge(RESEARCH_START)].copy()
    latest_month = research["month"].max()

    report: dict[str, object] = {
        "research_window": {
            "start": RESEARCH_START.date().isoformat(),
            "end": latest_month.date().isoformat(),
            "rows": int(len(research)),
            "markets": int(research["market_id"].nunique()),
        },
        "latest_snapshot": latest_snapshot(research),
        "cohorts": cohort_snapshot(research, latest_month),
        "lead_lag": lead_lag_table(research),
        "seasonality": seasonality_table(research),
        "inflection_candidates": inflection_candidates(research),
        "top_100_market_extremes": _market_extremes(
            research,
            latest_month,
        ),
    }

    output = Path(output_dir)
    reports = output / "reports"
    figures = output / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    (reports / "market_diagnostics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (reports / "market_diagnostics.md").write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    write_signal_overview_svg(
        research,
        figures / "market_signal_overview.svg",
    )
    write_lead_lag_svg(
        report["lead_lag"],
        figures / "lead_lag_3m.svg",
    )
    write_market_matrix_svg(
        research,
        latest_month,
        figures / "latest_market_matrix.svg",
    )

    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v0.2 Zillow market diagnostics."
    )
    parser.add_argument(
        "--mart",
        default="data/processed/mart_market_monthly.parquet",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_market_diagnostics(args.mart, args.output_dir)


if __name__ == "__main__":
    main()
