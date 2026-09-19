from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

UNSUPERVISED_FEATURES = [
    "zhvi_growth_3m",
    "inventory_growth_3m",
    "price_cut_change_3m",
    "days_pending_change_3m",
    "market_heat_change_3m",
]


def fit_unsupervised_regimes(
    df: pd.DataFrame,
    n_clusters: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    usable = df.loc[:, UNSUPERVISED_FEATURES].copy()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "cluster",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=20,
                    random_state=42,
                ),
            ),
        ]
    )

    labels = pipeline.fit_predict(usable)
    transformed = pipeline[:-1].transform(usable)

    out = df.copy()
    out["cluster_regime"] = [f"cluster_{label}" for label in labels]

    sample_size = min(len(out), 20_000)
    sample = out.sample(sample_size, random_state=42).index
    sample_positions = out.index.get_indexer(sample)
    score = silhouette_score(
        transformed[sample_positions],
        labels[sample_positions],
    )

    diagnostics = {
        "n_clusters": float(n_clusters),
        "silhouette_score_sample": float(score),
        "sample_size": float(sample_size),
    }
    return out, diagnostics
