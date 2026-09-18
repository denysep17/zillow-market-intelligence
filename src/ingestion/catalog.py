from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_source_catalog(path: str | Path = "config/data_sources.yaml") -> dict[str, Any]:
    """Load and validate the configured Zillow Research source registry."""
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    if not catalog or "sources" not in catalog:
        raise ValueError("Source catalog must define a top-level 'sources' mapping")

    required = {"name", "metric", "url", "geography", "frequency", "value_type", "role"}
    for key, source in catalog["sources"].items():
        missing = required - set(source)
        if missing:
            raise ValueError(f"Source '{key}' missing fields: {sorted(missing)}")
        if source["geography"] != "metro":
            raise ValueError(f"Source '{key}' must use metro geography for v0.1")
        if source["frequency"] != "monthly":
            raise ValueError(f"Source '{key}' must be monthly for v0.1")

    return catalog
