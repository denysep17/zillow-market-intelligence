from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.ingestion.catalog import load_source_catalog


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(
    key: str,
    url: str,
    output_dir: Path,
    *,
    timeout: int = 60,
) -> dict[str, str | int]:
    """Download one source atomically and return provenance metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{key}.csv"
    temp_path = output_dir / f".{key}.csv.part"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    temp_path.write_bytes(response.content)
    temp_path.replace(final_path)

    return {
        "source_key": key,
        "url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": final_path.stat().st_size,
        "sha256": sha256_file(final_path),
        "path": str(final_path),
    }


def download_all(
    catalog_path: str | Path = "config/data_sources.yaml",
    output_dir: str | Path = "data/raw",
) -> list[dict[str, str | int]]:
    catalog = load_source_catalog(catalog_path)
    output = Path(output_dir)
    manifests: list[dict[str, str | int]] = []

    for key, source in catalog["sources"].items():
        manifests.append(download_source(key, source["url"], output))

    manifest_path = output / "_download_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured Zillow Research datasets.")
    parser.add_argument("--catalog", default="config/data_sources.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    manifests = download_all(args.catalog, args.output_dir)
    for item in manifests:
        print(
            f"{item['source_key']}: {item['bytes']} bytes "
            f"sha256={str(item['sha256'])[:12]}..."
        )


if __name__ == "__main__":
    main()
