from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.catalog import load_source_catalog
from src.transformation.zillow_wide import normalize_csv


def build_interim(
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    catalog_path: str | Path = "config/data_sources.yaml",
) -> None:
    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    catalog = load_source_catalog(catalog_path)
    for key, source in catalog["sources"].items():
        input_path = raw_path / f"{key}.csv"
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing raw source: {input_path}. Run the ingestion step first."
            )

        normalized = normalize_csv(input_path, metric=source["metric"])
        output_path = interim_path / f"{key}.parquet"
        normalized.to_parquet(output_path, index=False)
        print(f"{key}: {len(normalized):,} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Zillow source CSVs.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--catalog", default="config/data_sources.yaml")
    args = parser.parse_args()
    build_interim(args.raw_dir, args.interim_dir, args.catalog)


if __name__ == "__main__":
    main()
