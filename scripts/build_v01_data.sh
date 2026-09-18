#!/usr/bin/env bash
set -euo pipefail

python -m src.ingestion.download
python -m src.transformation.build_interim
python -m src.transformation.build_mart
python -m src.validation.audit

echo "v0.1 data foundation build complete."
