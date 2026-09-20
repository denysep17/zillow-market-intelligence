#!/usr/bin/env bash
set -euo pipefail

echo "Preparing Zillow Market Intelligence dashboard..."

if [ ! -f "data/processed/mart_market_monthly.parquet" ]; then
  echo "Canonical mart not found. Building from current Zillow Research releases..."
  python -m src.ingestion.download
  python -m src.transformation.build_interim
  python -m src.transformation.build_mart
fi

echo "Starting Streamlit..."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec streamlit run dashboard/app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
