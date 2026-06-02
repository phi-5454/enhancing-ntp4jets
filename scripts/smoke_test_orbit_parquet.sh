#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PARQUET_FILE="${PARQUET_FILE:-${REPO_DIR}/../test_data/VBFHbb-NEVENT10000-RS35000001.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/enhancing-ntp4jets-uv-smoke}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/enhancing-mplconfig}"

cd "${REPO_DIR}"

run_smoke_test() {
  local experiment="$1"
  echo "Running ${experiment}"
  LOG_DIR="${OUTPUT_DIR}" \
  PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  uv run --locked python gabbro/train.py \
    "experiment=${experiment}" \
    "data.parquet_files_train=[${PARQUET_FILE}]" \
    "data.parquet_files_val=[${PARQUET_FILE}]" \
    "data.parquet_files_test=[${PARQUET_FILE}]"
}

run_smoke_test orbit_parquet_smoke
run_smoke_test orbit_jet_parquet_smoke
