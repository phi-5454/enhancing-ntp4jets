#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 PROJECT_DIR OUTPUT_DIR BENCHMARK MANIFEST RUN_ARGS..." >&2
  exit 2
fi

PROJECT_DIR="$1"
OUTPUT_DIR="$2"
BENCHMARK="$3"
MANIFEST="$4"
shift 4

CONDA_ENV="${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

case "${BENCHMARK}" in
  higgs)
    SCRIPT="scripts/evaluate_orbit_higgs_mass.py"
    MANIFEST_ARGUMENT="--gghbb-test-manifest"
    ;;
  z_mumu)
    SCRIPT="scripts/evaluate_orbit_z_mumu_mass.py"
    MANIFEST_ARGUMENT="--dyjets-test-manifest"
    ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}" >&2
    exit 2
    ;;
esac

conda run --no-capture-output -p "${CONDA_ENV}" python "${SCRIPT}" \
  "${MANIFEST_ARGUMENT}" "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" --events "${MASS_BENCHMARK_EVENTS:-10000}" \
  --bootstrap-replicas "${MASS_BENCHMARK_BOOTSTRAP_REPLICAS:-10}" --device cuda "$@"
