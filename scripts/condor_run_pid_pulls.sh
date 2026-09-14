#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 PROJECT_DIR OUTPUT_DIR RUN_DIR TEST_MANIFEST" >&2
  exit 2
fi

PROJECT_DIR="$1"
OUTPUT_DIR="$2"
RUN_DIR="$3"
TEST_MANIFEST="$4"
CONDA_ENV="${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

conda run --no-capture-output -p "${CONDA_ENV}" \
  python scripts/evaluate_orbit_pid_pulls.py \
  --run-dir "${RUN_DIR}" \
  --test-manifest "${TEST_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --events "${PID_PULL_EVENTS:-2000}" \
  --device cuda \
  --wandb \
  --wandb-name "${PID_PULL_WANDB_NAME:-$(basename "${OUTPUT_DIR}")}" \
  --wandb-group "${PID_PULL_WANDB_GROUP:-orbit-pid-conditional-residuals}"
