#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 PAIRED_DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PAIRED_DATA_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath -m "$2")
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${REPO_DIR}"
export ORBIT_DOWNSTREAM_DIR="${PAIRED_DATA_DIR}"
export LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

for seed in 0 1 2 3 4; do
  for representation in original decoded; do
    run_dir="${OUTPUT_DIR}/${representation}/seed_${seed}"
    echo "Training seed=${seed} representation=${representation} -> ${run_dir}"
    "${PYTHON_BIN}" gabbro/train.py \
      experiment=orbit_downstream_event_classifier_full_event \
      data.train_representation="${representation}" \
      seed="${seed}" \
      hydra.run.dir="${run_dir}"
  done
done

original_runs=()
decoded_runs=()
for seed in 0 1 2 3 4; do
  original_runs+=("${OUTPUT_DIR}/original/seed_${seed}")
  decoded_runs+=("${OUTPUT_DIR}/decoded/seed_${seed}")
done

"${PYTHON_BIN}" scripts/aggregate_orbit_classifier_matrix.py \
  --original-runs "${original_runs[@]}" \
  --decoded-runs "${decoded_runs[@]}" \
  --output-dir "${OUTPUT_DIR}/matrix"

