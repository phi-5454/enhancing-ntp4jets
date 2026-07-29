#!/usr/bin/env bash
set -euo pipefail

# End-to-end, small downstream-fidelity run for a tokenizer checkpoint.  The
# test loader in PairedOrbitDataModule evaluates both original and decoded
# representations, using the same 10k paired events.
if [ "$#" -ne 6 ]; then
  cat >&2 <<'USAGE'
Usage:
  condor_run_downstream_classifier_smoke.sh PROJECT_DIR OUTPUT_DIR SUITE_ID JOB_INDEX TOKENIZER_RUN MANIFEST_DIR
USAGE
  exit 2
fi

PROJECT_DIR="$1"
OUTPUT_DIR="$2"
SUITE_ID="$3"
JOB_INDEX="$4"
TOKENIZER_RUN="$5"
MANIFEST_DIR="$6"

CONDA_ENV="${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}"
RUN_ROOT="${OUTPUT_DIR}/${SUITE_ID}"
PAIRED_DIR="${RUN_ROOT}/paired_events"
CLASSIFIER_DIR="${RUN_ROOT}/classifier_original_seed_12345"
CONDOR_LOG_DIR="${OUTPUT_DIR}/condor_logs"

mkdir_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if mkdir -p "$@"; then
      return 0
    fi
    echo "mkdir -p failed on attempt ${attempt}/5 for: $*" >&2
    sleep $((attempt * 10))
  done
  mkdir -p "$@"
}

mkdir_retry "${RUN_ROOT}" "${CONDOR_LOG_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/matplotlib"

if [ ! -d "${CONDA_ENV}" ]; then
  echo "CONDA_ENV does not exist or is not a directory: ${CONDA_ENV}" >&2
  exit 2
fi
if [ ! -f "${TOKENIZER_RUN}/checkpoints/best.ckpt" ]; then
  echo "Tokenizer checkpoint does not exist: ${TOKENIZER_RUN}/checkpoints/best.ckpt" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/vqtorch:${PROJECT_DIR}:${PYTHONPATH:-}"
export LOG_DIR="${RUN_ROOT}/logs"
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib"
export ORBIT_DOWNSTREAM_DIR="${PAIRED_DIR}"
export PYTHONDONTWRITEBYTECODE=1

run_in_conda() {
  conda run --no-capture-output -p "${CONDA_ENV}" "$@"
}

echo "Host: $(hostname)"
echo "Tokenizer run: ${TOKENIZER_RUN}"
echo "Manifest directory: ${MANIFEST_DIR}"
echo "Paired data: ${PAIRED_DIR}"
echo "Classifier run: ${CLASSIFIER_DIR}"

# `957029_5` has no PID inputs.  The explicit flag makes original and decoded
# samples equally kinematics-only; it is harmless only for this intended run.
run_in_conda python scripts/export_orbit_downstream_events.py \
  --run-dir "${TOKENIZER_RUN}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-dir "${PAIRED_DIR}" \
  --allow-no-pid \
  --train-events 10000 \
  --val-events 10000 \
  --test-events 10000 \
  --events-per-shard 10000 \
  --device cuda

# Train once on original events.  Test always includes the paired original and
# decoded loaders, yielding a direct reconstruction-fidelity comparison.
run_in_conda python gabbro/train.py \
  experiment=orbit_downstream_event_classifier \
  data.train_representation=original \
  data.num_workers=4 \
  seed=12345 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  hydra.run.dir="${CLASSIFIER_DIR}"
