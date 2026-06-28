#!/usr/bin/env bash
set -euo pipefail

# Shared HTCondor entrypoint for training jobs in this repository.
# The submit files pass site-specific paths first, then ordinary Hydra
# overrides exactly as they would appear after `python gabbro/train.py`.
# Worker nodes for this setup are expected to have conda, but not uv, so Python
# is launched through `conda run -p "${CONDA_ENV}"`.
if [ "$#" -lt 5 ]; then
  cat >&2 <<'USAGE'
Usage:
  condor_run_training.sh PROJECT_DIR OUTPUT_DIR MODE SUITE_ID JOB_INDEX [HYDRA_OVERRIDES...]

MODE is one of:
  train    Run gabbro/train.py with the supplied Hydra overrides.
  collect  Run scripts/collect_orbit_multirun.py for OUTPUT_DIR/SUITE_ID.
USAGE
  exit 2
fi

PROJECT_DIR="$1"
OUTPUT_DIR="$2"
MODE="$3"
SUITE_ID="$4"
JOB_INDEX="$5"
shift 5

# Default to the canonical shared Condor conda environment path on EOS. The
# /eos/user/... spelling may resolve differently on worker nodes; use the path
# where Python packages are installed.
CONDA_ENV="${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}"

# Keep all job artifacts for a submitted suite below one directory. Hydra uses
# LOG_DIR for run/multirun outputs, while Condor writes stdout/stderr/user logs
# to OUTPUT_DIR/condor_logs from the submit file.
RUN_ROOT="${OUTPUT_DIR}/${SUITE_ID}"
CONDOR_LOG_DIR="${OUTPUT_DIR}/condor_logs"
HYDRA_PROJECT_NAME="orbit-particle-ggHbb-minbias"

mkdir -p "${RUN_ROOT}" "${CONDOR_LOG_DIR}"

# Batch workers may have read-only or slow home directories. Point Matplotlib
# and W&B at the shared run directory unless the submit file explicitly set
# different locations.
export LOG_DIR="${RUN_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p \
  "${MPLCONFIGDIR}" \
  "${WANDB_DIR}" \
  "${RUN_ROOT}/${HYDRA_PROJECT_NAME}/runs" \
  "${RUN_ROOT}/${HYDRA_PROJECT_NAME}/multiruns"

# gabbro/train.py loads GABBRO_ENV_FILE before Hydra composes the config. Fail
# early here so missing credential/env files are obvious in the Condor stderr.
if [ -n "${GABBRO_ENV_FILE:-}" ] && [ ! -f "${GABBRO_ENV_FILE}" ]; then
  echo "GABBRO_ENV_FILE is set but does not exist: ${GABBRO_ENV_FILE}" >&2
  exit 2
fi

if [ -n "${GABBRO_ENV_FILE:-}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${GABBRO_ENV_FILE}"
  set +a
fi

if [ ! -d "${CONDA_ENV}" ]; then
  echo "CONDA_ENV does not exist or is not a directory: ${CONDA_ENV}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Echo the resolved runtime context into the Condor stdout log. This makes
# failed jobs much easier to reproduce from the command line.
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Mode: ${MODE}"
echo "Suite: ${SUITE_ID}"
echo "Job index: ${JOB_INDEX}"
echo "LOG_DIR: ${LOG_DIR}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "WANDB_DIR: ${WANDB_DIR}"
echo "WANDB_MODE: ${WANDB_MODE:-<unset>}"
echo "WANDB_ENTITY: ${WANDB_ENTITY:-<unset>}"
echo "WANDB_API_KEY set: $([ -n "${WANDB_API_KEY:-}" ] && echo yes || echo no)"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "Extra arguments: $*"

run_in_conda() {
  conda run --no-capture-output -p "${CONDA_ENV}" "$@"
}

# Verify the exact Python that will run the job and fail before launching a long
# training command if the environment is missing a core dependency.
run_in_conda python -c 'import sys, pyrootutils; print(f"Python executable: {sys.executable}"); print(f"pyrootutils: {pyrootutils.__file__}")'

case "${MODE}" in
  train)
    # Hydra overrides are passed through unchanged.
    run_in_conda python gabbro/train.py "$@"
    ;;
  collect)
    # Optional collection helper. If a multirun path is provided as the first
    # extra argument, use it; otherwise assume the default Hydra layout under
    # this suite's LOG_DIR.
    MULTIRUN_DIR="${RUN_ROOT}/${HYDRA_PROJECT_NAME}/multiruns/${JOB_INDEX}"
    if [ "$#" -gt 0 ]; then
      MULTIRUN_DIR="$1"
      shift
    fi
    run_in_conda python scripts/collect_orbit_multirun.py \
      --multirun-dir "${MULTIRUN_DIR}" \
      "$@"
    ;;
  *)
    echo "Unknown MODE: ${MODE}" >&2
    exit 2
    ;;
esac
