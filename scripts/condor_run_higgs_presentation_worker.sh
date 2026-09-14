#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 PROJECT_DIR CACHE_DIR MANIFEST INDEX GGHBB_MANIFEST" >&2
  exit 2
fi

PROJECT_DIR=$1
CACHE_DIR=$2
MANIFEST=$3
INDEX=$4
GGHBB_MANIFEST=$5
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}

cd "${PROJECT_DIR}"
RUN_DIR=$("${CONDA_ENV}/bin/python" -c \
  'import json, sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["run_dir"])' \
  "${MANIFEST}" "${INDEX}")
if [ -z "${RUN_DIR}" ] || [ "${RUN_DIR}" = null ]; then
  echo "No run_dir at manifest index ${INDEX}" >&2
  exit 2
fi
RUN_OUTPUT=$(printf '%s/%03d' "${CACHE_DIR}" "${INDEX}")
export ORBIT_MANIFEST_DIR
ORBIT_MANIFEST_DIR=$(dirname -- "${GGHBB_MANIFEST}")

conda run --no-capture-output -p "${CONDA_ENV}" python scripts/evaluate_orbit_higgs_mass.py \
  --run-dir "${RUN_DIR}" \
  --gghbb-test-manifest "${GGHBB_MANIFEST}" \
  --output-dir "${RUN_OUTPUT}" \
  --events 10000 \
  --bootstrap-replicas 0 \
  --candidate-mode leading_pt \
  --device cuda \
  --no-wandb
