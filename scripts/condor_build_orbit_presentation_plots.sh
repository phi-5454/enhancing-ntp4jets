#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 9 ]; then
  echo "Usage: $0 PROJECT_DIR OUTPUT_DIR CACHE_DIR FULL_MANIFEST NEAR_MANIFEST" \
    "FULL_TT FULL_GGHBB NEAR_TT NEAR_GGHBB" >&2
  exit 2
fi

PROJECT_DIR=$1
OUTPUT_DIR=$2
CACHE_DIR=$3
OBSERVABLE_CACHE_DIR=${HIGGS_OBSERVABLE_CACHE_DIR:-${CACHE_DIR}}
FULL_MANIFEST=$4
NEAR_MANIFEST=$5
FULL_TT=$6
FULL_GGHBB=$7
NEAR_TT=$8
NEAR_GGHBB=$9
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}

cd "${PROJECT_DIR}"
conda run --no-capture-output -p "${CONDA_ENV}" python \
  scripts/build_orbit_presentation_plots.py \
  --golden-collection full/training_like "${FULL_TT}" \
  --golden-collection full/ggHbb "${FULL_GGHBB}" \
  --golden-collection near_4096/training_like "${NEAR_TT}" \
  --golden-collection near_4096/ggHbb "${NEAR_GGHBB}" \
  --higgs-full-manifest "${FULL_MANIFEST}" \
  --higgs-near-manifest "${NEAR_MANIFEST}" \
  --higgs-cache-dir "${CACHE_DIR}" \
  --higgs-observable-cache-dir "${OBSERVABLE_CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-replicas 50 \
  --wandb-project orbit-tokenizer \
  --wandb-name canonical_tt_presentation_plots \
  --wandb-group presentation_plots
