#!/usr/bin/env bash
# Selected split comparison with FAISS k-means as the only non-neural baseline.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:-"${REPO_DIR}/../enhancing_ntp4jets_runs"}
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}
MPLCONFIGDIR=${MPLCONFIGDIR:-"${RUN_ROOT}/matplotlib_multirun_faiss_only"}

mkdir -p "${MPLCONFIGDIR}"

suite_runs() {
  local suite_dir="${RUN_ROOT}/$1"
  if [[ ! -d "${suite_dir}" ]]; then
    echo "Missing run suite: ${suite_dir}" >&2
    return 1
  fi

  find "${suite_dir}" \
    -type f \
    -path '*/.hydra/config.yaml' \
    -printf '%h\n' |
    sed 's#/.hydra$##' |
    sort
}

load_suite() {
  local array_name=$1
  local suite_name=$2
  local -n runs="${array_name}"
  mapfile -t runs < <(suite_runs "${suite_name}")
  if (( ${#runs[@]} == 0 )); then
    echo "No Hydra runs found in ${RUN_ROOT}/${suite_name}" >&2
    exit 2
  fi
}

load_suite FSQ orbit_fsq_codebook_scan_957028
load_suite VQ_STE orbit_vq_ste_scan_957029
load_suite VQ_ROT orbit_vq_rotation_scan_957030
load_suite FSQ_FSQ64 orbit_split_fsq_alpha64_scan_957032
load_suite VQ_FSQ32 orbit_split_vq_mu_fsq_alpha32_scan_957034
load_suite FAISS_KMEANS orbit_faiss_kmeans_scan_960568

if command -v python >/dev/null 2>&1; then
  PYTHON_COMMAND=(python)
elif command -v conda >/dev/null 2>&1 && [[ -d "${CONDA_ENV}" ]]; then
  PYTHON_COMMAND=(conda run --no-capture-output -p "${CONDA_ENV}" python)
else
  echo "Neither python nor the Condor environment is available." >&2
  exit 2
fi

MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_COMMAND[@]}" \
  "${REPO_DIR}/scripts/collect_orbit_multirun.py" \
  --family "FSQ" "${FSQ[@]}" \
  --family "VQ STE" "${VQ_STE[@]}" \
  --family "VQ rotation" "${VQ_ROT[@]}" \
  --family "FSQ-mu + FSQ-alpha64" "${FSQ_FSQ64[@]}" \
  --family "VQ-mu + FSQ-alpha32" "${VQ_FSQ32[@]}" \
  --family "FAISS k-means" "${FAISS_KMEANS[@]}" \
  --stage test \
  --group all \
  --output-dir "${RUN_ROOT}/comparisons/957028-960568-faiss-only-baseline/test/all" \
  --wandb-project orbit-tokenizer \
  --wandb-name codebook_families_957028_960568_faiss_only_baseline_test_all \
  --wandb-group orbit_multirun_comparisons
