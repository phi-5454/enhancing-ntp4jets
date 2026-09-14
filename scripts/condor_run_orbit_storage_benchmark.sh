#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 PROJECT_DIR RUN_DIR MANIFEST_ROOT OUTPUT_DIR CMSSW_BASE" >&2
  exit 2
fi

project_dir=$1
run_dir=$2
manifest_root=$3
output_dir=$4
cmssw_base=$5
conda_env=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}

mkdir -p "${output_dir}"
cd "${project_dir}"

if ! find "${cmssw_base}/lib" -type f \
  -name 'pluginOrbitCompressionStorageBenchmarkPlugins.so' -print -quit \
  | grep -q .; then
  echo "Storage plugin is not built below ${cmssw_base}." >&2
  echo "Run scripts/setup_orbit_storage_cmssw.sh offline before submission." >&2
  exit 2
fi

conda run --no-capture-output -p "${conda_env}" \
  python scripts/benchmark_orbit_storage.py \
  --run-dir "${run_dir}" \
  --manifest-root "${manifest_root}" \
  --output-dir "${output_dir}" \
  --cmssw-base "${cmssw_base}" \
  --events-per-sample "${STORAGE_BENCHMARK_EVENTS:-1000}" \
  --samples minbias gghbb tt \
  --model-label "${STORAGE_BENCHMARK_MODEL_LABEL:-Canonical tt VQ-STE 4096}" \
  --keep-existing-containers \
  --device cuda
