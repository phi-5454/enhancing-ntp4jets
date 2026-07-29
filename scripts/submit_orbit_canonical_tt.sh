#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_DIR}"

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "condor_submit is not available; load the HTCondor environment first." >&2
  exit 2
fi

experiment=orbit_canonical_tt
submit_files=(
  condor/orbit_continuous_autoencoder_baseline_canonical.sub
  condor/orbit_dumb_learned_quantization_baseline_scan_canonical.sub
  condor/orbit_dumb_quantization_baseline_scan_canonical.sub
  condor/orbit_faiss_kmeans_scan_canonical.sub
  condor/orbit_fsq_codebook_scan_canonical.sub
  condor/orbit_vq_rotation_scan_canonical.sub
  condor/orbit_vq_ste_scan_canonical.sub
  condor/orbit_split_vq_mu_fsq_alpha128_scan_canonical.sub
  condor/orbit_split_vq_mu_fsq_alpha64_scan_canonical.sub
  condor/orbit_split_vq_mu_fsq_alpha32_scan_canonical.sub
  condor/orbit_split_fsq_alpha128_scan_canonical.sub
  condor/orbit_split_fsq_alpha64_scan_canonical.sub
  condor/orbit_split_fsq_alpha32_scan_canonical.sub
)

for submit_file in "${submit_files[@]}"; do
  echo "Submitting ${submit_file} with experiment=${experiment}"
  condor_submit \
    -append "CANONICAL_EXPERIMENT=${experiment}" \
    "${submit_file}"
done
