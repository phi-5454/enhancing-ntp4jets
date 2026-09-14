#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_DIR}"

dry_run=false
selection=all
priority=0

usage() {
  cat <<'EOF'
Usage: scripts/submit_orbit_canonical_missing.sh [--dry-run] [--dataset DATASET] [--priority N]

Submit the canonical configurations found to be incomplete in the Condor logs.

DATASET may be one of:
  all          all four canonical datasets (default)
  tt           orbit_canonical_tt
  tt-full      orbit_canonical_tt_full_event
  mixed        orbit_canonical_qcd_tt_vjets_vv
  mixed-golden missing mixture counterparts of the 46-model tt golden set
  mixed-full   orbit_canonical_qcd_tt_vjets_vv_full_event

Use --dry-run to print every submission without contacting the scheduler.
Use --priority to set the HTCondor job priority (default: 0; larger runs first).
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --dataset)
      if (($# < 2)); then
        echo "--dataset requires a value" >&2
        exit 2
      fi
      selection=$2
      shift 2
      ;;
    --priority)
      if (($# < 2)); then
        echo "--priority requires an integer" >&2
        exit 2
      fi
      priority=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${selection}" in
  all|tt|tt-full|mixed|mixed-golden|mixed-full) ;;
  *)
    echo "Unknown dataset selection: ${selection}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "${priority}" =~ ^-?[0-9]+$ ]]; then
  echo "--priority must be an integer: ${priority}" >&2
  exit 2
fi

if ! ${dry_run} && ! command -v condor_submit >/dev/null 2>&1; then
  echo "condor_submit is not available; load the HTCondor environment first." >&2
  exit 2
fi

tmp_dir=$(mktemp -d /tmp/orbit-canonical-missing.XXXXXX)
trap 'rm -rf -- "${tmp_dir}"' EXIT

submitted_jobs=0
submitted_clusters=0

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

condor_options() {
  local experiment=$1
  CONDOR_OPTIONS=(
    -append "CANONICAL_EXPERIMENT=${experiment}"
    -append "priority = ${priority}"
  )
  if [[ "${experiment}" == *_full_event ]]; then
    CONDOR_OPTIONS+=(
      -append "request_memory = 12GB"
      -append '+JobFlavour = "tomorrow"'
    )
  fi
}

submit_family() {
  local experiment=$1
  local submit_file=$2
  local number_of_jobs=$3
  local label=$4
  local -a command

  condor_options "${experiment}"
  command=(condor_submit "${CONDOR_OPTIONS[@]}" "${submit_file}")

  echo "Submitting ${label} (${number_of_jobs} jobs)"
  if ${dry_run}; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
  submitted_jobs=$((submitted_jobs + number_of_jobs))
  submitted_clusters=$((submitted_clusters + 1))
}

queue_less_submit_file() {
  local submit_file=$1
  local target
  target="${tmp_dir}/$(basename "${submit_file}")"
  if [[ ! -f "${target}" ]]; then
    sed '/^[[:space:]]*queue[[:space:]]/,$d' "${submit_file}" > "${target}"
  fi
  printf '%s\n' "${target}"
}

submit_one() {
  local experiment=$1
  local submit_file=$2
  local label=$3
  shift 3
  local stripped_submit
  local assignment
  local -a command

  stripped_submit=$(queue_less_submit_file "${submit_file}")
  condor_options "${experiment}"
  command=(condor_submit "${CONDOR_OPTIONS[@]}")
  for assignment in "$@"; do
    command+=(-append "${assignment}")
  done
  command+=("${stripped_submit}" -queue 1)

  echo "Submitting ${label} (1 job)"
  if ${dry_run}; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
  submitted_jobs=$((submitted_jobs + 1))
  submitted_clusters=$((submitted_clusters + 1))
}

submit_fsq_codebook_one() {
  local experiment=$1
  local level=$2
  submit_one \
    "${experiment}" \
    condor/orbit_fsq_codebook_scan_canonical.sub \
    "${experiment}: fsq_mu_${level}x3" \
    "RUN_NAME=fsq_mu_${level}x3" \
    "MODEL_CONFIG=model_vqvae_transformer_split_fsq_mu_${level}x3" \
    "INCLUDE_CODE_HIST=false"
}

submit_split_one() {
  local experiment=$1
  local submit_file=$2
  local run_name=$3
  submit_one \
    "${experiment}" \
    "${submit_file}" \
    "${experiment}: ${run_name}" \
    "RUN_NAME=${run_name}" \
    "MODEL_CONFIG=model_vqvae_transformer_split_${run_name}"
}

submit_tt() {
  local experiment=orbit_canonical_tt

  submit_fsq_codebook_one "${experiment}" 13
  submit_one \
    "${experiment}" condor/orbit_vq_rotation_scan_canonical.sub \
    "${experiment}: vq_rotation_codes_8192" "NUM_CODES=8192"
  submit_one \
    "${experiment}" condor/orbit_vq_ste_scan_canonical.sub \
    "${experiment}: vq_ste_codes_256" "NUM_CODES=256"
}

submit_tt_full() {
  local experiment=orbit_canonical_tt_full_event
  local value
  local checkpoint=/eos/user/y/yelberke/enhancing_ntp4jets_runs/orbit_vq_ste_scan_canonical_orbit_canonical_tt_full_event_1001367/orbit-canonical-tt-full-event/runs/2026-08-24_04-12-30_b9pgpun205_ConstantWing/checkpoints/best.ckpt

  submit_family \
    "${experiment}" condor/orbit_continuous_autoencoder_baseline_canonical.sub 1 \
    "${experiment}: continuous autoencoder"

  for value in 8 10 13 16; do
    submit_one \
      "${experiment}" condor/orbit_dumb_learned_quantization_baseline_scan_canonical.sub \
      "${experiment}: dumb_learned_quant_baseline_${value}x3" "LEVEL=${value}"
  done
  for value in 8 10 13; do
    submit_one \
      "${experiment}" condor/orbit_dumb_quantization_baseline_scan_canonical.sub \
      "${experiment}: dumb_quant_baseline_${value}x3" "LEVEL=${value}"
  done
  for value in 512 2048 4096; do
    submit_one \
      "${experiment}" condor/orbit_faiss_kmeans_scan_canonical.sub \
      "${experiment}: faiss_kmeans_codes_${value}" "NUM_CODES=${value}"
  done
  for value in 5 8 16; do
    submit_fsq_codebook_one "${experiment}" "${value}"
  done
  for value in 2048 8192; do
    submit_one \
      "${experiment}" condor/orbit_vq_rotation_scan_canonical.sub \
      "${experiment}: vq_rotation_codes_${value}" "NUM_CODES=${value}"
  done
  for value in 256 512 1024 2048 4096 16384; do
    submit_one \
      "${experiment}" condor/orbit_vq_ste_scan_canonical.sub \
      "${experiment}: vq_ste_codes_${value}" "NUM_CODES=${value}"
  done

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Required test-only checkpoint does not exist: ${checkpoint}" >&2
    exit 2
  fi
  submit_one \
    "${experiment}" condor/orbit_vq_ste_scan_canonical.sub \
    "${experiment}: vq_ste_codes_8192 test-only recovery" \
    "NUM_CODES=8192" \
    "NUM_WORKERS=0" \
    "MODEL_ARGS=model=model_vqvae_transformer model.model_kwargs.vq_kwargs.num_codes=8192 model.model_kwargs.vq_kwargs.rotation_tr=false callbacks.orbit_plotting_callback.include_codebook_histogram=false train=false ckpt_path_for_evaluation=${checkpoint}"

  submit_family \
    "${experiment}" condor/orbit_split_vq_mu_fsq_alpha128_scan_canonical.sub 8 \
    "${experiment}: complete split VQ+FSQ alpha128 family"
  submit_family \
    "${experiment}" condor/orbit_split_vq_mu_fsq_alpha64_scan_canonical.sub 8 \
    "${experiment}: complete split VQ+FSQ alpha64 family"
  submit_family \
    "${experiment}" condor/orbit_split_vq_mu_fsq_alpha32_scan_canonical.sub 8 \
    "${experiment}: complete split VQ+FSQ alpha32 family"
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha128_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha128 family"
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha64_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha64 family"
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha32_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha32 family"
}

submit_mixed() {
  local experiment=orbit_canonical_qcd_tt_vjets_vv
  local value

  submit_one \
    "${experiment}" condor/orbit_vq_rotation_scan_canonical.sub \
    "${experiment}: vq_rotation_codes_16384" "NUM_CODES=16384"
  submit_split_one \
    "${experiment}" condor/orbit_split_vq_mu_fsq_alpha128_scan_canonical.sub \
    vq_mu_8_fsq_alpha_128
  for value in 64 128 256 512; do
    submit_split_one \
      "${experiment}" condor/orbit_split_vq_mu_fsq_alpha32_scan_canonical.sub \
      "vq_mu_${value}_fsq_alpha_32"
  done
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha128_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha128 family"
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha64_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha64 family"
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha32_scan_canonical.sub 8 \
    "${experiment}: complete split FSQ alpha32 family"
}

submit_mixed_golden() {
  local experiment=orbit_canonical_qcd_tt_vjets_vv
  local value

  # Missing counterparts of the 46 entries in the canonical-tt golden
  # comparison manifest, audited on 2026-09-09. Keep this focused target
  # separate from `mixed`, which covers every incomplete 94-scan entry.
  submit_one \
    "${experiment}" condor/orbit_vq_rotation_scan_canonical.sub \
    "${experiment}: vq_rotation_codes_16384" "NUM_CODES=16384"
  for value in 64 128 256 512; do
    submit_split_one \
      "${experiment}" condor/orbit_split_vq_mu_fsq_alpha32_scan_canonical.sub \
      "vq_mu_${value}_fsq_alpha_32"
  done
  submit_family \
    "${experiment}" condor/orbit_split_fsq_alpha64_scan_canonical.sub 8 \
    "${experiment}: golden-set split FSQ alpha64 family"
}

submit_mixed_full() {
  local experiment=orbit_canonical_qcd_tt_vjets_vv_full_event
  local -a submit_files=(
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
  local -a job_counts=(1 7 7 8 7 8 8 8 8 8 8 8 8)
  local index

  for index in "${!submit_files[@]}"; do
    submit_family \
      "${experiment}" "${submit_files[index]}" "${job_counts[index]}" \
      "${experiment}: $(basename "${submit_files[index]}" .sub)"
  done
}

case "${selection}" in
  all)
    submit_tt
    submit_tt_full
    submit_mixed
    submit_mixed_full
    ;;
  tt) submit_tt ;;
  tt-full) submit_tt_full ;;
  mixed) submit_mixed ;;
  mixed-golden) submit_mixed_golden ;;
  mixed-full) submit_mixed_full ;;
esac

echo "Prepared ${submitted_jobs} jobs in ${submitted_clusters} Condor clusters."
if ${dry_run}; then
  echo "Dry run only; nothing was submitted."
fi
