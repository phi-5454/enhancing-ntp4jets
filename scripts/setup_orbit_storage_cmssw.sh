#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cmssw_base=${1:-${CMSSW_BASE:-}}
if [[ -z "${cmssw_base}" ]]; then
  echo "Usage: $0 /path/to/CMSSW_X_Y_Z (or set CMSSW_BASE)" >&2
  exit 2
fi
if [[ ! -d "${cmssw_base}/src" ]]; then
  echo "CMSSW source directory does not exist: ${cmssw_base}/src" >&2
  exit 2
fi

# shellcheck source=/dev/null
source /cvmfs/cms.cern.ch/cmsset_default.sh
target=${cmssw_base}/src/OrbitCompression
source_package=${repo_dir}/cmssw/OrbitCompression
if [[ -L "${target}" ]]; then
  if [[ $(readlink -f -- "${target}") != $(readlink -f -- "${source_package}") ]]; then
    echo "Existing symlink points elsewhere: ${target}" >&2
    exit 2
  fi
elif [[ -e "${target}" ]]; then
  echo "Refusing to replace existing CMSSW package: ${target}" >&2
  exit 2
else
  ln -s "${source_package}" "${target}"
fi

cd "${cmssw_base}/src"
# SCRAM project areas remember their absolute creation path. Refresh it here
# so a moved/snapshotted CMSSW area also works on Condor workers.
runtime_cache=${cmssw_base}/.SCRAM/${SCRAM_ARCH}/RuntimeCache.json
recorded_base=""
if [[ -f "${runtime_cache}" ]]; then
  recorded_base=$(sed -n '/"CMSSW_BASE"/{n;s/.*"\(.*\)".*/\1/p;q;}' "${runtime_cache}")
fi
if [[ "${recorded_base}" != "${cmssw_base}" ]]; then
  scram b ProjectRename
fi
eval "$(scram runtime -sh)"
scram b -j "${ORBIT_CMSSW_BUILD_JOBS:-4}"
