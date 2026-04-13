#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  EXPERIMENTS=(DCRNN_original GWNet_distthre)
else
  EXPERIMENTS=("$@")
fi

cd "${BASICTS_ROOT}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  case "${EXPERIMENT}" in
    DCRNN|DCRNN_original)
      CONFIG="baselines/DCRNN/SD.py"
      LABEL="DCRNN_original"
      ;;
    GWNet_distthre)
      CONFIG="baselines/GWNet/SD_fixed.py"
      LABEL="GWNet_distthre"
      ;;
    GWNet|GWNET|gwnet|GWNet_original|GWNet_original_adaptive)
      CONFIG="baselines/GWNet/SD.py"
      LABEL="GWNet_original_adaptive"
      ;;
    *)
      echo "Unsupported experiment: ${EXPERIMENT}" >&2
      exit 1
      ;;
  esac

  echo "[$(date '+%F %T')] Running ${LABEL} on LargeST-SD via ${CONFIG}"
  python experiments/train.py -c "${CONFIG}" -g "${GPU}"
done
