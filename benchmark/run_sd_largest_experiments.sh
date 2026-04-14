#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  EXPERIMENTS=(
    DCRNN_distthre_full
    DCRNN_distthre_1m
    GWNet_distthre_full
    GWNet_distthre_1m
  )
else
  EXPERIMENTS=("$@")
fi

cd "${BASICTS_ROOT}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  WINDOW="full"
  case "${EXPERIMENT}" in
    *_1m)
      WINDOW="1m"
      ;;
    *_full)
      WINDOW="full"
      ;;
  esac
  case "${EXPERIMENT}" in
    DCRNN|DCRNN_original|DCRNN_distthre|DCRNN_distthre_full|DCRNN_distthre_1m)
      CONFIG="baselines/DCRNN/SD_LargeST_${WINDOW}_largeST_original.py"
      LABEL="DCRNN_distthre_${WINDOW}"
      ;;
    GWNet_distthre|GWNet_distthre_full|GWNet_distthre_1m)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_largeST_original.py"
      LABEL="GWNet_distthre_${WINDOW}"
      ;;
    GWNet|GWNET|gwnet|GWNet_original|GWNet_original_adaptive)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_largeST_original.py"
      LABEL="GWNet_original_adaptive_${WINDOW}"
      ;;
    *)
      echo "Unsupported experiment: ${EXPERIMENT}" >&2
      exit 1
      ;;
  esac

  echo "[$(date '+%F %T')] Running ${LABEL} on LargeST-SD via ${CONFIG}"
  python experiments/train.py -c "${CONFIG}" -g "${GPU}"
done
