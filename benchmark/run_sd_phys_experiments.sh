#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  EXPERIMENTS=(
    DCRNN_directed_full
    DCRNN_directed_1m
    DCRNN_bidir_full
    DCRNN_bidir_1m
    GWNet_directed_full
    GWNet_directed_1m
    GWNet_bidir_full
    GWNet_bidir_1m
    GWNet_adaptive_only_full
    GWNet_adaptive_only_1m
    GWNet_phys_adaptive_full
    GWNet_phys_adaptive_1m
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
    DCRNN|DCRNN_directed|DCRNN_directed_full|DCRNN_directed_1m)
      CONFIG="baselines/DCRNN/SD_LargeST_${WINDOW}_physical_forward.py"
      LABEL="DCRNN_directed_${WINDOW}"
      ;;
    DCRNN_bidir|DCRNN_bidir_full|DCRNN_bidir_1m)
      CONFIG="baselines/DCRNN/SD_LargeST_${WINDOW}_physical_bidir.py"
      LABEL="DCRNN_bidir_${WINDOW}"
      ;;
    GWNet|GWNET|gwnet|GWNet_phys_adaptive|GWNet_phys_adaptive_full|GWNet_phys_adaptive_1m)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_adaptive_plus_phys.py"
      LABEL="GWNet_phys_adaptive_${WINDOW}"
      ;;
    GWNet_adaptive_only|GWNet_adaptive_only_full|GWNet_adaptive_only_1m)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_adaptive_only.py"
      LABEL="GWNet_adaptive_only_${WINDOW}"
      ;;
    GWNet_directed|GWNet_directed_full|GWNet_directed_1m)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_physical_forward.py"
      LABEL="GWNet_directed_${WINDOW}"
      ;;
    GWNet_bidir|GWNet_bidir_full|GWNet_bidir_1m)
      CONFIG="baselines/GWNet/SD_LargeST_${WINDOW}_physical_bidir.py"
      LABEL="GWNet_bidir_${WINDOW}"
      ;;
    *)
      echo "Unsupported experiment: ${EXPERIMENT}" >&2
      exit 1
      ;;
  esac

  echo "[$(date '+%F %T')] Running ${LABEL} on SD_phys via ${CONFIG}"
  python experiments/train.py -c "${CONFIG}" -g "${GPU}"
done
