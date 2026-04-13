#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  EXPERIMENTS=(
    DCRNN_directed
    DCRNN_bidir
    GWNet_directed
    GWNet_bidir
    GWNet_adaptive
  )
else
  EXPERIMENTS=("$@")
fi

cd "${BASICTS_ROOT}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  case "${EXPERIMENT}" in
    DCRNN|DCRNN_directed)
      CONFIG="baselines/DCRNN/SD_phys.py"
      LABEL="DCRNN_directed"
      ;;
    DCRNN_bidir)
      CONFIG="baselines/DCRNN/SD_phys_bidir.py"
      LABEL="DCRNN_bidir"
      ;;
    GWNet|GWNET|gwnet|GWNet_adaptive)
      CONFIG="baselines/GWNet/SD_phys_adaptive.py"
      LABEL="GWNet_adaptive"
      ;;
    GWNet_directed)
      CONFIG="baselines/GWNet/SD_phys_directed.py"
      LABEL="GWNet_directed"
      ;;
    GWNet_bidir)
      CONFIG="baselines/GWNet/SD_phys_bidir.py"
      LABEL="GWNet_bidir"
      ;;
    *)
      echo "Unsupported experiment: ${EXPERIMENT}" >&2
      exit 1
      ;;
  esac

  echo "[$(date '+%F %T')] Running ${LABEL} on SD_phys via ${CONFIG}"
  python experiments/train.py -c "${CONFIG}" -g "${GPU}"
done
