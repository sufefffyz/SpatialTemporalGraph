#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  WINDOWS=(1m full)
else
  WINDOWS=("$@")
fi

for WINDOW in "${WINDOWS[@]}"; do
  case "${WINDOW}" in
    full|1m)
      ;;
    *)
      echo "Unsupported window: ${WINDOW}" >&2
      exit 1
      ;;
  esac
done

GRAPHS=(distthre phys adaptive distthre+adaptive phys+adaptive)

cd "${BASICTS_ROOT}"

for WINDOW in "${WINDOWS[@]}"; do
  for GRAPH in "${GRAPHS[@]}"; do
    echo "[$(date '+%F %T')] Running old SD GWNet: graph=${GRAPH}, window=${WINDOW}"
    PYTHONPATH="${REPO_ROOT}:${BASICTS_ROOT}:${PYTHONPATH:-}" \
      SD_OLD_GRAPH="${GRAPH}" SD_OLD_WINDOW="${WINDOW}" \
      python experiments/train.py -c benchmark.configs.GWNet_SD_old -g "${GPU}"
  done
done
