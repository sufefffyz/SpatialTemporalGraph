#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  WINDOWS=(full 1m)
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

GRAPHS=(distthre phys_dir phys_bidir adaptive phys+adaptive distthre+adaptive)

cd "${BASICTS_ROOT}"

for WINDOW in "${WINDOWS[@]}"; do
  for GRAPH in "${GRAPHS[@]}"; do
    echo "[$(date '+%F %T')] Running GWNet on SD_5min: graph=${GRAPH}, window=${WINDOW}"
    PYTHONPATH="${REPO_ROOT}:${BASICTS_ROOT}:${PYTHONPATH:-}" \
      SD_BENCH_GRAPH="${GRAPH}" SD_BENCH_WINDOW="${WINDOW}" \
      python experiments/train.py -c benchmark.configs.GWNet_SD_5min -g "${GPU}"
  done
done
