#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
RUN_TAG="${2:-dynamic_threshold_priority_$(date '+%Y%m%d_%H%M%S')}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_dynamic_weight}"
WAIT_UTIL="${DYNAMIC_QUEUE_WAIT_UTIL:-40}"
WAIT_MEM_MIB="${DYNAMIC_QUEUE_WAIT_MEM_MIB:-20000}"
POLL_SECONDS="${DYNAMIC_QUEUE_POLL_SECONDS:-300}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LOG_ROOT="${REPO_ROOT}/BasicTS/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/${RUN_TAG}/queue"
mkdir -p "${LOG_ROOT}"
QUEUE_LOG="${LOG_ROOT}/queue.log"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local line util mem
    line="$(nvidia-smi --id="${gpu}" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1)"
    util="$(echo "${line}" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
    mem="$(echo "${line}" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
    echo "[$(date '+%F %T')] GPU ${gpu}: util=${util}% mem=${mem}MiB; thresholds util<=${WAIT_UTIL}% mem<=${WAIT_MEM_MIB}MiB" | tee -a "${QUEUE_LOG}"
    if [[ "${util}" -le "${WAIT_UTIL}" && "${mem}" -le "${WAIT_MEM_MIB}" ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

RUNS=(
  "flownet_soft:soft"
  "gwnet:hard"
  "dcrnn:hard"
  "stgcn:hard"
)

echo "[$(date '+%F %T')] Dynamic threshold priority queue start: gpu=${GPU}, run_tag=${RUN_TAG}, project=${PROJECT}" | tee -a "${QUEUE_LOG}"
for item in "${RUNS[@]}"; do
  model="${item%%:*}"
  mode="${item##*:}"
  wait_for_gpu "${GPU}"
  echo "[$(date '+%F %T')] Launching ${model} ${mode}" | tee -a "${QUEUE_LOG}"
  WANDB_PROJECT="${PROJECT}" WANDB_MODE=online \
    "${REPO_ROOT}/mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/run_dynamic_threshold_basicts.sh" \
    "${model}" "${mode}" "${GPU}" "${RUN_TAG}"
done
echo "[$(date '+%F %T')] Dynamic threshold priority queue finished." | tee -a "${QUEUE_LOG}"
