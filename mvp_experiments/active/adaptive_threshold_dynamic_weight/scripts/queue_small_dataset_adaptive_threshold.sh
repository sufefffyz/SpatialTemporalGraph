#!/usr/bin/env bash
set -euo pipefail

DATASET_KEY="${1:-}"
GPU="${2:-}"
RUN_TAG="${3:-smallbench_$(date '+%Y%m%d_%H%M%S')}"
shift 3

if [[ -z "${DATASET_KEY}" || -z "${GPU}" || "$#" -eq 0 ]]; then
  echo "Usage: $0 {pems04|metrla} gpu_id run_tag model:graph [model:graph ...]" >&2
  exit 2
fi

MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
QUEUE_LOG="${REPO_ROOT}/BasicTS/logs/adaptive_threshold_dynamic_weight/smallbench/${RUN_TAG}/${DATASET_KEY}_queue_gpu${GPU}.log"
mkdir -p "$(dirname "${QUEUE_LOG}")"
cd "${REPO_ROOT}"

count_basicts_jobs() {
  local pids pid cmd count
  count=0
  pids=$(nvidia-smi pmon -c 1 2>/dev/null | awk -v g="${GPU}" '$1 == g && $3 == "C" {print $2}') || true
  for pid in ${pids}; do
    cmd=$(ps -p "${pid}" -o cmd= 2>/dev/null || true)
    if [[ "${cmd}" == *@BasicTS* ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

wait_slot() {
  local count
  while true; do
    count=$(count_basicts_jobs)
    echo "[$(date '+%F %T')] gpu=${GPU} basicts_jobs=${count} max=${MAX_TRAIN_PER_GPU}" | tee -a "${QUEUE_LOG}"
    if (( count < MAX_TRAIN_PER_GPU )); then
      return 0
    fi
    sleep "${QUEUE_POLL_SECONDS:-300}"
  done
}

for spec in "$@"; do
  model="${spec%%:*}"
  graph="${spec#*:}"
  wait_slot
  echo "[$(date '+%F %T')] launch ${DATASET_KEY} ${model} ${graph} on gpu ${GPU}" | tee -a "${QUEUE_LOG}"
  bash mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/run_small_dataset_adaptive_threshold.sh \
    "${DATASET_KEY}" "${model}" "${graph}" "${GPU}" "${RUN_TAG}"
  echo "[$(date '+%F %T')] done ${DATASET_KEY} ${model} ${graph} on gpu ${GPU}" | tee -a "${QUEUE_LOG}"
done
