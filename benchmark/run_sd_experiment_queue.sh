#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash benchmark/run_sd_experiment_queue.sh <gpu_csv> <min_free_mb> [task ...]" >&2
  echo "Task format: phys:<experiment> or large:<experiment>" >&2
  echo "Examples:" >&2
  echo "  bash benchmark/run_sd_experiment_queue.sh 0,1 12000 phys:DCRNN_directed phys:GWNet_adaptive large:DCRNN_original" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
LOG_DIR="${REPO_ROOT}/benchmark/logs/sd_queue"
mkdir -p "${LOG_DIR}"

GPU_CSV="$1"
MIN_FREE_MB="$2"
shift 2

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

if [[ "$#" -eq 0 ]]; then
  TASKS=(
    "phys:DCRNN_directed"
    "phys:DCRNN_bidir"
    "phys:GWNet_directed"
    "phys:GWNet_bidir"
    "phys:GWNet_adaptive"
    "large:DCRNN_original"
    "large:GWNet_original"
  )
else
  TASKS=("$@")
fi

if ! [[ "${MIN_FREE_MB}" =~ ^[0-9]+$ ]] || [[ "${MIN_FREE_MB}" -le 0 ]]; then
  echo "min_free_mb must be a positive integer, got: ${MIN_FREE_MB}" >&2
  exit 1
fi

if ! [[ "${CHECK_INTERVAL}" =~ ^[0-9]+$ ]] || [[ "${CHECK_INTERVAL}" -le 0 ]]; then
  echo "CHECK_INTERVAL must be a positive integer, got: ${CHECK_INTERVAL}" >&2
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in gpu_csv: ${gpu}" >&2
    exit 1
  fi
done

timestamp() {
  date "+%F %T"
}

sanitize_task_name() {
  echo "$1" | tr ':/' '__'
}

query_free_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

build_command() {
  local task="$1"
  local gpu="$2"
  local scope="${task%%:*}"
  local experiment="${task#*:}"
  case "${scope}" in
    phys)
      printf 'bash %q %q %q' "${REPO_ROOT}/benchmark/run_sd_phys_experiments.sh" "${gpu}" "${experiment}"
      ;;
    large)
      printf 'bash %q %q %q' "${REPO_ROOT}/benchmark/run_sd_largest_experiments.sh" "${gpu}" "${experiment}"
      ;;
    *)
      echo "Unsupported task scope: ${scope}" >&2
      exit 1
      ;;
  esac
}

declare -A GPU_PID=()
declare -A GPU_TASK=()
declare -A GPU_LOG=()

pending_index=0
total_tasks="${#TASKS[@]}"

echo "Queue started at $(timestamp)"
echo "GPUs           : ${GPUS[*]}"
echo "MIN_FREE_MB    : ${MIN_FREE_MB}"
echo "CHECK_INTERVAL : ${CHECK_INTERVAL}"
echo "Tasks          : ${TASKS[*]}"
echo "Log dir        : ${LOG_DIR}"

while true; do
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "$(timestamp) Finished ${GPU_TASK[$gpu]} on GPU ${gpu} (log: ${GPU_LOG[$gpu]})"
      unset GPU_PID["$gpu"]
      unset GPU_TASK["$gpu"]
      unset GPU_LOG["$gpu"]
    fi
  done

  if [[ "${pending_index}" -ge "${total_tasks}" ]]; then
    all_idle=1
    for gpu in "${GPUS[@]}"; do
      if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
        all_idle=0
        break
      fi
    done
    if [[ "${all_idle}" -eq 1 ]]; then
      break
    fi
    sleep "${CHECK_INTERVAL}"
    continue
  fi

  launched_any=0
  for gpu in "${GPUS[@]}"; do
    if [[ "${pending_index}" -ge "${total_tasks}" ]]; then
      break
    fi
    if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
      continue
    fi

    free_mb="$(query_free_mb "${gpu}")"
    if [[ -z "${free_mb}" ]]; then
      echo "$(timestamp) Failed to query GPU ${gpu}, skip this round"
      continue
    fi
    if [[ "${free_mb}" -lt "${MIN_FREE_MB}" ]]; then
      echo "$(timestamp) GPU ${gpu} free=${free_mb}MB < ${MIN_FREE_MB}MB, waiting"
      continue
    fi

    task="${TASKS[$pending_index]}"
    pending_index=$((pending_index + 1))
    log_file="${LOG_DIR}/$(date +%Y%m%d-%H%M%S)_gpu${gpu}_$(sanitize_task_name "${task}").log"
    cmd="$(build_command "${task}" "${gpu}")"
    echo "$(timestamp) Launch ${task} on GPU ${gpu} (free=${free_mb}MB)"
    nohup bash -lc "${cmd}" > "${log_file}" 2>&1 &
    GPU_PID["$gpu"]=$!
    GPU_TASK["$gpu"]="${task}"
    GPU_LOG["$gpu"]="${log_file}"
    launched_any=1
  done

  if [[ "${launched_any}" -eq 0 ]]; then
    sleep "${CHECK_INTERVAL}"
  fi
done

echo "Queue finished at $(timestamp)"
