#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash benchmark/run_sd_5min_gwnet_queue.sh <gpu_csv> <min_free_mb> [task ...]" >&2
  echo "Task format: <graph>:<window>" >&2
  echo "Example: bash benchmark/run_sd_5min_gwnet_queue.sh 0 12000 distthre:1m phys+adaptive:1m" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
POST_LAUNCH_WAIT="${POST_LAUNCH_WAIT:-15}"
LOG_DIR="${REPO_ROOT}/benchmark/logs/sd_5min_gwnet_queue"
mkdir -p "${LOG_DIR}"

GPU_CSV="$1"
MIN_FREE_MB="$2"
shift 2

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

if [[ "$#" -eq 0 ]]; then
  TASKS=(
    "distthre:1m"
    "phys_dir:1m"
    "phys_bidir:1m"
    "adaptive:1m"
    "phys+adaptive:1m"
    "distthre+adaptive:1m"
    "distthre:full"
    "phys_dir:full"
    "phys_bidir:full"
    "adaptive:full"
    "phys+adaptive:full"
    "distthre+adaptive:full"
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

if ! [[ "${MAX_JOBS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${MAX_JOBS_PER_GPU}" -le 0 ]]; then
  echo "MAX_JOBS_PER_GPU must be a positive integer, got: ${MAX_JOBS_PER_GPU}" >&2
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
  echo "$1" | tr ':+/' '___'
}

query_free_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

build_command() {
  local task="$1"
  local gpu="$2"
  local graph="${task%%:*}"
  local window="${task#*:}"
  printf 'cd %q && PYTHONPATH=%q SD_BENCH_GRAPH=%q SD_BENCH_WINDOW=%q python experiments/train.py -c benchmark.configs.GWNet_SD_5min -g %q' \
    "${BASICTS_ROOT}" "${REPO_ROOT}:${BASICTS_ROOT}:${PYTHONPATH:-}" "${graph}" "${window}" "${gpu}"
}

join_by_pipe() {
  local IFS='|'
  echo "$*"
}

active_count() {
  local gpu="$1"
  local pid_blob="${GPU_PIDS[$gpu]:-}"
  if [[ -z "${pid_blob}" ]]; then
    echo 0
    return
  fi
  read -r -a pid_array <<< "${pid_blob}"
  echo "${#pid_array[@]}"
}

cleanup_gpu_state() {
  local gpu="$1"
  local pid_blob="${GPU_PIDS[$gpu]:-}"
  local task_blob="${GPU_TASKS[$gpu]:-}"
  local log_blob="${GPU_LOGS[$gpu]:-}"

  if [[ -z "${pid_blob}" ]]; then
    return
  fi

  read -r -a pid_array <<< "${pid_blob}"
  local old_ifs="${IFS}"
  IFS='|' read -r -a task_array <<< "${task_blob}"
  IFS='|' read -r -a log_array <<< "${log_blob}"
  IFS="${old_ifs}"

  local kept_pids=()
  local kept_tasks=()
  local kept_logs=()
  local idx
  for idx in "${!pid_array[@]}"; do
    local pid="${pid_array[$idx]}"
    local task="${task_array[$idx]:-unknown}"
    local log="${log_array[$idx]:-unknown}"
    if kill -0 "${pid}" 2>/dev/null; then
      kept_pids+=("${pid}")
      kept_tasks+=("${task}")
      kept_logs+=("${log}")
    else
      echo "$(timestamp) Finished ${task} on GPU ${gpu} (log: ${log})"
    fi
  done

  if [[ "${#kept_pids[@]}" -eq 0 ]]; then
    unset GPU_PIDS["$gpu"]
    unset GPU_TASKS["$gpu"]
    unset GPU_LOGS["$gpu"]
  else
    GPU_PIDS["$gpu"]="${kept_pids[*]}"
    GPU_TASKS["$gpu"]="$(join_by_pipe "${kept_tasks[@]}")"
    GPU_LOGS["$gpu"]="$(join_by_pipe "${kept_logs[@]}")"
  fi
}

append_gpu_state() {
  local gpu="$1"
  local pid="$2"
  local task="$3"
  local log_file="$4"

  local pid_blob="${GPU_PIDS[$gpu]:-}"
  local task_blob="${GPU_TASKS[$gpu]:-}"
  local log_blob="${GPU_LOGS[$gpu]:-}"

  if [[ -z "${pid_blob}" ]]; then
    GPU_PIDS["$gpu"]="${pid}"
    GPU_TASKS["$gpu"]="${task}"
    GPU_LOGS["$gpu"]="${log_file}"
  else
    GPU_PIDS["$gpu"]="${pid_blob} ${pid}"
    GPU_TASKS["$gpu"]="${task_blob}|${task}"
    GPU_LOGS["$gpu"]="${log_blob}|${log_file}"
  fi
}

declare -A GPU_PIDS=()
declare -A GPU_TASKS=()
declare -A GPU_LOGS=()

pending_index=0
total_tasks="${#TASKS[@]}"

echo "Queue started at $(timestamp)"
echo "GPUs            : ${GPUS[*]}"
echo "MIN_FREE_MB     : ${MIN_FREE_MB}"
echo "CHECK_INTERVAL  : ${CHECK_INTERVAL}"
echo "MAX_JOBS_PER_GPU: ${MAX_JOBS_PER_GPU}"
echo "POST_LAUNCH_WAIT: ${POST_LAUNCH_WAIT}"
echo "Tasks           : ${TASKS[*]}"
echo "Log dir         : ${LOG_DIR}"

while true; do
  for gpu in "${GPUS[@]}"; do
    cleanup_gpu_state "${gpu}"
  done

  if [[ "${pending_index}" -ge "${total_tasks}" ]]; then
    all_idle=1
    for gpu in "${GPUS[@]}"; do
      if [[ "$(active_count "${gpu}")" -gt 0 ]]; then
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
    if [[ "$(active_count "${gpu}")" -ge "${MAX_JOBS_PER_GPU}" ]]; then
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
    echo "$(timestamp) Launch ${task} on GPU ${gpu} (free=${free_mb}MB, active=$(active_count "${gpu}")/${MAX_JOBS_PER_GPU})"
    nohup bash -lc "${cmd}" > "${log_file}" 2>&1 &
    append_gpu_state "${gpu}" "$!" "${task}" "${log_file}"
    launched_any=1
  done

  if [[ "${launched_any}" -eq 0 ]]; then
    sleep "${CHECK_INTERVAL}"
  elif [[ "${POST_LAUNCH_WAIT}" -gt 0 ]]; then
    sleep "${POST_LAUNCH_WAIT}"
  fi
done

echo "Queue finished at $(timestamp)"
