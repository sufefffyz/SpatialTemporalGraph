#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark_traffic_paper_baselines}"
DATASET_NAME="${DATASET_NAME:-city_traffic_m_volume__category__1_0}"
LABEL_TAG="${LABEL_TAG:-paperlike}"
N_VARS="${N_VARS:-3}"
SCORE_N_JOBS="${SCORE_N_JOBS:-32}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"
METHODS="${METHODS:-var cc rp combo pcmci varlingam}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESERVED_MEM_GB="${RESERVED_MEM_GB:-64}"
POLL_SECONDS="${POLL_SECONDS:-15}"
ESTIMATED_JOB_MEM_GB="${ESTIMATED_JOB_MEM_GB:-}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(random confounder close root_cause)
fi

read -r -a METHODS_ARRAY <<< "$METHODS"

method_override() {
  local method="$1"
  case "$method" in
    var)
      echo 'methods=[{name:var,max_lag:3,var_absolute_values:true,map_to_summary_graph:max,var_min_std:1.0e-12,var_verbose_failures:false}]'
      ;;
    cc)
      echo 'methods=[{name:cc,max_lag:3,cc_absolute_values:true,map_to_summary_graph:max,naive_child_selection:all}]'
      ;;
    rp)
      echo 'methods=[{name:rp,max_lag:3,rp_score_mode:difference,naive_child_selection:all}]'
      ;;
    combo)
      echo 'methods=[{name:combo,max_lag:3,cc_absolute_values:true,map_to_summary_graph:max,rp_score_mode:difference,combo_union_mode:max,naive_child_selection:all}]'
      ;;
    pcmci)
      echo 'methods=[{name:pcmci,max_lag:3,map_to_summary_graph:max,pcmci_absolute_values:true,pcmci_pc_alpha:0.05,pcmci_alpha_level:0.05,pcmci_filter_nonsignificant:true,pcmci_verbosity:0}]'
      ;;
    varlingam)
      echo 'methods=[{name:varlingam,max_lag:3,map_to_summary_graph:max,varlingam_absolute_values:true,varlingam_criterion:null,varlingam_prune:true,varlingam_random_state:0}]'
      ;;
    dynotears)
      echo 'methods=[{name:dynotears,max_lag:3,map_to_summary_graph:max,dynotears_absolute_values:true,dynotears_lambda_w:0.1,dynotears_lambda_a:0.1,dynotears_max_iter:100,dynotears_h_tol:1.0e-8,dynotears_w_threshold:0.0}]'
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

estimate_job_mem_gb() {
  if [ -n "$ESTIMATED_JOB_MEM_GB" ]; then
    echo "$ESTIMATED_JOB_MEM_GB"
    return
  fi

  case "$N_VARS" in
    3) echo "24" ;;
    4) echo "40" ;;
    5) echo "64" ;;
    6) echo "96" ;;
    *) echo "48" ;;
  esac
}

mem_available_gb() {
  awk '/MemAvailable:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo
}

running_count() {
  local count=0
  for pid in "${RUN_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

remove_pid() {
  local target="$1"
  local next=()
  for pid in "${RUN_PIDS[@]:-}"; do
    if [ "$pid" != "$target" ]; then
      next+=("$pid")
    fi
  done
  RUN_PIDS=("${next[@]:-}")
}

collect_finished_jobs() {
  local pid status task
  for pid in "${RUN_PIDS[@]:-}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      status=$?
      task="${PID_TO_TASK[$pid]}"
      if [ "$status" -eq 0 ]; then
        echo "Finished: $task"
      else
        echo "Failed  : $task (exit=$status)" >&2
        FAILED_RUNS+=("${task}:exit-${status}")
      fi
      unset "PID_TO_TASK[$pid]"
      remove_pid "$pid"
    fi
  done
}

launch_job() {
  local strategy="$1"
  local method="$2"
  local label_path="$3"
  local log_path="$4"
  local override="$5"

  echo "Launching strategy=$strategy method=$method"
  echo "  label_path : $label_path"
  echo "  log        : $log_path"

  (
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" benchmark.py --config-name "$CONFIG_NAME" \
      load_mode=eager \
      n_jobs=1 \
      score_n_jobs="$SCORE_N_JOBS" \
      score_chunk_size="$SCORE_CHUNK_SIZE" \
      label_path="$label_path" \
      "$override"
  ) >"$log_path" 2>&1 &

  local pid=$!
  RUN_PIDS+=("$pid")
  PID_TO_TASK["$pid"]="${strategy}:${method}"
}

declare -a RUN_PIDS=()
declare -a FAILED_RUNS=()
declare -A PID_TO_TASK=()

JOB_MEM_GB="$(estimate_job_mem_gb)"

echo "Config          : $CONFIG_NAME"
echo "Dataset         : $DATASET_NAME"
echo "Label tag       : $LABEL_TAG"
echo "n_vars          : $N_VARS"
echo "Strategies      : ${STRATEGIES[*]}"
echo "Methods         : ${METHODS_ARRAY[*]}"
echo "Score n_jobs    : $SCORE_N_JOBS"
echo "Score chunk     : $SCORE_CHUNK_SIZE"
echo "Max parallel    : $MAX_PARALLEL"
echo "Reserved mem GB : $RESERVED_MEM_GB"
echo "Estimate/job GB : $JOB_MEM_GB"
echo "Poll seconds    : $POLL_SECONDS"
echo

for strategy in "${STRATEGIES[@]}"; do
  label_path="datasets/traffic_${DATASET_NAME}/${strategy}_${N_VARS}_${LABEL_TAG}/${DATASET_NAME}.p"

  if [ ! -f "$label_path" ]; then
    echo "Missing label file: $label_path" >&2
    echo "Skip strategy: $strategy" >&2
    echo >&2
    continue
  fi

  for method in "${METHODS_ARRAY[@]}"; do
    override="$(method_override "$method")" || {
      FAILED_RUNS+=("${strategy}:${method}:unknown-method")
      continue
    }
    log_path="${LOG_DIR}/${strategy}_${N_VARS}_${LABEL_TAG}_${method}_eager.log"

    while true; do
      collect_finished_jobs

      current_running="$(running_count)"
      available_gb="$(mem_available_gb)"
      launchable_mem=$(( available_gb - RESERVED_MEM_GB ))

      if [ "$current_running" -lt "$MAX_PARALLEL" ] && [ "$launchable_mem" -ge "$JOB_MEM_GB" ]; then
        break
      fi

      echo "Waiting: running=${current_running}/${MAX_PARALLEL}, available=${available_gb}GB, required=${JOB_MEM_GB}GB, reserved=${RESERVED_MEM_GB}GB"
      sleep "$POLL_SECONDS"
    done

    launch_job "$strategy" "$method" "$label_path" "$log_path" "$override"
    echo
  done
done

while [ "$(running_count)" -gt 0 ]; do
  collect_finished_jobs
  if [ "$(running_count)" -gt 0 ]; then
    sleep "$POLL_SECONDS"
  fi
done

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All scheduled jobs completed successfully."
