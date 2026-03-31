#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark}"
SCORE_N_JOBS="${SCORE_N_JOBS:-1}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"
METHODS="${METHODS:-var cc rp pcmci varlingam}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESERVED_MEM_GB="${RESERVED_MEM_GB:-64}"
POLL_SECONDS="${POLL_SECONDS:-15}"
ESTIMATED_JOB_MEM_GB="${ESTIMATED_JOB_MEM_GB:-}"
N_VARS_LIST="${N_VARS_LIST:-3 5}"
REGIONS="${REGIONS:-east bav flood}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(debug_set random 1_random root_cause confounder close)
fi

read -r -a METHODS_ARRAY <<< "$METHODS"
read -r -a N_VARS_ARRAY <<< "$N_VARS_LIST"
read -r -a REGION_ARRAY <<< "$REGIONS"

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
    pcmci)
      echo 'methods=[{name:pcmci,max_lag:3,map_to_summary_graph:max,pcmci_absolute_values:true,pcmci_pc_alpha:0.05,pcmci_alpha_level:0.05,pcmci_filter_nonsignificant:true,pcmci_verbosity:0}]'
      ;;
    varlingam)
      echo 'methods=[{name:varlingam,max_lag:3,map_to_summary_graph:max,varlingam_absolute_values:true,varlingam_criterion:null,varlingam_prune:true,varlingam_random_state:0}]'
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

region_data_path() {
  local region="$1"
  case "$region" in
    east)
      echo "product/rivers_ts_east_germany.csv"
      ;;
    bav)
      echo "product/rivers_ts_bavaria.csv"
      ;;
    flood)
      echo "product/rivers_ts_flood.csv"
      ;;
    *)
      echo "Unknown region: $region" >&2
      return 1
      ;;
  esac
}

estimate_job_mem_gb() {
  if [ -n "$ESTIMATED_JOB_MEM_GB" ]; then
    echo "$ESTIMATED_JOB_MEM_GB"
    return
  fi

  case "$N_VARS_LIST" in
    "3") echo "8" ;;
    "5") echo "16" ;;
    *) echo "16" ;;
  esac
}

mem_available_gb() {
  awk '/MemAvailable:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo
}

running_count() {
  local count=0
  for pid in "${RUN_PIDS[@]}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

remove_pid() {
  local target="$1"
  local next=()
  for pid in "${RUN_PIDS[@]}"; do
    if [ -n "$pid" ] && [ "$pid" != "$target" ]; then
      next+=("$pid")
    fi
  done
  RUN_PIDS=("${next[@]}")
}

collect_finished_jobs() {
  local pid status task
  for pid in "${RUN_PIDS[@]}"; do
    if [ -z "$pid" ]; then
      continue
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      status=$?
      task="${PID_TO_TASK[$pid]:-${pid}}"
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
  local region="$1"
  local strategy="$2"
  local n_vars="$3"
  local method="$4"
  local label_path="$5"
  local data_path="$6"
  local log_path="$7"
  local override="$8"

  echo "Launching region=$region strategy=$strategy n_vars=$n_vars method=$method"
  echo "  label_path : $label_path"
  echo "  data_path  : $data_path"
  echo "  log        : $log_path"

  (
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" benchmark.py --config-name "$CONFIG_NAME" \
      load_mode=eager \
      n_jobs=1 \
      score_n_jobs="$SCORE_N_JOBS" \
      score_chunk_size="$SCORE_CHUNK_SIZE" \
      label_path="$label_path" \
      data_path="$data_path" \
      "$override"
  ) >"$log_path" 2>&1 &

  local pid=$!
  RUN_PIDS+=("$pid")
  PID_TO_TASK["$pid"]="${region}:${strategy}:${n_vars}:${method}"
}

label_count() {
  local label_path="$1"
  "$PYTHON_BIN" -c 'import pickle, sys; print(len(pickle.load(open(sys.argv[1], "rb"))))' "$label_path"
}

declare -a RUN_PIDS=()
declare -a FAILED_RUNS=()
declare -a SKIPPED_RUNS=()
declare -A PID_TO_TASK=()

JOB_MEM_GB="$(estimate_job_mem_gb)"

echo "Config          : $CONFIG_NAME"
echo "Regions         : ${REGION_ARRAY[*]}"
echo "Strategies      : ${STRATEGIES[*]}"
echo "n_vars list     : ${N_VARS_ARRAY[*]}"
echo "Methods         : ${METHODS_ARRAY[*]}"
echo "Score n_jobs    : $SCORE_N_JOBS"
echo "Score chunk     : $SCORE_CHUNK_SIZE"
echo "Max parallel    : $MAX_PARALLEL"
echo "Reserved mem GB : $RESERVED_MEM_GB"
echo "Estimate/job GB : $JOB_MEM_GB"
echo "Poll seconds    : $POLL_SECONDS"
echo

for n_vars in "${N_VARS_ARRAY[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    label_dir="datasets/${strategy}_${n_vars}"
    if [ ! -d "$label_dir" ]; then
      echo "Missing label directory: $label_dir" >&2
      SKIPPED_RUNS+=("${label_dir}:missing-dir")
      continue
    fi

    for region in "${REGION_ARRAY[@]}"; do
      label_path="${label_dir}/${region}.p"
      data_path="$(region_data_path "$region")" || {
        FAILED_RUNS+=("${region}:${strategy}:${n_vars}:unknown-region")
        continue
      }

      if [ ! -f "$label_path" ]; then
        echo "Missing label file: $label_path" >&2
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:missing-label")
        continue
      fi

      if [ ! -f "$data_path" ]; then
        echo "Missing data file: $data_path" >&2
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:missing-data")
        continue
      fi

      count="$(label_count "$label_path" 2>/dev/null || echo -1)"
      if [ "$count" -le 0 ]; then
        echo "Skipping empty label set: $label_path"
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:empty-labels")
        continue
      fi

      for method in "${METHODS_ARRAY[@]}"; do
        override="$(method_override "$method")" || {
          FAILED_RUNS+=("${region}:${strategy}:${n_vars}:${method}:unknown-method")
          continue
        }
        log_path="${LOG_DIR}/rivers_${region}_${strategy}_${n_vars}_${method}_eager.log"

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

        launch_job "$region" "$strategy" "$n_vars" "$method" "$label_path" "$data_path" "$log_path" "$override"
        echo
      done
    done
  done
done

while [ "$(running_count)" -gt 0 ]; do
  collect_finished_jobs
  if [ "$(running_count)" -gt 0 ]; then
    sleep "$POLL_SECONDS"
  fi
done

if [ "${#SKIPPED_RUNS[@]}" -gt 0 ]; then
  echo "Skipped runs:"
  printf '  %s\n' "${SKIPPED_RUNS[@]}"
  echo
fi

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All scheduled rivers jobs completed successfully."
