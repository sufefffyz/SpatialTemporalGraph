#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark}"
SCORE_N_JOBS="${SCORE_N_JOBS:-1}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"
N_VARS_LIST="${N_VARS_LIST:-3 5}"
REGIONS="${REGIONS:-east bav flood}"

FAST_METHODS="${FAST_METHODS:-var cc rp}"
PCMCI_METHODS="${PCMCI_METHODS:-pcmci}"
VARLINGAM_METHODS="${VARLINGAM_METHODS:-varlingam}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(debug_set random 1_random root_cause confounder close)
fi

declare -a RUN_PIDS=()
declare -a FAILED_BATCHES=()
declare -A PID_TO_BATCH=()

launch_group() {
  local batch_name="$1"
  local methods="$2"
  local log_path="$LOG_DIR/${batch_name}.parallel.log"

  if [ -z "$methods" ]; then
    echo "Skip empty batch: $batch_name"
    return 0
  fi

  echo "Launching batch : $batch_name"
  echo "Methods         : $methods"
  echo "Strategies      : ${STRATEGIES[*]}"
  echo "n_vars          : $N_VARS_LIST"
  echo "Regions         : $REGIONS"
  echo "Log             : $log_path"

  (
    PYTHON_BIN="$PYTHON_BIN" \
    CONFIG_NAME="$CONFIG_NAME" \
    SCORE_N_JOBS="$SCORE_N_JOBS" \
    SCORE_CHUNK_SIZE="$SCORE_CHUNK_SIZE" \
    LOG_DIR="$LOG_DIR" \
    N_VARS_LIST="$N_VARS_LIST" \
    REGIONS="$REGIONS" \
    METHODS="$methods" \
    bash "$SCRIPT_DIR/run_rivers_paper_eager.sh" "${STRATEGIES[@]}"
  ) >"$log_path" 2>&1 &

  local pid=$!
  RUN_PIDS+=("$pid")
  PID_TO_BATCH["$pid"]="$batch_name"
  echo "PID             : $pid"
  echo
}

collect_finished_jobs() {
  local next=()
  local pid batch status
  for pid in "${RUN_PIDS[@]}"; do
    if [ -z "$pid" ]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      next+=("$pid")
      continue
    fi

    if wait "$pid"; then
      batch="${PID_TO_BATCH[$pid]:-$pid}"
      echo "Finished batch: $batch"
    else
      status=$?
      batch="${PID_TO_BATCH[$pid]:-$pid}"
      echo "Failed batch: $batch (exit=${status})" >&2
      FAILED_BATCHES+=("${batch}:exit-${status}")
    fi
    unset "PID_TO_BATCH[$pid]"
  done
  RUN_PIDS=("${next[@]}")
}

echo "Running rivers benchmark as three parallel groups"
echo "Strategies       : ${STRATEGIES[*]}"
echo "N_VARS_LIST      : $N_VARS_LIST"
echo "Regions          : $REGIONS"
echo "FAST_METHODS     : $FAST_METHODS"
echo "PCMCI_METHODS    : $PCMCI_METHODS"
echo "VARLINGAM_METHODS: $VARLINGAM_METHODS"
echo

launch_group "rivers_fast" "$FAST_METHODS"
launch_group "rivers_pcmci" "$PCMCI_METHODS"
launch_group "rivers_varlingam" "$VARLINGAM_METHODS"

while [ "${#RUN_PIDS[@]}" -gt 0 ]; do
  collect_finished_jobs
  if [ "${#RUN_PIDS[@]}" -gt 0 ]; then
    sleep 5
  fi
done

if [ "${#FAILED_BATCHES[@]}" -gt 0 ]; then
  echo "Completed with failed batches:" >&2
  printf '  %s\n' "${FAILED_BATCHES[@]}" >&2
  exit 1
fi

echo "All rivers parallel groups completed successfully."
