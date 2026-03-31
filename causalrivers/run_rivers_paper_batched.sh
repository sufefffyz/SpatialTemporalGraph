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

FAILED_BATCHES=()

run_batch() {
  local batch_name="$1"
  local methods="$2"
  local log_path="$LOG_DIR/${batch_name}.batch.log"

  if [ -z "$methods" ]; then
    echo "Skip empty batch: $batch_name"
    return 0
  fi

  echo "========================================"
  echo "Batch      : $batch_name"
  echo "Methods    : $methods"
  echo "Strategies : ${STRATEGIES[*]}"
  echo "n_vars     : $N_VARS_LIST"
  echo "Regions    : $REGIONS"
  echo "Log        : $log_path"
  echo "========================================"

  if PYTHON_BIN="$PYTHON_BIN" \
    CONFIG_NAME="$CONFIG_NAME" \
    SCORE_N_JOBS="$SCORE_N_JOBS" \
    SCORE_CHUNK_SIZE="$SCORE_CHUNK_SIZE" \
    LOG_DIR="$LOG_DIR" \
    N_VARS_LIST="$N_VARS_LIST" \
    REGIONS="$REGIONS" \
    METHODS="$methods" \
    bash "$SCRIPT_DIR/run_rivers_paper_eager.sh" "${STRATEGIES[@]}" \
    2>&1 | tee "$log_path"; then
    echo "Batch finished: $batch_name"
  else
    local status=$?
    echo "Batch failed: $batch_name (exit=${status})" >&2
    FAILED_BATCHES+=("${batch_name}:exit-${status}")
  fi

  echo
}

echo "Running rivers benchmark in separated batches"
echo "Strategies      : ${STRATEGIES[*]}"
echo "N_VARS_LIST     : $N_VARS_LIST"
echo "Regions         : $REGIONS"
echo "FAST_METHODS    : $FAST_METHODS"
echo "PCMCI_METHODS   : $PCMCI_METHODS"
echo "VARLINGAM_METHODS: $VARLINGAM_METHODS"
echo

run_batch "rivers_fast" "$FAST_METHODS"
run_batch "rivers_pcmci" "$PCMCI_METHODS"
run_batch "rivers_varlingam" "$VARLINGAM_METHODS"

if [ "${#FAILED_BATCHES[@]}" -gt 0 ]; then
  echo "Completed with failed batches:" >&2
  printf '  %s\n' "${FAILED_BATCHES[@]}" >&2
  exit 1
fi

echo "All rivers batches completed successfully."
