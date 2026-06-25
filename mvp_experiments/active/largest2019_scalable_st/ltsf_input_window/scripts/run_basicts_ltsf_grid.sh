#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$ROOT_DIR/mvp_experiments/active/largest2019_scalable_st/ltsf_input_window"
BASICTS_DIR="$ROOT_DIR/BasicTS"

PHASE="${PHASE:-smoke}"
DATASETS="${DATASETS:-SD}"
MODELS="${MODELS:-STID,DLinear,CycleNet,TimeMixer,PatchTST}"
INPUT_LENGTHS="${INPUT_LENGTHS:-96,192,336,672}"
HORIZON="${HORIZON:-672}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$PHASE" in
  smoke|smoke_*)
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    PATIENCE="${PATIENCE:-1}"
    RUN_TAG="${RUN_TAG:-ltsf_${PHASE}}"
    ;;
  pilot|pilot_*)
    NUM_EPOCHS="${NUM_EPOCHS:-20}"
    PATIENCE="${PATIENCE:-10}"
    RUN_TAG="${RUN_TAG:-ltsf_${PHASE}}"
    ;;
  full|full_*)
    NUM_EPOCHS="${NUM_EPOCHS:-50}"
    PATIENCE="${PATIENCE:-10}"
    RUN_TAG="${RUN_TAG:-ltsf_${PHASE}}"
    ;;
  *)
    echo "Unknown PHASE=$PHASE. Use smoke, pilot, full, or a suffixed variant such as smoke_h12." >&2
    exit 2
    ;;
esac

CONFIG_DIR="$BASICTS_DIR/baselines/LargeSTLTSF/generated/$PHASE"
LOG_DIR="$EXP_DIR/outputs/logs/basicts/$PHASE"
MEM_DIR="$EXP_DIR/outputs/gpu_memory/basicts/$PHASE"
RUN_LIST="$EXP_DIR/outputs/generated_lists/basicts_${PHASE}_$(date +%Y%m%dT%H%M%S)_$$.txt"
mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$MEM_DIR"
mkdir -p "$(dirname "$RUN_LIST")"

"$PYTHON_BIN" "$EXP_DIR/scripts/generate_basicts_ltsf_configs.py" \
  --datasets "$DATASETS" \
  --models "$MODELS" \
  --input-lengths "$INPUT_LENGTHS" \
  --horizon "$HORIZON" \
  --num-epochs "$NUM_EPOCHS" \
  --run-tag "$RUN_TAG" \
  --out-dir "$CONFIG_DIR" \
  > "$RUN_LIST"

run_one() {
  local cfg="$1"
  local stem
  stem="$(basename "$cfg" .py)"
  local log="$LOG_DIR/${stem}.log"
  local mem="$MEM_DIR/${stem}.csv"
  echo "==> [$PHASE] $stem on GPU $GPU"
  echo "timestamp_ms,memory_used_mib" > "$mem"
  (
    cd "$BASICTS_DIR"
    local cfg_for_train
    cfg_for_train="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$cfg" "$BASICTS_DIR")"
    export LARGEST_LTSF_NUM_EPOCHS="$NUM_EPOCHS"
    export LARGEST_LTSF_PATIENCE="$PATIENCE"
    export LARGEST_LTSF_RUN_TAG="$RUN_TAG"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    local start_ts end_ts status
    start_ts="$(date +%s)"
    "$PYTHON_BIN" experiments/train.py -c "$cfg_for_train" -g "$GPU" > "$log" 2>&1 &
    local train_pid=$!
    while kill -0 "$train_pid" 2>/dev/null; do
      if command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s,' "$(date +%s%3N)" >> "$mem"
        nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits >> "$mem" || true
      fi
      sleep "${GPU_SAMPLE_INTERVAL:-5}"
    done
    set +e
    wait "$train_pid"
    status=$?
    set -e
    end_ts="$(date +%s)"
    echo "WALL_SECONDS=$((end_ts - start_ts))" >> "$log"
    echo "EXIT_STATUS=$status" >> "$log"
    return "$status"
  )
}

while IFS= read -r cfg; do
  run_one "$cfg"
done < "$RUN_LIST"

echo "All BasicTS $PHASE jobs finished. Logs: $LOG_DIR"
