#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$ROOT_DIR/mvp_experiments/active/largest2019_scalable_st/ltsf_input_window"
LARGEST_DIR="$ROOT_DIR/LargeST"

PHASE="${PHASE:-smoke}"
DATASETS="${DATASETS:-SD}"
INPUT_LENGTHS="${INPUT_LENGTHS:-96,192,336,672}"
HORIZON="${HORIZON:-672}"
GPU="${GPU:-0}"
BS="${BS:-64}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$PHASE" in
  smoke|smoke_*)
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    PATIENCE="${PATIENCE:-1}"
    ;;
  pilot|pilot_*)
    MAX_EPOCHS="${MAX_EPOCHS:-20}"
    PATIENCE="${PATIENCE:-10}"
    ;;
  full|full_*)
    MAX_EPOCHS="${MAX_EPOCHS:-50}"
    PATIENCE="${PATIENCE:-10}"
    ;;
  *)
    echo "Unknown PHASE=$PHASE. Use smoke, pilot, full, or a suffixed variant such as smoke_h12." >&2
    exit 2
    ;;
esac

LOG_DIR="$EXP_DIR/outputs/logs/bist/$PHASE"
MEM_DIR="$EXP_DIR/outputs/gpu_memory/bist/$PHASE"
mkdir -p "$LOG_DIR" "$MEM_DIR"

core_for_dataset() {
  case "$1" in
    SD) echo 8 ;;
    GBA) echo 24 ;;
    GLA) echo 32 ;;
    CA) echo 64 ;;
    *) echo "Unknown dataset $1" >&2; return 2 ;;
  esac
}

lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

run_one() {
  local dataset="$1"
  local input_len="$2"
  local tag="2019_L${input_len}_H${HORIZON}"
  local dataset_lower
  dataset_lower="$(lower "$dataset")"
  local cache_dir="$LARGEST_DIR/data/$dataset_lower/$tag"
  if [[ ! -f "$cache_dir/his.npz" ]]; then
    echo "Missing BiST cache: $cache_dir/his.npz" >&2
    echo "Create it with prepare_largest_ltsf_npz.py before launching BiST." >&2
    return 3
  fi

  local core
  core="$(core_for_dataset "$dataset")"
  local stem="BiST_${dataset}_L${input_len}_H${HORIZON}_${PHASE}"
  local log="$LOG_DIR/${stem}.log"
  local mem="$MEM_DIR/${stem}.csv"
  echo "==> [$PHASE] $stem on GPU $GPU"
  echo "timestamp_ms,memory_used_mib" > "$mem"
  (
    cd "$LARGEST_DIR"
    local start_ts end_ts status
    start_ts="$(date +%s)"
    "$PYTHON_BIN" experiments/bist/main.py \
      --dataset "$dataset" \
      --kernel_size 3 \
      --device "cuda:$GPU" \
      --bs "$BS" \
      --model_name "bist_ltsf_${PHASE}_L${input_len}_H${HORIZON}" \
      --core "$core" \
      --seq_len "$input_len" \
      --horizon "$HORIZON" \
      --years "$tag" \
      --max_epochs "$MAX_EPOCHS" \
      --patience "$PATIENCE" \
      > "$log" 2>&1 &
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

IFS=',' read -r -a dataset_arr <<< "$DATASETS"
IFS=',' read -r -a input_arr <<< "$INPUT_LENGTHS"
for dataset in "${dataset_arr[@]}"; do
  for input_len in "${input_arr[@]}"; do
    run_one "$dataset" "$input_len"
  done
done

echo "All BiST $PHASE jobs finished. Logs: $LOG_DIR"
