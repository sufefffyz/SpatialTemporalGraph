#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASICTS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$BASICTS_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"
FORCE_RERUN="${FORCE_RERUN:-0}"
LOG_DIR="${LOG_DIR:-logs/stgraph_resume}"
MODELS="${MODELS:-AGCRN D2STGNN GTS MTGNN GWNET}"
DATASETS="${DATASETS:-TRAFFIC_VOLUME_5MIN RIVERS_EAST_GERMANY_15MIN}"

INPUT_LEN=12
OUTPUT_LEN=12

mkdir -p "$LOG_DIR"

model_dir_for() {
  case "$1" in
    GWNET) echo "GraphWaveNet" ;;
    *) echo "$1" ;;
  esac
}

config_path_for() {
  local model="$1"
  local dataset="$2"
  case "$dataset" in
    TRAFFIC_VOLUME_5MIN)
      echo "stgraph_ext/configs/${model}_TRAFFIC_VOLUME_5MIN.py"
      ;;
    RIVERS_EAST_GERMANY_15MIN)
      echo "stgraph_ext/configs/${model}_RIVERS_EAST_GERMANY_15MIN.py"
      ;;
    *)
      echo "Unsupported dataset for resume script: $dataset" >&2
      return 1
      ;;
  esac
}

metrics_path_for() {
  local model="$1"
  local dataset="$2"
  local model_dir
  model_dir="$(model_dir_for "$model")"
  echo "checkpoints/${model_dir}/${dataset}_${EPOCHS}_${INPUT_LEN}_${OUTPUT_LEN}/test_metrics.json"
}

graph_snapshot_for() {
  local model="$1"
  local dataset="$2"
  local model_dir
  model_dir="$(model_dir_for "$model")"
  echo "checkpoints/${model_dir}/${dataset}_${EPOCHS}_${INPUT_LEN}_${OUTPUT_LEN}/learned_graphs/final.npz"
}

read -r -a MODEL_ARRAY <<< "$MODELS"
read -r -a DATASET_ARRAY <<< "$DATASETS"

echo "Python        : $PYTHON_BIN"
echo "GPU           : $GPU_ID"
echo "Epochs        : $EPOCHS"
echo "Force rerun   : $FORCE_RERUN"
echo "Models        : ${MODEL_ARRAY[*]}"
echo "Datasets      : ${DATASET_ARRAY[*]}"
echo "Log dir       : $LOG_DIR"
echo

for dataset in "${DATASET_ARRAY[@]}"; do
  for model in "${MODEL_ARRAY[@]}"; do
    config_path="$(config_path_for "$model" "$dataset")"
    metrics_path="$(metrics_path_for "$model" "$dataset")"
    graph_path="$(graph_snapshot_for "$model" "$dataset")"
    log_path="${LOG_DIR}/${dataset}_${model}_e${EPOCHS}.log"

    if [ "$FORCE_RERUN" != "1" ] && [ -f "$metrics_path" ] && [ -f "$graph_path" ]; then
      echo "Skip existing: dataset=$dataset model=$model"
      echo "  metrics: $metrics_path"
      echo "  graph  : $graph_path"
      echo
      continue
    fi

    echo "Running: dataset=$dataset model=$model"
    echo "  config: $config_path"
    echo "  metrics: $metrics_path"
    echo "  log    : $log_path"

    STGRAPH_DATASET_NAME="$dataset" STGRAPH_NUM_EPOCHS="$EPOCHS" \
      "$PYTHON_BIN" experiments/train.py \
        -c "$config_path" \
        -g "$GPU_ID" >"$log_path" 2>&1

    echo "Finished: dataset=$dataset model=$model"
    echo
  done
done

echo "All requested runs are complete."
