#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${ACTIVE_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/naive_baselines}"
SUMMARY_CSV="${SUMMARY_CSV:-$EXPERIMENT_DIR/results/summary/mvp_ranking_with_masks.csv}"
DATASETS="${ZA_DATASETS:-TRAFFIC_VOLUME_FULL_5MIN TRAFFIC_VOLUME_FULL_15MIN TRAFFIC_VOLUME_FULL_30MIN TRAFFIC_VOLUME_FULL_1H}"
NODE_FILTERS="${NODE_FILTERS:-train_nonzero train_active24 full_nonzero}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$SUMMARY_CSV")"
cd "$REPO_ROOT"

for dataset_name in $DATASETS; do
  for node_filter in $NODE_FILTERS; do
    echo "[zero-aware] dataset=$dataset_name node_filter=$node_filter"
    "$PYTHON_BIN" "${SCRIPT_DIR}/run_naive_baselines.py" \
      --dataset-name "$dataset_name" \
      --node-filter "$node_filter" \
      --output-dir "$OUTPUT_DIR" \
      "$@"
  done
done

"$PYTHON_BIN" "${SCRIPT_DIR}/collect_mvp_results.py" \
  --results-root "$EXPERIMENT_DIR/results" \
  --output-csv "$SUMMARY_CSV"
