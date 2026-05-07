#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/zero_aware_mvp/results/naive_baselines}"
SUMMARY_CSV="${SUMMARY_CSV:-$REPO_ROOT/zero_aware_mvp/results/summary/mvp_ranking_with_masks.csv}"
DATASETS="${ZA_DATASETS:-TRAFFIC_VOLUME_FULL_5MIN TRAFFIC_VOLUME_FULL_15MIN TRAFFIC_VOLUME_FULL_30MIN TRAFFIC_VOLUME_FULL_1H}"
NODE_FILTERS="${NODE_FILTERS:-train_nonzero train_active24 full_nonzero}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$SUMMARY_CSV")"
cd "$REPO_ROOT"

for dataset_name in $DATASETS; do
  for node_filter in $NODE_FILTERS; do
    echo "[zero-aware] dataset=$dataset_name node_filter=$node_filter"
    "$PYTHON_BIN" zero_aware_mvp/scripts/run_naive_baselines.py \
      --dataset-name "$dataset_name" \
      --node-filter "$node_filter" \
      --output-dir "$OUTPUT_DIR" \
      "$@"
  done
done

"$PYTHON_BIN" zero_aware_mvp/scripts/collect_mvp_results.py \
  --results-root "$REPO_ROOT/zero_aware_mvp/results" \
  --output-csv "$SUMMARY_CSV"
