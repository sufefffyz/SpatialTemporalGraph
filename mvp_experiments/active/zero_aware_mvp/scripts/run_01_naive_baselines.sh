#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
DATASET_NAME="${ZA_DATASET_NAME:-TRAFFIC_VOLUME_FULL_5MIN}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/zero_aware_mvp/results/naive_baselines}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" zero_aware_mvp/scripts/run_naive_baselines.py \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
