#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash zero_aware_mvp/scripts/run_03_posthoc_eval.sh <BasicTS checkpoint dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
DATASET_NAME="${ZA_DATASET_NAME:-TRAFFIC_VOLUME_FULL_5MIN}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/zero_aware_mvp/results/posthoc}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" zero_aware_mvp/scripts/evaluate_predictions.py \
  --dataset-name "$DATASET_NAME" \
  --ckpt-dir "$1" \
  --output-dir "$OUTPUT_DIR"
