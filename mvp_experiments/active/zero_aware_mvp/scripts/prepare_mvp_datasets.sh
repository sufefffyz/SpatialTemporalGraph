#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${ACTIVE_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
RESOLUTIONS="${RESOLUTIONS:-5min 15min 30min 60min}"
INPUT_NPZ="${INPUT_NPZ:-/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz}"
DATASET_BASE_NAME="${DATASET_BASE_NAME:-TRAFFIC_VOLUME_FULL_5MIN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/BasicTS/datasets}"
ADJ_MODE="${ADJ_MODE:-none}"

cd "$REPO_ROOT/BasicTS"

for resolution in $RESOLUTIONS; do
  echo "[zero-aware] preparing TRAFFIC_VOLUME_$(printf '%s' "$resolution" | tr '[:lower:]' '[:upper:]')"
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "${SCRIPT_DIR}/prepare_basicts_traffic_volume.py" \
    --input-npz "$INPUT_NPZ" \
    --output-root "$OUTPUT_ROOT" \
    --base-name "$DATASET_BASE_NAME" \
    --adj-mode "$ADJ_MODE" \
    --resolution "$resolution"
done
