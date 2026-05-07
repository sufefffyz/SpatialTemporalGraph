#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
RESOLUTIONS="${RESOLUTIONS:-5min 15min 30min 60min}"
INPUT_NPZ="${INPUT_NPZ:-/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz}"
DATASET_BASE_NAME="${DATASET_BASE_NAME:-TRAFFIC_VOLUME_FULL_5MIN}"

cd "$REPO_ROOT/BasicTS"

for resolution in $RESOLUTIONS; do
  echo "[zero-aware] preparing TRAFFIC_VOLUME_$(printf '%s' "$resolution" | tr '[:lower:]' '[:upper:]')"
  cd "$REPO_ROOT"
  "$PYTHON_BIN" zero_aware_mvp/scripts/prepare_basicts_traffic_volume.py \
    --input-npz "$INPUT_NPZ" \
    --output-root "$REPO_ROOT/BasicTS/datasets" \
    --base-name "$DATASET_BASE_NAME" \
    --resolution "$resolution"
done
