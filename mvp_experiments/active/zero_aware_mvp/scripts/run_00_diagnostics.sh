#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${ACTIVE_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/diagnostics}"
INPUT_NPZ="${INPUT_NPZ:-/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" "${SCRIPT_DIR}/zero_diagnostics.py" \
  --input-npz "$INPUT_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
