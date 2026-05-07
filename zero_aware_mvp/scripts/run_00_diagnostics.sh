#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/zero_aware_mvp/results/diagnostics}"
INPUT_NPZ="${INPUT_NPZ:-/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" zero_aware_mvp/scripts/zero_diagnostics.py \
  --input-npz "$INPUT_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
