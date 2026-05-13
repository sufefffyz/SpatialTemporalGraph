#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LARGEST_ROOT="${LARGEST_ROOT:-/data/yuzhang_fei/LargeST}"
OUTPUT_ROOT="${OUTPUT_ROOT:-BasicTS/datasets}"
GRAPH_ROOT="${GRAPH_ROOT:-graphs}"
YEAR="${YEAR:-2019}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_largest_5min_basicts.py" \
  --largest-root "${LARGEST_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --graph-root "${GRAPH_ROOT}" \
  --year "${YEAR}" \
  "${@}"
