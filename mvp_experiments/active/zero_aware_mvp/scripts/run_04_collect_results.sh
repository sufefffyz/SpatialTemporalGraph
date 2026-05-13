#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${ACTIVE_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"

mkdir -p "$EXPERIMENT_DIR/results/summary"
cd "$REPO_ROOT"

"$PYTHON_BIN" "${SCRIPT_DIR}/collect_mvp_results.py" \
  --results-root "$EXPERIMENT_DIR/results" \
  --output-csv "$EXPERIMENT_DIR/results/summary/mvp_ranking.csv"
