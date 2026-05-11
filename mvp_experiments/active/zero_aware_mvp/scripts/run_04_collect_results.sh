#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"

mkdir -p "$REPO_ROOT/zero_aware_mvp/results/summary"
cd "$REPO_ROOT"

"$PYTHON_BIN" zero_aware_mvp/scripts/collect_mvp_results.py \
  --results-root "$REPO_ROOT/zero_aware_mvp/results" \
  --output-csv "$REPO_ROOT/zero_aware_mvp/results/summary/mvp_ranking.csv"

