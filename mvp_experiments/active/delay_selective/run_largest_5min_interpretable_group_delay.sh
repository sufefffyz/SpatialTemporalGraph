#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_largest_5min_interpretable_group_delay.py" "$@"
