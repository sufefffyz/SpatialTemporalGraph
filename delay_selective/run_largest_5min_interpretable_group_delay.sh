#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" delay_selective/run_largest_5min_interpretable_group_delay.py "$@"
