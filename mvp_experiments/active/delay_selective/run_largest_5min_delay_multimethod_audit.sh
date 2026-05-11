#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" delay_selective/run_largest_5min_delay_multimethod_audit.py \
  --graph-variants distthre \
  --min-corr 0.8 \
  --methods mcc_5min_resid stdde_spline_fft_mcc lift_fft_abs \
  --windows month week_daily \
  "$@"

"${PYTHON_BIN}" delay_selective/plot_largest_5min_delay_multimethod_audit.py
