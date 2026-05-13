#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_quick_delay_audit.py" \
  --dataset-npz data/city_traffic_m_speed__category__1_0.npz \
  --targets-npy product/traffic_city_traffic_m_speed__category__1_0/targets.npy \
  --output-dir "${SCRIPT_DIR}/outputs/quick_delay_audit" \
  --split train \
  --max-time-steps 10000 \
  --max-edges 1500 \
  --max-lag 12 \
  --min-pair-coverage 0.80 \
  --residualize time_of_day \
  --min-corr 0.20 \
  --min-improvement 0.03 \
  --seed 42
