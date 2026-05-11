#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE_DIR="${PROFILE_DIR:-delay_selective/outputs/dataset_profile}"
OUTPUT_DIR="${OUTPUT_DIR:-delay_selective/figures/dataset_profile}"
FORMATS="${FORMATS:-pdf,png}"

"${PYTHON_BIN}" delay_selective/plot_city_traffic_profile_figures.py \
  --profile-dir "${PROFILE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --formats "${FORMATS}" \
  --target-y-scale "${TARGET_Y_SCALE:-log}" \
  --degree-y-scale "${DEGREE_Y_SCALE:-log}" \
  "${@}"
