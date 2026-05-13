#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/outputs/dataset_profile}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/figures/dataset_profile}"
FORMATS="${FORMATS:-pdf,png}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_city_traffic_profile_figures.py" \
  --profile-dir "${PROFILE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --formats "${FORMATS}" \
  --target-y-scale "${TARGET_Y_SCALE:-log}" \
  --degree-y-scale "${DEGREE_Y_SCALE:-log}" \
  "${@}"
