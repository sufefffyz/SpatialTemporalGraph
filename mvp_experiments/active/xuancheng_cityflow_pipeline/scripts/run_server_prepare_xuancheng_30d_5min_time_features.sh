#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph}"
PYTHON="${PYTHON:-python3}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/BasicTS/datasets}"
SCRIPT="${SCRIPT:-${REPO_ROOT}/mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/add_basicts_time_features.py}"

INPUT_FLOW="${INPUT_FLOW:-${BASE_DIR}/XCHENG30D_5MIN_FLOW}"
OUTPUT_FLOW="${OUTPUT_FLOW:-${BASE_DIR}/XCHENG30D_5MIN_FLOW_TIME}"
INPUT_STOCK="${INPUT_STOCK:-${BASE_DIR}/XCHENG30D_5MIN_STOCK}"
OUTPUT_STOCK="${OUTPUT_STOCK:-${BASE_DIR}/XCHENG30D_5MIN_STOCK_TIME}"

echo "[info] building time-feature datasets"
echo "[info] flow:  ${INPUT_FLOW} -> ${OUTPUT_FLOW}"
echo "[info] stock: ${INPUT_STOCK} -> ${OUTPUT_STOCK}"

mkdir -p "$(dirname "${OUTPUT_FLOW}")"

"${PYTHON}" "${SCRIPT}" --input-dataset "${INPUT_FLOW}" --output-dataset "${OUTPUT_FLOW}"
"${PYTHON}" "${SCRIPT}" --input-dataset "${INPUT_STOCK}" --output-dataset "${OUTPUT_STOCK}"

echo "[done] time-feature datasets ready"
