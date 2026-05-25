#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
INPUT_NPZ="${INPUT_NPZ:-${DATA_ROOT}/dtignn_turn_1d_10s_repaired_nocycle/xuancheng_2023-04-03_dtignn_turn_10s_start0_dur86400.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/BasicTS/datasets}"
PREFIX="${PREFIX:-XCHENG}"
TASKS="${TASKS:-flow,stock}"
RESOLUTIONS="${RESOLUTIONS:-10s,5min}"
PYTHON_BIN="${PYTHON_BIN:-/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] Preparing Xuancheng BasicTS tasks"
echo "[info] input_npz=${INPUT_NPZ}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] prefix=${PREFIX} tasks=${TASKS} resolutions=${RESOLUTIONS}"
echo "[info] python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_xuancheng_basicts_tasks.py" \
  --input-npz "${INPUT_NPZ}" \
  --output-root "${OUTPUT_ROOT}" \
  --prefix "${PREFIX}" \
  --tasks "${TASKS}" \
  --resolutions "${RESOLUTIONS}"
