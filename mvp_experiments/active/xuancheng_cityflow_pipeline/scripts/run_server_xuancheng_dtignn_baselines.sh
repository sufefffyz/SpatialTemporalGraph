#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python}"
INPUT_NPZ="${INPUT_NPZ:-${DATA_ROOT}/dtignn_turn_1d_10s_repaired_nocycle/xuancheng_2023-04-03_dtignn_turn_10s_start0_dur86400.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/dtignn_baselines_daytime_10s}"
TIME_WINDOW_START="${TIME_WINDOW_START:-07:00}"
TIME_WINDOW_END="${TIME_WINDOW_END:-19:00}"
MISSING_RATIOS="${MISSING_RATIOS:-dense,0.1,0.3,0.5,0.7,0.9}"
AR_TRAIN_ROWS="${AR_TRAIN_ROWS:-300000}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" \
  mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/evaluate_dtignn_classic_baselines.py \
  --input-npz "${INPUT_NPZ}" \
  --output-dir "${OUTPUT_DIR}" \
  --time-window-start "${TIME_WINDOW_START}" \
  --time-window-end "${TIME_WINDOW_END}" \
  --missing-ratios "${MISSING_RATIOS}" \
  --ar-train-rows "${AR_TRAIN_ROWS}"
