#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/dtignn_turn_1d_10s_repaired_nocycle}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/dtignn_state_active_lsr_1d_10s}"
PATTERN="${PATTERN:-xuancheng_*_dtignn_turn_10s_start0_dur86400.npz}"
FEATURE_KEY="${FEATURE_KEY:-active_volume_lsr}"
INPUT_WINDOW="${INPUT_WINDOW:-30}"
HORIZON="${HORIZON:-1}"
SPLIT_MODE="${SPLIT_MODE:-chronological}"
MISSING_RATIO="${MISSING_RATIO:-dense}"
COMPACT_ONLY="${COMPACT_ONLY:-1}"
SAVE_NPZ_SPLIT="${SAVE_NPZ_SPLIT:-1}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUTPUT_DIR}"

echo "[info] Preparing Xuancheng DTIGNN-state dataset"
echo "[info] input_dir=${INPUT_DIR}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] feature_key=${FEATURE_KEY}"
echo "[info] input_window=${INPUT_WINDOW} horizon=${HORIZON} split_mode=${SPLIT_MODE}"
echo "[info] missing_ratio=${MISSING_RATIO}"
echo "[info] compact_only=${COMPACT_ONLY}"
echo "[info] python=${PYTHON_BIN}"

ARGS=(
  "${SCRIPT_DIR}/prepare_xuancheng_dtignn_state_dataset.py"
  --input-dir "${INPUT_DIR}"
  --pattern "${PATTERN}"
  --output-dir "${OUTPUT_DIR}"
  --feature-key "${FEATURE_KEY}"
  --input-window "${INPUT_WINDOW}"
  --horizon "${HORIZON}"
  --split-mode "${SPLIT_MODE}"
  --missing-ratio "${MISSING_RATIO}"
)

if [[ "${SAVE_NPZ_SPLIT}" == "1" ]]; then
  ARGS+=(--save-npz-split)
fi
if [[ "${COMPACT_ONLY}" == "1" ]]; then
  ARGS+=(--compact-only)
fi
if [[ "${APPLY_OFFICIAL_MASK_OP:-0}" == "1" ]]; then
  ARGS+=(--apply-official-mask-op)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
