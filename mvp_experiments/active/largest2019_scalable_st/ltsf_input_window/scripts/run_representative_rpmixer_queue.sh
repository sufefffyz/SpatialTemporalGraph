#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <GPU_ID>" >&2
  exit 2
fi

GPU_ID="$1"
REPO_ROOT="${REPO_ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph}"
RPMIXER_ROOT="${REPO_ROOT}/mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/third_party/RPMixer"
LOG_DIR="${REPO_ROOT}/mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/outputs/logs/fast_ltsf_representative_full"

mkdir -p "${LOG_DIR}"
cd "${RPMIXER_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RUNS=(
  "SD sd_96_48.py"
  "SD sd_96_96.py"
  "SD sd_96_192.py"
  "SD sd_96_672.py"
  "GBA gba_96_48.py"
  "GBA gba_96_96.py"
  "GBA gba_96_192.py"
  "GLA gla_96_48.py"
  "GLA gla_96_96.py"
)

for ITEM in "${RUNS[@]}"; do
  DATASET="${ITEM%% *}"
  SCRIPT="${ITEM#* }"
  HORIZON="${SCRIPT##*_}"
  HORIZON="${HORIZON%.py}"
  LOG="${LOG_DIR}/RPMixer_${DATASET}_L96_H${HORIZON}.log"
  echo "[$(date '+%F %T')] START RPMixer ${DATASET} L96 H${HORIZON}" | tee -a "${LOG}"
  python "${SCRIPT}" --n_worker 4 2>&1 | tee -a "${LOG}"
  echo "[$(date '+%F %T')] END RPMixer ${DATASET} L96 H${HORIZON}" | tee -a "${LOG}"
done
