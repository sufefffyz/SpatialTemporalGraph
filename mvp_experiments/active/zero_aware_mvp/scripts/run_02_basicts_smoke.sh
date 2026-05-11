#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASICTS_PYTHON="${BASICTS_PYTHON:-python}"
ZA_MODEL="${ZA_MODEL:-GWNET}"
ZA_NUM_EPOCHS="${ZA_NUM_EPOCHS:-3}"
ZA_DATASET_NAME="${ZA_DATASET_NAME:-TRAFFIC_VOLUME_FULL_5MIN}"
ZA_GPUS="${ZA_GPUS:-0}"

case "$ZA_MODEL" in
  AGCRN)
    CFG="../zero_aware_mvp/configs/AGCRN_TRAFFIC_VOLUME_ZERO_MVP.py"
    ;;
  GWNET|GWNet|gwnet)
    CFG="../zero_aware_mvp/configs/GWNET_TRAFFIC_VOLUME_ZERO_MVP.py"
    ;;
  *)
    echo "Unsupported ZA_MODEL=$ZA_MODEL. Use AGCRN or GWNET." >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT/BasicTS"
echo "[zero-aware] BasicTS smoke: model=$ZA_MODEL dataset=$ZA_DATASET_NAME epochs=$ZA_NUM_EPOCHS"
ZA_NUM_EPOCHS="$ZA_NUM_EPOCHS" ZA_DATASET_NAME="$ZA_DATASET_NAME" \
  "$BASICTS_PYTHON" experiments/train.py -c "$CFG" -g "$ZA_GPUS"
