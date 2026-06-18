#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"
PYTHON_BIN="${PYTHON:-python3}"
GPU="${GPU:-0}"
MODELS="${MODELS:-AGCRN MTGNN}"
DATASETS="${DATASETS:-METR-LA PEMS04}"
EPOCHS="${EPOCHS:-5}"
RUN_STAGE="${RUN_STAGE:-smoke}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
AGCRN_VARIANTS="${AGCRN_VARIANTS:-original identity_support frozen_random_embedding graph_no_relu}"
MTGNN_VARIANTS="${MTGNN_VARIANTS:-original fixed_physical frozen_random_graph no_relu_score}"

export WANDB_PROJECT="${WANDB_PROJECT:-adaptive_graph_importance_ablation}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export BASICTS_DETERMINISTIC="${BASICTS_DETERMINISTIC:-0}"
export BASICTS_CUDNN_DETERMINISTIC="${BASICTS_CUDNN_DETERMINISTIC:-0}"
export BASICTS_CUDNN_BENCHMARK="${BASICTS_CUDNN_BENCHMARK:-1}"
export BASICTS_ENV_TAG="${BASICTS_ENV_TAG:-det0_cudnndet0}"
export BASICTS_EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}"

cd "${BASICTS_ROOT}"

train_agcrn_variant() {
  local dataset="$1"
  local variant="$2"
  export BASICTS_DATA_NAME="${dataset}"
  export BASICTS_NUM_EPOCHS="${EPOCHS}"
  export BASICTS_RUN_STAGE="${RUN_STAGE}"
  export BASICTS_AGCRN_VARIANT="${variant}"
  export WANDB_RUN_GROUP="${dataset}_${RUN_STAGE}_agcrn"
  export WANDB_NAME="AGCRN_${dataset}_${RUN_STAGE}_${variant}"
  "${PYTHON_BIN}" experiments/train.py -c "baselines/AGCRN/AdaptiveImportance.py" -g "${GPU}"
}

train_mtgnn_variant() {
  local dataset="$1"
  local variant="$2"
  export BASICTS_DATA_NAME="${dataset}"
  export BASICTS_NUM_EPOCHS="${EPOCHS}"
  export BASICTS_RUN_STAGE="${RUN_STAGE}"
  export BASICTS_MTGNN_VARIANT="${variant}"
  export WANDB_RUN_GROUP="${dataset}_${RUN_STAGE}_mtgnn"
  export WANDB_NAME="MTGNN_${dataset}_${RUN_STAGE}_${variant}"
  "${PYTHON_BIN}" experiments/train.py -c "baselines/MTGNN/AdaptiveImportance.py" -g "${GPU}"
}

for dataset in ${DATASETS}; do
  for model in ${MODELS}; do
    case "${model}" in
      AGCRN)
        for variant in ${AGCRN_VARIANTS}; do
          echo "==> Training AGCRN ${variant} on ${dataset}"
          train_agcrn_variant "${dataset}" "${variant}"
        done
        ;;
      MTGNN)
        for variant in ${MTGNN_VARIANTS}; do
          echo "==> Training MTGNN ${variant} on ${dataset}"
          train_mtgnn_variant "${dataset}" "${variant}"
        done
        ;;
      *)
        echo "Unknown model ${model}; expected AGCRN or MTGNN" >&2
        exit 1
        ;;
    esac
  done
done
