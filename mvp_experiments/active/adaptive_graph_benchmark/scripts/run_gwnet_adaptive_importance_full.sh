#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_ROOT="${REPO_ROOT}/BasicTS"
CONFIG="baselines/GWNet/AdaptiveImportance.py"
PYTHON_BIN="${PYTHON:-python3}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-METR-LA PEMS04 PEMS07 SD}"
TRAIN_VARIANTS="${TRAIN_VARIANTS:-original no_adaptive_from_scratch frozen_random_adaptive_from_scratch learned_no_relu signal_mlp_relu signal_mlp_no_relu}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_POSTHOC="${RUN_POSTHOC:-1}"
EPOCHS="${EPOCHS:-100}"
RUN_STAGE="${RUN_STAGE:-full}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"

export WANDB_PROJECT="${WANDB_PROJECT:-adaptive_graph_importance_ablation}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export BASICTS_DETERMINISTIC="${BASICTS_DETERMINISTIC:-0}"
export BASICTS_CUDNN_DETERMINISTIC="${BASICTS_CUDNN_DETERMINISTIC:-0}"
export BASICTS_CUDNN_BENCHMARK="${BASICTS_CUDNN_BENCHMARK:-1}"
export BASICTS_ENV_TAG="${BASICTS_ENV_TAG:-det0_cudnndet0}"
export BASICTS_EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}"

cd "${BASICTS_ROOT}"

ratio_tag() {
  local ratio="$1"
  "${PYTHON_BIN}" - "$ratio" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}".replace(".", "p"))
PY
}

best_original_ckpt() {
  local dataset="$1"
  local root="checkpoints/GraphWaveNetAdaptiveImportance/${dataset}_${RUN_STAGE}_original_eval-learned_keep-1p000_${BASICTS_ENV_TAG}_${EPOCHS}_12_12"
  find "${root}" -type f -name "GraphWaveNetAdaptiveImportance_best_val_MAE.pt" -print 2>/dev/null | sort | tail -n 1
}

train_variant() {
  local dataset="$1"
  local variant="$2"
  export BASICTS_DATA_NAME="${dataset}"
  export BASICTS_NUM_EPOCHS="${EPOCHS}"
  export BASICTS_RUN_STAGE="${RUN_STAGE}"
  export BASICTS_GWNET_VARIANT="${variant}"
  export BASICTS_ADAPTIVE_EVAL_MODE="learned"
  export BASICTS_ADAPTIVE_KEEP_RATIO="1.0"
  export BASICTS_ADAPTIVE_KEEP_TAG="1p000"
  export WANDB_RUN_GROUP="${dataset}_${RUN_STAGE}"
  export WANDB_NAME="GWNet_${dataset}_${RUN_STAGE}_${variant}"
  "${PYTHON_BIN}" experiments/train.py -c "${CONFIG}" -g "${GPU}"
}

eval_original() {
  local dataset="$1"
  local mode="$2"
  local ratio="$3"
  local tag
  tag="$(ratio_tag "${ratio}")"
  local ckpt
  ckpt="$(best_original_ckpt "${dataset}")"
  if [[ -z "${ckpt}" ]]; then
    echo "Missing original checkpoint for ${dataset}" >&2
    exit 1
  fi

  export BASICTS_DATA_NAME="${dataset}"
  export BASICTS_NUM_EPOCHS="${EPOCHS}"
  export BASICTS_RUN_STAGE="${RUN_STAGE}"
  export BASICTS_GWNET_VARIANT="original"
  export BASICTS_ADAPTIVE_EVAL_MODE="${mode}"
  export BASICTS_ADAPTIVE_KEEP_RATIO="${ratio}"
  export BASICTS_ADAPTIVE_KEEP_TAG="${tag}"
  export WANDB_RUN_GROUP="${dataset}_${RUN_STAGE}_posthoc"
  export WANDB_NAME="GWNet_${dataset}_${RUN_STAGE}_posthoc_${mode}_keep${tag}"
  "${PYTHON_BIN}" experiments/evaluate_metrics_only.py -cfg "${CONFIG}" -ckpt "${ckpt}" -g "${GPU}"
}

for dataset in ${DATASETS}; do
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    for variant in ${TRAIN_VARIANTS}; do
      echo "==> Training ${variant} on ${dataset}"
      train_variant "${dataset}" "${variant}"
    done
  fi

  if [[ "${RUN_POSTHOC}" == "1" ]]; then
    echo "==> Post-hoc eval adaptive keep 100% on ${dataset}"
    eval_original "${dataset}" "learned" "1.0"

    for ratio in 0.7 0.5 0.3 0.1; do
      echo "==> Post-hoc eval adaptive top ${ratio} on ${dataset}"
      eval_original "${dataset}" "topk" "${ratio}"
    done

    echo "==> Post-hoc eval shuffle on ${dataset}"
    eval_original "${dataset}" "shuffle" "1.0"

    echo "==> Post-hoc eval disable adaptive on ${dataset}"
    eval_original "${dataset}" "off" "1.0"
  fi
done
