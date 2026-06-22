#!/usr/bin/env bash
set -euo pipefail

# Run LargeST-framework baselines on the three IGSTGNN incident datasets.
#
# Default mode is a low-cost smoke run. Use RUN_MODE=full for full training.
# The IGSTGNN split files are consumed as-is: incident_train/val/test.npy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IGSTGNN_DATA_ROOT="${IGSTGNN_DATA_ROOT:-/home/yuzhang_fei/data/event-traffic-prediction/IGSTGNN/data}"
GPU="${GPU:-2}"
SEED="${SEED:-2025}"
RUN_MODE="${RUN_MODE:-smoke}"
DATASETS="${DATASETS:-Alameda Contra_Costa Orange}"
MODELS="${MODELS:-dstagnn d2stgnn bist}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/reproduction/logs/largest_igstgnn}"

mkdir -p "$ROOT_DIR/data" "$LOG_ROOT"

case "$RUN_MODE" in
  smoke)
    DEFAULT_MAX_EPOCHS=1
    DEFAULT_PATIENCE=1
    export LARGEST_IGSTGNN_SAMPLE_LIMIT="${SMOKE_SAMPLE_LIMIT:-64}"
    ;;
  full)
    DEFAULT_MAX_EPOCHS=100
    DEFAULT_PATIENCE=20
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE. Use smoke or full." >&2
    exit 2
    ;;
esac

ensure_dataset_link() {
  local dataset="$1"
  local src="$IGSTGNN_DATA_ROOT/$dataset"
  local dst="$ROOT_DIR/data/$dataset"
  if [[ ! -d "$src" ]]; then
    echo "Missing IGSTGNN dataset directory: $src" >&2
    exit 1
  fi
  ln -sfn "$src" "$dst"
}

dataset_batch_size() {
  local _model="$1"
  local dataset="$2"
  if [[ "$RUN_MODE" == "smoke" ]]; then
    echo "${SMOKE_BS:-4}"
    return
  fi

  if [[ -n "${IGSTGNN_BATCH_SIZE:-}" ]]; then
    echo "$IGSTGNN_BATCH_SIZE"
    return
  fi

  case "$dataset" in
    Alameda|Contra_Costa) echo 48 ;;
    Orange) echo 24 ;;
    *) echo 48 ;;
  esac
}

model_extra_args() {
  local model="$1"
  local dataset="$2"
  local bs="$3"
  case "$model" in
    dstagnn)
      echo "--input_dim 1 --max_epochs ${DSTAGNN_MAX_EPOCHS:-$DEFAULT_MAX_EPOCHS} --patience ${DSTAGNN_PATIENCE:-$DEFAULT_PATIENCE} --bs $bs"
      ;;
    d2stgnn)
      local max_epochs="${D2STGNN_MAX_EPOCHS:-$DEFAULT_MAX_EPOCHS}"
      local patience="${D2STGNN_PATIENCE:-$DEFAULT_PATIENCE}"
      echo "--input_dim 3 --num_feat 1 --tpd 288 --max_epochs $max_epochs --patience $patience --bs $bs"
      ;;
    bist)
      local max_epochs="${BIST_MAX_EPOCHS:-$DEFAULT_MAX_EPOCHS}"
      local patience="${BIST_PATIENCE:-$DEFAULT_PATIENCE}"
      local core=8
      case "$dataset" in
        Alameda|Contra_Costa) core=8 ;;
        Orange) core=16 ;;
      esac
      echo "--input_dim 3 --kernel_size 3 --core ${BIST_CORE:-$core} --max_epochs $max_epochs --patience $patience --bs $bs"
      ;;
    *)
      echo "Unsupported model: $model" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local model="$1"
  local dataset="$2"
  ensure_dataset_link "$dataset"

  local bs
  bs="$(dataset_batch_size "$model" "$dataset")"
  local extra
  extra="$(model_extra_args "$model" "$dataset" "$bs")"

  local log_file="$LOG_ROOT/${model}_${dataset}_s${SEED}_${RUN_MODE}.log"
  echo "[$(date '+%F %T')] Running $model on $dataset, mode=$RUN_MODE, seed=$SEED, gpu=$GPU, bs=$bs"
  # shellcheck disable=SC2086
  python "experiments/${model}/main.py" \
    --device "cuda:${GPU}" \
    --dataset "$dataset" \
    --years igstgnn \
    --model_name "${model}_igstgnn_${RUN_MODE}" \
    --seed "$SEED" \
    --seq_len 12 \
    --horizon 12 \
    $extra 2>&1 | tee "$log_file"
}

for dataset in $DATASETS; do
  ensure_dataset_link "$dataset"
done

for model in $MODELS; do
  for dataset in $DATASETS; do
    run_one "$model" "$dataset"
  done
done
