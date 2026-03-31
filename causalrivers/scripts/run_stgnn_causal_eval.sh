#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-${1:-}}"
MODEL="${MODEL:-${2:-}}"
STRATEGY="${STRATEGY:-${3:-}}"
N_VARS="${N_VARS:-${4:-3}}"
LABEL_TAG="${LABEL_TAG:-paperlike}"
REGION="${REGION:-east}"
EPOCHS="${EPOCHS:-100}"
INPUT_LEN="${INPUT_LEN:-12}"
OUTPUT_LEN="${OUTPUT_LEN:-12}"
BASICTS_ROOT="${BASICTS_ROOT:-${PROJECT_ROOT}/../BasicTS}"
LEARNED_GRAPH="${LEARNED_GRAPH:-}"
ADJ_MX_PKL="${ADJ_MX_PKL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESTRICT_TO="${RESTRICT_TO:--1}"
SCORE_N_JOBS="${SCORE_N_JOBS:-1}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
KEEP_AUTOREGRESSIVE="${KEEP_AUTOREGRESSIVE:-0}"
ALL_SNAPSHOTS="${ALL_SNAPSHOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_stgnn_causal_eval.sh <DATASET> <MODEL> <STRATEGY> [N_VARS]

Examples:
  bash scripts/run_stgnn_causal_eval.sh TRAFFIC_VOLUME_15MIN AGCRN 2_hop 3
  bash scripts/run_stgnn_causal_eval.sh RIVERS_EAST_GERMANY_6H GWNET random 3

Environment overrides:
  PYTHON_BIN, DATASET, MODEL, STRATEGY, N_VARS, LABEL_TAG, REGION
  EPOCHS, INPUT_LEN, OUTPUT_LEN, BASICTS_ROOT
  LEARNED_GRAPH, ADJ_MX_PKL, OUTPUT_DIR
  RESTRICT_TO, SCORE_N_JOBS, SCORE_CHUNK_SIZE
  KEEP_AUTOREGRESSIVE=1, ALL_SNAPSHOTS=1, DRY_RUN=1
EOF
}

if [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ -z "$STRATEGY" ]; then
  usage >&2
  exit 1
fi

model_dir_for() {
  case "$1" in
    GWNET) echo "GraphWaveNet" ;;
    *) echo "$1" ;;
  esac
}

traffic_label_dataset_name() {
  local dataset="$1"
  case "$dataset" in
    TRAFFIC_VOLUME_5MIN)
      echo "city_traffic_m_volume__category__1_0"
      ;;
    TRAFFIC_VOLUME_*)
      local suffix="${dataset#TRAFFIC_VOLUME_}"
      echo "city_traffic_m_volume__category__1_0_$(printf '%s' "$suffix" | tr '[:upper:]' '[:lower:]')"
      ;;
    *)
      echo "Unsupported traffic dataset name: $dataset" >&2
      return 1
      ;;
  esac
}

resolve_label_path() {
  case "$DATASET" in
    TRAFFIC_VOLUME_*)
      local dataset_name
      dataset_name="$(traffic_label_dataset_name "$DATASET")"
      local label_group="${STRATEGY}_${N_VARS}"
      if [ -n "$LABEL_TAG" ]; then
        label_group="${label_group}_${LABEL_TAG}"
      fi
      echo "${PROJECT_ROOT}/datasets/traffic_${dataset_name}/${label_group}/${dataset_name}.p"
      ;;
    RIVERS_EAST_GERMANY_*)
      echo "${PROJECT_ROOT}/datasets/${STRATEGY}_${N_VARS}/${REGION}.p"
      ;;
    *)
      echo "Unsupported dataset family for automatic label resolution: $DATASET" >&2
      return 1
      ;;
  esac
}

if [ -z "$LEARNED_GRAPH" ]; then
  ckpt_dir="${BASICTS_ROOT}/checkpoints/$(model_dir_for "$MODEL")/${DATASET}_${EPOCHS}_${INPUT_LEN}_${OUTPUT_LEN}"
  if [ "$ALL_SNAPSHOTS" = "1" ]; then
    LEARNED_GRAPH="${ckpt_dir}/learned_graphs"
  else
    LEARNED_GRAPH="${ckpt_dir}/learned_graphs/final.npz"
  fi
fi

if [ -z "$ADJ_MX_PKL" ]; then
  ADJ_MX_PKL="${BASICTS_ROOT}/datasets/${DATASET}/adj_mx.pkl"
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${PROJECT_ROOT}/results/learned_graph_eval/$(printf '%s_%s_%s_%s' "$DATASET" "$MODEL" "$STRATEGY" "$N_VARS" | tr '[:upper:]' '[:lower:]')"
fi

LABEL_PATH="$(resolve_label_path)"

CMD=(
  "$PYTHON_BIN"
  "scripts/evaluate_learned_graphs_against_labels.py"
  "--label-path" "$LABEL_PATH"
  "--learned-graph" "$LEARNED_GRAPH"
  "--adj-mx-pkl" "$ADJ_MX_PKL"
  "--output-dir" "$OUTPUT_DIR"
  "--restrict-to" "$RESTRICT_TO"
  "--n-jobs" "$SCORE_N_JOBS"
  "--chunk-size" "$SCORE_CHUNK_SIZE"
)

if [ "$KEEP_AUTOREGRESSIVE" = "1" ]; then
  CMD+=("--keep-autoregressive")
fi

echo "Dataset        : $DATASET"
echo "Model          : $MODEL"
echo "Strategy       : $STRATEGY"
echo "n_vars         : $N_VARS"
echo "Label path     : $LABEL_PATH"
echo "Learned graph  : $LEARNED_GRAPH"
echo "adj_mx.pkl     : $ADJ_MX_PKL"
echo "Output dir     : $OUTPUT_DIR"
echo "Score n_jobs   : $SCORE_N_JOBS"
echo "Score chunk    : $SCORE_CHUNK_SIZE"
echo
echo "Resolved command:"
printf ' %q' "${CMD[@]}"
echo

if [ ! -e "$LABEL_PATH" ]; then
  echo "Missing label path: $LABEL_PATH" >&2
  exit 1
fi
if [ ! -e "$LEARNED_GRAPH" ]; then
  echo "Missing learned graph path: $LEARNED_GRAPH" >&2
  exit 1
fi
if [ ! -e "$ADJ_MX_PKL" ]; then
  echo "Missing adj_mx.pkl path: $ADJ_MX_PKL" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${CMD[@]}"
