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

CONFIG_NAME="${CONFIG_NAME:-}"
LABEL_PATH="${LABEL_PATH:-}"
DATA_PATH="${DATA_PATH:-}"
LEARNED_GRAPH="${LEARNED_GRAPH:-}"
ADJ_MX_PKL="${ADJ_MX_PKL:-}"
SAVE_PATH="${SAVE_PATH:-results/}"

LOAD_MODE="${LOAD_MODE:-eager}"
DT_PREPROCESS="${DT_PREPROCESS:-False}"
RESTRICT_TO="${RESTRICT_TO:--1}"
REMOVE_DIAGONAL="${REMOVE_DIAGONAL:-True}"
N_JOBS="${N_JOBS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
SCORE_N_JOBS="${SCORE_N_JOBS:-1}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
SAVE_FULL_OUT="${SAVE_FULL_OUT:-True}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_stgnn_causal_eval.sh <DATASET> <MODEL> <STRATEGY> [N_VARS]

Examples:
  bash scripts/run_stgnn_causal_eval.sh RIVERS_EAST_GERMANY_6H D2STGNN random 3
  bash scripts/run_stgnn_causal_eval.sh TRAFFIC_VOLUME_15MIN AGCRN 2_hop 3

This wrapper runs benchmark.py with the stgnn_precomputed baseline so the
learned graph is scored through the same pipeline as VAR/CC/etc.

Environment overrides:
  PYTHON_BIN, DATASET, MODEL, STRATEGY, N_VARS
  LABEL_TAG, REGION, EPOCHS, INPUT_LEN, OUTPUT_LEN, BASICTS_ROOT
  CONFIG_NAME, LABEL_PATH, DATA_PATH, LEARNED_GRAPH, ADJ_MX_PKL, SAVE_PATH
  LOAD_MODE, DT_PREPROCESS, RESTRICT_TO, REMOVE_DIAGONAL
  N_JOBS, CHUNK_SIZE, SCORE_N_JOBS, SCORE_CHUNK_SIZE, SAVE_FULL_OUT
  DRY_RUN=1
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

resolve_checkpoint_artifact() {
  local ckpt_dir="$1"
  local direct_path="${ckpt_dir}/learned_graphs/final.npz"

  if [ -e "$direct_path" ]; then
    printf '%s\n' "$direct_path"
    return 0
  fi

  local pattern="${ckpt_dir}"/*/learned_graphs/final.npz
  local matches=()
  local candidate
  for candidate in $pattern; do
    if [ -e "$candidate" ]; then
      matches+=("$candidate")
    fi
  done

  if [ "${#matches[@]}" -eq 0 ]; then
    return 1
  fi

  if [ "${#matches[@]}" -gt 1 ]; then
    printf 'Multiple checkpoint artifacts found under %s; using the most recent one.\n' "$ckpt_dir" >&2
  fi

  ls -td "${matches[@]}" | head -n 1
}

traffic_label_dataset_name() {
  local dataset="$1"
  local prefix
  local suffix

  case "$dataset" in
    TRAFFIC_VOLUME_*)
      prefix="city_traffic_m_volume__category__1_0"
      suffix="${dataset#TRAFFIC_VOLUME_}"
      ;;
    TRAFFIC_SPEED_*)
      prefix="city_traffic_m_speed__category__1_0"
      suffix="${dataset#TRAFFIC_SPEED_}"
      ;;
    *)
      echo "Unsupported traffic dataset name: $dataset" >&2
      return 1
      ;;
  esac

  if [ "$suffix" = "5MIN" ]; then
    echo "$prefix"
  else
    echo "${prefix}_$(printf '%s' "$suffix" | tr '[:upper:]' '[:lower:]')"
  fi
}

dataset_resolution_override() {
  case "$DATASET" in
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
      printf '%s\n' "$(printf '%s' "${DATASET##*_}" | tr '[:upper:]' '[:lower:]')"
      ;;
    RIVERS_EAST_GERMANY_*)
      printf '%s\n' "$(printf '%s' "${DATASET##*_}" | tr '[:upper:]' '[:lower:]')"
      ;;
    *)
      return 0
      ;;
  esac
}

resolve_label_path() {
  case "$DATASET" in
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
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

resolve_data_path() {
  case "$DATASET" in
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
      local dataset_name
      dataset_name="$(traffic_label_dataset_name "$DATASET")"
      echo "${PROJECT_ROOT}/product/traffic_${dataset_name}"
      ;;
    RIVERS_EAST_GERMANY_*)
      case "$REGION" in
        east) echo "${PROJECT_ROOT}/product/rivers_ts_east_germany.csv" ;;
        bav) echo "${PROJECT_ROOT}/product/rivers_ts_bavaria.csv" ;;
        flood) echo "${PROJECT_ROOT}/product/rivers_ts_flood.csv" ;;
        *)
          echo "Unsupported rivers region: $REGION" >&2
          return 1
          ;;
      esac
      ;;
    *)
      echo "Unsupported dataset family for automatic data path resolution: $DATASET" >&2
      return 1
      ;;
  esac
}

resolve_config_name() {
  if [ -n "$CONFIG_NAME" ]; then
    echo "$CONFIG_NAME"
    return 0
  fi

  case "$DATASET" in
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
      echo "benchmark_traffic_paper_baselines"
      ;;
    RIVERS_EAST_GERMANY_*)
      echo "benchmark"
      ;;
    *)
      echo "Unsupported dataset family for automatic config resolution: $DATASET" >&2
      return 1
      ;;
  esac
}

methods_override() {
  local learned_graph="$1"
  local adj_mx="$2"
  local config_name="$3"
  local key="methods"

  case "$config_name" in
    benchmark)
      key="+methods"
      ;;
    *)
      key="methods"
      ;;
  esac

  printf '%s=[{name:stgnn_precomputed,learned_graph_path:%s,adj_mx_pkl:%s}]' "$key" "$learned_graph" "$adj_mx"
}

if [ -z "$LEARNED_GRAPH" ]; then
  ckpt_dir="${BASICTS_ROOT}/checkpoints/$(model_dir_for "$MODEL")/${DATASET}_${EPOCHS}_${INPUT_LEN}_${OUTPUT_LEN}"
  if ! LEARNED_GRAPH="$(resolve_checkpoint_artifact "$ckpt_dir")"; then
    echo "Unable to resolve learned graph automatically under: $ckpt_dir" >&2
    echo "Set LEARNED_GRAPH manually to a learned_graphs/final.npz path." >&2
    exit 1
  fi
fi

if [ -z "$ADJ_MX_PKL" ]; then
  ADJ_MX_PKL="${BASICTS_ROOT}/datasets/${DATASET}/adj_mx.pkl"
fi

if [ -z "$LABEL_PATH" ]; then
  LABEL_PATH="$(resolve_label_path)"
fi

if [ -z "$DATA_PATH" ]; then
  DATA_PATH="$(resolve_data_path)"
fi

CONFIG_NAME="$(resolve_config_name)"
METHOD_OVERRIDE="$(methods_override "$LEARNED_GRAPH" "$ADJ_MX_PKL" "$CONFIG_NAME")"
RESOLUTION_OVERRIDE="$(dataset_resolution_override || true)"

CMD=(
  "$PYTHON_BIN"
  "benchmark.py"
  "--config-name" "$CONFIG_NAME"
  "label_path=${LABEL_PATH}"
  "data_path=${DATA_PATH}"
  "dt_preprocess=${DT_PREPROCESS}"
  "save_path=${SAVE_PATH}"
  "save_full_out=${SAVE_FULL_OUT}"
  "remove_diagonal=${REMOVE_DIAGONAL}"
  "restrict_to=${RESTRICT_TO}"
  "load_mode=${LOAD_MODE}"
  "n_jobs=${N_JOBS}"
  "chunk_size=${CHUNK_SIZE}"
  "score_n_jobs=${SCORE_N_JOBS}"
  "score_chunk_size=${SCORE_CHUNK_SIZE}"
  "$METHOD_OVERRIDE"
)

if [ -n "$RESOLUTION_OVERRIDE" ]; then
  CMD+=("data_preprocess.resolution=${RESOLUTION_OVERRIDE}")
fi

echo "Dataset        : $DATASET"
echo "Model          : $MODEL"
echo "Strategy       : $STRATEGY"
echo "n_vars         : $N_VARS"
echo "Config         : $CONFIG_NAME"
echo "Label path     : $LABEL_PATH"
echo "Data path      : $DATA_PATH"
echo "Learned graph  : $LEARNED_GRAPH"
echo "adj_mx.pkl     : $ADJ_MX_PKL"
echo "Save path      : $SAVE_PATH"
echo "Load mode      : $LOAD_MODE"
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
if [ ! -e "$DATA_PATH" ]; then
  echo "Missing data path: $DATA_PATH" >&2
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
