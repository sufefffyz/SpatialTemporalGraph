#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${DATASET:-${1:-}}"
MODEL="${MODEL:-${2:-}}"

LABEL_TAG="${LABEL_TAG:-paperlike}"
REGION="${REGION:-east}"
N_VARS_LIST="${N_VARS_LIST:-3 5}"
STRATEGIES="${STRATEGIES:-}"
LOG_DIR="${LOG_DIR:-logs/stgnn_eval}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_stgnn_causal_eval_all.sh <DATASET> <MODEL>

Examples:
  bash scripts/run_stgnn_causal_eval_all.sh RIVERS_EAST_GERMANY_6H D2STGNN
  bash scripts/run_stgnn_causal_eval_all.sh TRAFFIC_VOLUME_15MIN AGCRN

Environment overrides:
  LABEL_TAG, REGION, N_VARS_LIST, STRATEGIES, LOG_DIR

Defaults:
  Rivers strategies : random 1_random root_cause confounder close
  Traffic strategies: random root_cause confounder close 2_hop
  debug_set is excluded by default.

This script delegates each run to scripts/run_stgnn_causal_eval.sh and prints a
final success / failure / skipped summary.
EOF
}

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  usage >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

default_strategies() {
  case "$DATASET" in
    RIVERS_EAST_GERMANY_*)
      echo "random 1_random root_cause confounder close"
      ;;
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
      echo "random root_cause confounder close 2_hop"
      ;;
    *)
      echo "Unsupported dataset family for automatic strategy resolution: $DATASET" >&2
      return 1
      ;;
  esac
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

resolve_label_path() {
  local strategy="$1"
  local n_vars="$2"
  case "$DATASET" in
    TRAFFIC_VOLUME_*|TRAFFIC_SPEED_*)
      local dataset_name
      dataset_name="$(traffic_label_dataset_name "$DATASET")"
      local label_group="${strategy}_${n_vars}"
      if [ -n "$LABEL_TAG" ]; then
        label_group="${label_group}_${LABEL_TAG}"
      fi
      echo "${PROJECT_ROOT}/datasets/traffic_${dataset_name}/${label_group}/${dataset_name}.p"
      ;;
    RIVERS_EAST_GERMANY_*)
      echo "${PROJECT_ROOT}/datasets/${strategy}_${n_vars}/${REGION}.p"
      ;;
    *)
      echo "Unsupported dataset family for label resolution: $DATASET" >&2
      return 1
      ;;
  esac
}

if [ -z "$STRATEGIES" ]; then
  STRATEGIES="$(default_strategies)" || exit 1
fi

read -r -a STRATEGY_ARRAY <<< "$STRATEGIES"
read -r -a N_VARS_ARRAY <<< "$N_VARS_LIST"

FAILED_RUNS=()
SKIPPED_RUNS=()
COMPLETED_RUNS=()

echo "Dataset      : $DATASET"
echo "Model        : $MODEL"
echo "Region       : $REGION"
echo "Label tag    : $LABEL_TAG"
echo "Strategies   : ${STRATEGY_ARRAY[*]}"
echo "n_vars list  : ${N_VARS_ARRAY[*]}"
echo "Log dir      : $LOG_DIR"
echo

for strategy in "${STRATEGY_ARRAY[@]}"; do
  for n_vars in "${N_VARS_ARRAY[@]}"; do
    label_path="$(resolve_label_path "$strategy" "$n_vars")" || {
      FAILED_RUNS+=("${strategy}:${n_vars}:label-resolution")
      continue
    }

    if [ ! -f "$label_path" ]; then
      echo "Skipping missing label: $label_path"
      SKIPPED_RUNS+=("${strategy}:${n_vars}:missing-label")
      continue
    fi

    log_path="${LOG_DIR}/$(printf '%s_%s_%s_%s' "$DATASET" "$MODEL" "$strategy" "$n_vars" | tr '[:upper:]' '[:lower:]').log"

    echo "Running strategy : $strategy"
    echo "n_vars           : $n_vars"
    echo "Label path       : $label_path"
    echo "Log              : $log_path"

    if LABEL_TAG="$LABEL_TAG" REGION="$REGION" \
      bash scripts/run_stgnn_causal_eval.sh "$DATASET" "$MODEL" "$strategy" "$n_vars" \
      2>&1 | tee "$log_path"; then
      COMPLETED_RUNS+=("${strategy}:${n_vars}")
      echo "Finished: ${strategy}/${n_vars}"
    else
      status=$?
      FAILED_RUNS+=("${strategy}:${n_vars}:exit-${status}")
      echo "Failed: ${strategy}/${n_vars} (exit=${status})" >&2
    fi

    echo
  done
done

if [ "${#COMPLETED_RUNS[@]}" -gt 0 ]; then
  echo "Completed runs:"
  printf '  %s\n' "${COMPLETED_RUNS[@]}"
  echo
fi

if [ "${#SKIPPED_RUNS[@]}" -gt 0 ]; then
  echo "Skipped runs:"
  printf '  %s\n' "${SKIPPED_RUNS[@]}"
  echo
fi

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All STGNN causal evaluation runs completed successfully."
