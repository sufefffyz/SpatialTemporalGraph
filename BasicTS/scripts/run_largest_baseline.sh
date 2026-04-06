#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_largest_baseline.sh <STID|BigSTPreprocess|BigST> <CA|GBA|GLA|SD> [gpus]" >&2
  exit 1
fi

BASELINE="$1"
DATASET="$2"
GPUS="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$BASELINE" in
  STID)
    CFG_PATH="baselines/STID/${DATASET}.py"
    ;;
  BigSTPreprocess)
    CFG_PATH="baselines/BigST/Preprocess${DATASET}.py"
    ;;
  BigST)
    CFG_PATH="baselines/BigST/${DATASET}.py"
    ;;
  *)
    echo "Unsupported baseline: ${BASELINE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Missing config: $CFG_PATH" >&2
  exit 1
fi

bash scripts/run_profiled_training.sh "$CFG_PATH" "$GPUS" "${BASELINE}_${DATASET}"
