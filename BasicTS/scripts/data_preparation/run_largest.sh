#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
  DATASETS=(CA GBA GLA SD)
else
  DATASETS=("$@")
fi

for dataset in "${DATASETS[@]}"; do
  SCRIPT_PATH="scripts/data_preparation/${dataset}/generate_training_data.py"
  RAW_DIR="datasets/raw_data/${dataset}"

  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Unknown dataset: ${dataset}" >&2
    exit 1
  fi

  if [[ ! -d "$RAW_DIR" ]]; then
    echo "Missing raw data directory: ${RAW_DIR}" >&2
    exit 1
  fi

  echo "==== Preparing ${dataset} from ${RAW_DIR} ===="
  python "$SCRIPT_PATH"
done
