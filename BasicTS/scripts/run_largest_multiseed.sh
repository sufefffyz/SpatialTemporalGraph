#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_largest_multiseed.sh <STID|BigST> <CA|GBA|GLA|SD> [gpus]" >&2
  exit 1
fi

BASELINE="$1"
DATASET="$2"
GPUS="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_ARRAY <<< "$SEEDS"
else
  NUM_RUNS="${NUM_RUNS:-3}"
  BASE_SEED="${BASE_SEED:-2023}"
  SEED_ARRAY=()
  for ((i=0; i<NUM_RUNS; i++)); do
    SEED_ARRAY+=("$((BASE_SEED + i))")
  done
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$REPO_ROOT/profile_logs/${BASELINE}_${DATASET}_multiseed_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

SUMMARY_ARGS=()

for seed in "${SEED_ARRAY[@]}"; do
  echo "==== Running ${BASELINE} on ${DATASET} with seed ${seed} ===="
  if [[ "$BASELINE" == "BigST" ]]; then
    PROFILE_OUTPUT_DIR="$OUTPUT_DIR/BigSTPreprocess_${DATASET}_s${seed}" \
    BASICTS_SEED="$seed" \
    bash scripts/run_largest_baseline.sh BigSTPreprocess "$DATASET" "$GPUS"

    PROFILE_OUTPUT_DIR="$OUTPUT_DIR/BigST_${DATASET}_s${seed}" \
    BASICTS_SEED="$seed" \
    bash scripts/run_largest_baseline.sh BigST "$DATASET" "$GPUS"

    PRE_SUMMARY="$OUTPUT_DIR/BigSTPreprocess_${DATASET}_s${seed}/profile_summary.json"
    MAIN_SUMMARY="$OUTPUT_DIR/BigST_${DATASET}_s${seed}/profile_summary.json"
    COMBINED_SUMMARY="$OUTPUT_DIR/BigST_${DATASET}_s${seed}_combined.json"

    python3 - <<PY
import json
from pathlib import Path

pre = json.loads(Path("$PRE_SUMMARY").read_text())
main = json.loads(Path("$MAIN_SUMMARY").read_text())

def _num(value):
    return value if isinstance(value, (int, float)) else 0

pre_mem = pre.get("peak_gpu_memory_mb")
main_mem = main.get("peak_gpu_memory_mb")
peak = None
valid = [v for v in [pre_mem, main_mem] if isinstance(v, (int, float))]
if valid:
    peak = max(valid)

payload = {
    "baseline": "BigST",
    "dataset": "$DATASET",
    "seed": $seed,
    "duration_seconds": _num(pre.get("duration_seconds")) + _num(main.get("duration_seconds")),
    "peak_gpu_memory_mb": peak,
    "exit_code": main.get("exit_code"),
    "test_metrics_path": main.get("test_metrics_path"),
    "main_summary_path": "$MAIN_SUMMARY",
    "preprocess_summary_path": "$PRE_SUMMARY",
}
Path("$COMBINED_SUMMARY").write_text(json.dumps(payload, indent=2))
print(f"Saved combined summary to {'$COMBINED_SUMMARY'}")
PY

    SUMMARY_ARGS+=(--summary "$COMBINED_SUMMARY")
  else
    PROFILE_OUTPUT_DIR="$OUTPUT_DIR/${BASELINE}_${DATASET}_s${seed}" \
    BASICTS_SEED="$seed" \
    bash scripts/run_largest_baseline.sh "$BASELINE" "$DATASET" "$GPUS"

    SUMMARY_ARGS+=(--summary "$OUTPUT_DIR/${BASELINE}_${DATASET}_s${seed}/profile_summary.json")
  fi
done

python3 scripts/summarize_largest_multiseed.py \
  "${SUMMARY_ARGS[@]}" \
  --output "$OUTPUT_DIR/multiseed_summary.json"

echo "All done. Output directory: $OUTPUT_DIR"
