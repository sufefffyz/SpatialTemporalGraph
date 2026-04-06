#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_largest_multiseed.sh <STID|BigST> <CA|GBA|GLA|SD> [gpus] [num_runs]" >&2
  exit 1
fi

BASELINE="$1"
DATASET="$2"
GPUS="${3:-0}"
NUM_RUNS="${4:-${NUM_RUNS:-3}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -le 0 ]]; then
  echo "num_runs must be a positive integer, got: $NUM_RUNS" >&2
  exit 1
fi

SHARED_SEED_DIR="${SHARED_SEED_DIR:-$REPO_ROOT/profile_logs/shared_seeds}"
DEFAULT_SHARED_SEED_FILE="$SHARED_SEED_DIR/largest_shared_seeds_${NUM_RUNS}.txt"

load_seed_file() {
  local seed_file="$1"
  local seed_lines=()
  local seed_tokens=()
  mapfile -t seed_lines < "$seed_file"
  for line in "${seed_lines[@]}"; do
    line="${line%%#*}"
    if [[ -z "${line//[[:space:]]/}" ]]; then
      continue
    fi
    read -r -a tokens <<< "$line"
    seed_tokens+=("${tokens[@]}")
  done
  printf '%s\n' "${seed_tokens[@]}"
}

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_ARRAY <<< "$SEEDS"
  SEED_SOURCE="env:SEEDS"
elif [[ -n "${SEED_FILE:-}" ]]; then
  if [[ ! -f "$SEED_FILE" ]]; then
    echo "Missing seed file: $SEED_FILE" >&2
    exit 1
  fi
  mapfile -t SEED_ARRAY < <(load_seed_file "$SEED_FILE")
  SEED_SOURCE="$(cd "$(dirname "$SEED_FILE")" && pwd)/$(basename "$SEED_FILE")"
elif [[ -f "$DEFAULT_SHARED_SEED_FILE" ]]; then
  mapfile -t SEED_ARRAY < <(load_seed_file "$DEFAULT_SHARED_SEED_FILE")
  SEED_SOURCE="$DEFAULT_SHARED_SEED_FILE"
else
  GENERATOR_SEED_ARGS=()
  RANGE_ARGS=()
  if [[ -n "${GENERATOR_SEED:-}" ]]; then
    GENERATOR_SEED_ARGS+=(--generator-seed "$GENERATOR_SEED")
  fi
  if [[ -n "${SEED_LOW:-}" ]]; then
    RANGE_ARGS+=(--low "$SEED_LOW")
  fi
  if [[ -n "${SEED_HIGH:-}" ]]; then
    RANGE_ARGS+=(--high "$SEED_HIGH")
  fi

  mkdir -p "$SHARED_SEED_DIR"
  python3 scripts/generate_shared_seeds.py \
    --num-seeds "$NUM_RUNS" \
    --output "$DEFAULT_SHARED_SEED_FILE" \
    "${GENERATOR_SEED_ARGS[@]}" \
    "${RANGE_ARGS[@]}"
  mapfile -t SEED_ARRAY < <(load_seed_file "$DEFAULT_SHARED_SEED_FILE")
  SEED_SOURCE="auto-generated:${DEFAULT_SHARED_SEED_FILE}"
fi

if [[ ${#SEED_ARRAY[@]} -ne "$NUM_RUNS" ]]; then
  echo "Resolved ${#SEED_ARRAY[@]} seeds, but num_runs=$NUM_RUNS." >&2
  echo "Seed source: ${SEED_SOURCE}" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$REPO_ROOT/profile_logs/${BASELINE}_${DATASET}_multiseed_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

SUMMARY_ARGS=()

printf '%s\n' "${SEED_ARRAY[@]}" > "$OUTPUT_DIR/used_seeds.txt"
echo "Using seeds from ${SEED_SOURCE}"
echo "Requested num_runs: ${NUM_RUNS}"
echo "Seed list: ${SEED_ARRAY[*]}"
echo "Saved seed list to $OUTPUT_DIR/used_seeds.txt"

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
