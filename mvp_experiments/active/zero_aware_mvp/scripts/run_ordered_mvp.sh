#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${ACTIVE_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

bash "${SCRIPT_DIR}/check_env.sh"
bash "${SCRIPT_DIR}/prepare_mvp_datasets.sh"
bash "${SCRIPT_DIR}/run_00_diagnostics.sh"
bash "${SCRIPT_DIR}/run_01_naive_baselines.sh"

if [[ "${RUN_BASICTS:-0}" == "1" ]]; then
  ZA_MODEL=AGCRN ZA_NUM_EPOCHS="${ZA_NUM_EPOCHS:-3}" bash "${SCRIPT_DIR}/run_02_basicts_smoke.sh"
  ZA_MODEL=GWNET ZA_NUM_EPOCHS="${ZA_NUM_EPOCHS:-3}" bash "${SCRIPT_DIR}/run_02_basicts_smoke.sh"
else
  echo "[zero-aware] Skipping BasicTS training. Set RUN_BASICTS=1 in an env with torch/easytorch."
fi

bash "${SCRIPT_DIR}/run_04_collect_results.sh"
