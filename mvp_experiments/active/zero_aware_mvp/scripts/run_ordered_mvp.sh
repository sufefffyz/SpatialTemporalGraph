#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash zero_aware_mvp/scripts/check_env.sh
bash zero_aware_mvp/scripts/prepare_mvp_datasets.sh
bash zero_aware_mvp/scripts/run_00_diagnostics.sh
bash zero_aware_mvp/scripts/run_01_naive_baselines.sh

if [[ "${RUN_BASICTS:-0}" == "1" ]]; then
  ZA_MODEL=AGCRN ZA_NUM_EPOCHS="${ZA_NUM_EPOCHS:-3}" bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh
  ZA_MODEL=GWNET ZA_NUM_EPOCHS="${ZA_NUM_EPOCHS:-3}" bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh
else
  echo "[zero-aware] Skipping BasicTS training. Set RUN_BASICTS=1 in an env with torch/easytorch."
fi

bash zero_aware_mvp/scripts/run_04_collect_results.sh

