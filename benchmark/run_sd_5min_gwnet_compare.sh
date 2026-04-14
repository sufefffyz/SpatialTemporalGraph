#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:-0}"
shift || true

if [ "$#" -eq 0 ]; then
  WINDOWS=(full 1m)
else
  WINDOWS=("$@")
fi

for WINDOW in "${WINDOWS[@]}"; do
  case "${WINDOW}" in
    full|1m)
      ;;
    *)
      echo "Unsupported window: ${WINDOW}" >&2
      exit 1
      ;;
  esac
done

for WINDOW in "${WINDOWS[@]}"; do
  bash "${SCRIPT_DIR}/run_sd_largest_experiments.sh" "${GPU}" "GWNet_distthre_${WINDOW}"
  bash "${SCRIPT_DIR}/run_sd_phys_experiments.sh" "${GPU}" \
    "GWNet_directed_${WINDOW}" \
    "GWNet_bidir_${WINDOW}" \
    "GWNet_adaptive_only_${WINDOW}" \
    "GWNet_phys_adaptive_${WINDOW}"
done
