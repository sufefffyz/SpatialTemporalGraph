#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/causalrivers-macos/bin/python}"
BASICTS_PYTHON="${BASICTS_PYTHON:-python}"

echo "[zero-aware] diagnostics python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
for module in ["numpy", "pandas", "scipy", "sklearn"]:
    try:
        mod = __import__(module)
        print(f"{module}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{module}: MISSING ({exc})")
PY

echo
echo "[zero-aware] BasicTS training python: $BASICTS_PYTHON"
"$BASICTS_PYTHON" - <<'PY' || true
for module in ["torch", "easydict", "easytorch"]:
    try:
        mod = __import__(module)
        print(f"{module}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{module}: MISSING ({exc})")
PY

