#!/usr/bin/env bash
set -euo pipefail

_validate_python() {
  local cmd=("$@")
  "${cmd[@]}" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

detect_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    read -r -a candidate <<<"${PYTHON_BIN}"
    if _validate_python "${candidate[@]}"; then
      PYTHON_CMD=("${candidate[@]}")
      return
    fi
    echo "Provided PYTHON_BIN is not Python 3.11: ${PYTHON_BIN}" >&2
    exit 1
  fi

  local candidates=(
    "C:/Users/sefaa/AppData/Local/Programs/Python/Python311/python.exe"
    "/mnt/c/Users/sefaa/AppData/Local/Programs/Python/Python311/python.exe"
    "python"
    "python3"
    "py -3"
  )

  for entry in "${candidates[@]}"; do
    read -r -a cmd <<<"${entry}"
    if { command -v "${cmd[0]}" >/dev/null 2>&1 || [ -x "${cmd[0]}" ]; } \
       && _validate_python "${cmd[@]}"; then
      PYTHON_CMD=("${cmd[@]}")
      return
    fi
  done

  echo "Python 3.11 interpreter not found. Set PYTHON_BIN." >&2
  exit 1
}

detect_python

echo "========================================"
echo "🚀 GEMMA FINAL TEST - COMPLETE SETUP"
echo "========================================"

echo "Step 1: Setting up timeframe models..."
./scripts/create_timeframe_models.sh
echo ""

echo "Step 2: Fixing regime LSTM..."
"${PYTHON_CMD[@]}" scripts/fix_regime_lstm.py
echo ""

echo "Step 3: Verification..."
if [ -f "artifacts/gemma/final/manifest.json" ]; then
  echo "✅ Manifest"
else
  echo "❌ Manifest missing"
fi
if [ -f "data/models/lstm/5m/lstm_5m.pth" ]; then
  echo "✅ LSTM 5m"
else
  echo "❌ LSTM 5m missing"
fi
if [ -f "data/models/regime_lstm/best_model.pth" ]; then
  echo "✅ Regime LSTM"
else
  echo "❌ Regime LSTM missing"
fi
echo ""

echo "Step 4: Running GEMMA Paper Trading..."
./scripts/launch_gemma_full.sh

echo ""
echo "========================================"
echo "✅ GEMMA FINAL TEST COMPLETE"
echo "========================================"