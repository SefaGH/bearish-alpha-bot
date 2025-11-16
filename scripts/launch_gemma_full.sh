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
echo "🧬 GEMMA FULL ACTIVATION - PAPER TRADING"
echo "========================================"
echo "Time: $(date -u +%Y-%m-%d_%H:%M:%S)"
echo ""
echo "📊 Setting ML Environment..."
export ML_ENABLED=true
export GEMMA_ENABLED=true
export ML_RL_TRAINING_MODE=false
export ML_REGIME_MIN_CONFIDENCE=0.60
export ML_RL_HOLD_CONFIDENCE_THRESHOLD=0.60
export GEMMA_MIN_CONFIDENCE=0.66
export PAPER_TRADING=true
export CAPITAL_USDT=100
export TRADING_SYMBOLS="BTC/USDT:USDT"
export LOG_LEVEL=INFO
export DEBUG_MODE=false
export PYTHONIOENCODING="utf-8"

echo "✅ Environment configured"
echo "   ML_ENABLED: $ML_ENABLED"
echo "   GEMMA_ENABLED: $GEMMA_ENABLED"
echo ""

echo "🔍 Verifying GEMMA Manifest..."
if [ -f "artifacts/gemma/final/manifest.json" ]; then
    "${PYTHON_CMD[@]}" - <<'PY'
import json
with open('artifacts/gemma/final/manifest.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)
print(f"[OK] Manifest Version: {manifest.get('version')}")
print(f"[OK] Feature Count: {manifest.get('feature_count')}")
models = manifest.get('models')
if isinstance(models, dict):
    print(f"[OK] Models configured: {len(models)}")
PY
else
    echo "❌ Manifest not found!"
    exit 1
fi
echo ""

echo "🚀 Launching GEMMA Paper Trading (5 minutes)..."
LOG_FILE="logs/gemma_paper_$(date +%Y%m%d_%H%M%S).log"
"${PYTHON_CMD[@]}" -m scripts.live_trading_launcher \
    --paper \
    --duration 300 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "✅ GEMMA Paper Trading Complete"
echo "Log saved to: $LOG_FILE"
echo ""

echo "📊 Quick Analysis:"
echo "   GEMMA references: $(grep -c 'GEMMA' "$LOG_FILE" || true)"
echo "   82-feature confirmations: $(grep -c 'feature_count' "$LOG_FILE" || true)"
echo "   Trades executed: $(grep -c 'TRADE EXECUTED' "$LOG_FILE" || true)"
echo "   Errors: $(grep -c 'ERROR' "$LOG_FILE" || true)"
echo "========================================"