#!/usr/bin/env bash
set -euo pipefail

printf '========================================\n'
printf '🚀 GEMMA ACTIVATION SCRIPT\n'
printf '========================================\n\n'

declare -a PYTHON_CMD=()

find_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            PYTHON_CMD=("$PYTHON_BIN")
            return
        fi
        if [ -x "$PYTHON_BIN" ]; then
            PYTHON_CMD=("$PYTHON_BIN")
            return
        fi
    fi

    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_CMD=($candidate)
            return
        fi
    done

    if command -v py >/dev/null 2>&1; then
        PYTHON_CMD=(py -3)
        return
    fi

    printf '[ERROR] No suitable Python interpreter found on PATH.\n' >&2
    exit 1
}

find_python
printf 'Using Python interpreter: %s\n\n' "${PYTHON_CMD[*]}"

cat <<'ENVFILE' > .env.gemma
# GEMMA ACTIVATION
GEMMA_ENABLED=true
ML_ENABLED=true

# Paper Trading for initial GEMMA test
ENABLE_LIVE_TRADING=false
PAPER_TRADING=true

# Capital & Risk (balanced preset)
CAPITAL_USDT=500
PER_TRADE_RISK_PCT=0.003
MAX_POSITION_SIZE_PCT=0.20
DAILY_MAX_TRADES=10

# Trading Universe - BTC only
TRADING_SYMBOLS="BTC/USDT:USDT"
TRADING_SYMBOLS_PRIORITY="BTC/USDT:USDT"
UNIVERSE_AUTO_SELECT=false

# ML Configuration with GEMMA
ML_RL_TRAINING_MODE=false
ML_RL_EPSILON_INFERENCE=0.01
ML_REGIME_MIN_CONFIDENCE=0.60
ML_RL_HOLD_CONFIDENCE_THRESHOLD=0.60

# GEMMA Specific Settings
GEMMA_SHADOW_MODE=false  # Full activation, not shadow
GEMMA_MIN_CONFIDENCE=0.66
GEMMA_CIRCUIT_BREAKER_ENABLED=true

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
ENVFILE

printf '✅ GEMMA environment created\n\n'

"${PYTHON_CMD[@]}" - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil

config_path = Path('config/config.example.yaml')
if not config_path.exists():
    raise SystemExit('Configuration file missing at config/config.example.yaml')

backup_dir = config_path.parent
timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
backup_path = backup_dir / f'config.backup.{timestamp}.yaml'
shutil.copy2(config_path, backup_path)
print(f'✅ Config backed up to {backup_path}')
PY

"${PYTHON_CMD[@]}" - <<'PY'
from pathlib import Path

config_path = Path('config/config.example.yaml')
text = config_path.read_text(encoding='utf-8')
lines = text.splitlines()

in_gemma = False
gemma_indent = None
updated = False

for idx, line in enumerate(lines):
    stripped = line.lstrip()
    indent = len(line) - len(stripped)

    if stripped.startswith('gemma:') and indent >= 0:
        in_gemma = True
        gemma_indent = indent
        continue

    if in_gemma:
        if indent <= gemma_indent and stripped and not stripped.startswith('#'):
            break
        if stripped.startswith('enabled:'):
            if 'true' not in stripped:
                lines[idx] = ' ' * indent + 'enabled: true'
            updated = True
            break

if updated:
    config_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('✅ GEMMA enabled in config')
else:
    print('⚠️  GEMMA enable flag not updated; please verify config structure')
PY

printf '\n========================================\n'
printf '✅ GEMMA ACTIVATION COMPLETE!\n'
printf '========================================\n\n'
printf 'Next steps:\n'
printf '1. source .env.gemma\n'
printf '2. ./scripts/launch_gemma_test.sh\n'
