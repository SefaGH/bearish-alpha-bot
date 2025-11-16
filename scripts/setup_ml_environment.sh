#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.gemma_live"
CONFIG_FILE="${ROOT_DIR}/config/config.example.yaml"

log() {
  local message="$1"
  local level="${2:-INFO}"
  printf '%s [%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "${message}"
}

log "========================================"
log "SETTING UP ML ENVIRONMENT"
log "========================================"

cat > "${ENV_FILE}" <<'ENV'
# ML Configuration
declare -x ML_ENABLED=true
declare -x GEMMA_ENABLED=true
declare -x ML_RL_TRAINING_MODE=false

# Model paths
declare -x GEMMA_MODEL_PATH="artifacts/gemma/final"
declare -x GEMMA_MANIFEST_PATH="artifacts/gemma/final/manifest.json"

# Trading configuration
declare -x PAPER_TRADING=true
declare -x CAPITAL_USDT=100
declare -x TRADING_SYMBOLS="BTC/USDT:USDT"

# ML Thresholds
declare -x ML_REGIME_MIN_CONFIDENCE=0.60
declare -x ML_RL_HOLD_CONFIDENCE_THRESHOLD=0.60
declare -x GEMMA_MIN_CONFIDENCE=0.66

# Logging
declare -x LOG_LEVEL=INFO
declare -x DEBUG_MODE=false
ENV

log "Environment file written to ${ENV_FILE}"
log "Run 'source .env.gemma_live' before launching."

PY_CMD="${PYTHON_BIN:-python}"
if ! command -v "${PY_CMD}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY_CMD=python3
  fi
fi

if [[ -f "${CONFIG_FILE}" ]]; then
  if command -v "${PY_CMD}" >/dev/null 2>&1; then
    log "Verifying config/config.example.yaml flags via PyYAML..."
    "${PY_CMD}" - "${CONFIG_FILE}" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not available; skipping config verification")
    sys.exit(0)

config_path = Path(sys.argv[1])
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
issues = []
if data.get("ml", {}).get("enabled") is not True:
    issues.append("ml.enabled is not true; please update config")
gemma_cfg = data.get("ml", {}).get("gemma") or data.get("gemma") or {}
if gemma_cfg.get("enabled") is not True:
    issues.append("ml.gemma.enabled is not true; please update config")

if issues:
    for issue in issues:
        print(f"WARN: {issue}")
else:
    print("INFO: Config already enables ML and GEMMA")
PY
  else
    log "Python interpreter not found; skipping config verification" "WARN"
  fi
else
  log "config/config.example.yaml not found" "WARN"
fi
