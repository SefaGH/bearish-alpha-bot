#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DURATION_SECONDS=${DURATION_SECONDS:-300}
SYMBOLS=${SYMBOLS:-"BTC/USDT:USDT"}
LOG_FILE="gemma_paper_$(date -u +"%Y%m%d_%H%M%S").log"

export ML_ENABLED=true
export GEMMA_ENABLED=true
export ML_RL_TRAINING_MODE=false
export PAPER_TRADING=true
export CAPITAL_USDT=${CAPITAL_USDT:-500}
export TRADING_SYMBOLS="${SYMBOLS}"
export LOG_LEVEL=${LOG_LEVEL:-INFO}

printf '========================================\n'
printf 'RUNNING GEMMA PAPER TRADING\n'
printf '========================================\n'
printf 'Environment:\n'
printf '  ML_ENABLED=%s\n' "${ML_ENABLED}"
printf '  GEMMA_ENABLED=%s\n' "${GEMMA_ENABLED}"
printf '  SYMBOLS=%s\n' "${TRADING_SYMBOLS}"
printf '  DURATION=%s\n' "${DURATION_SECONDS}"

if [[ ! -f "${ROOT_DIR}/artifacts/gemma/final/gemma_price.pt" ]]; then
  printf 'Artifacts missing. Running setup_gemma_artifacts.sh...\n'
  bash "${ROOT_DIR}/scripts/setup_gemma_artifacts.sh"
fi

if [[ ! -L "${ROOT_DIR}/data/models/price/5m/model.pt" && ! -f "${ROOT_DIR}/data/models/price/5m/model.pt" ]]; then
  printf 'Model links missing. Running setup_ml_model_links.sh...\n'
  bash "${ROOT_DIR}/scripts/setup_ml_model_links.sh"
fi

pushd "${ROOT_DIR}" >/dev/null
python -m scripts.live_trading_launcher \
  --paper \
  --duration "${DURATION_SECONDS}" \
  --symbols "${SYMBOLS}" 2>&1 | tee "${LOG_FILE}"
popd >/dev/null

printf '\nRun complete. Log stored at %s\n' "${LOG_FILE}"
