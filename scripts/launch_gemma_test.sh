#!/usr/bin/env bash
set -euo pipefail

printf '========================================\n'
printf '🚀 GEMMA PAPER-TRADING SMOKE TEST\n'
printf '========================================\n'

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
printf 'Using Python interpreter: %s\n' "${PYTHON_CMD[*]}"

if [ -f ".env.gemma" ]; then
  printf 'Sourcing .env.gemma for overrides...\n'
  set -a
  # shellcheck disable=SC1091
  source ./.env.gemma
  set +a
else
  printf '[WARN] .env.gemma not found. Proceeding with existing environment.\n'
fi

DURATION_SECONDS=${GEMMA_TEST_DURATION:-300}
printf 'Launching paper trading run (duration=%ss)...\n' "$DURATION_SECONDS"

"${PYTHON_CMD[@]}" scripts/live_trading_launcher.py \
  --paper \
  --debug \
  --duration "$DURATION_SECONDS"
