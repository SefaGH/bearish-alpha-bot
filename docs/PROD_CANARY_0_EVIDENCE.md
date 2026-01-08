# Production Canary-0 Evidence Runbook (BingX, Hard Stop Only)

Goal: produce audit-grade artifacts for **Production Canary-0** (native hard stop only) and verify fail-fast safety.

This mode writes gitignored evidence JSON to `diagnostics/prod_canary/`.

## Preconditions

- BingX account is in **hedge mode** (`dualSidePosition=true`).
- You can restrict to **exactly one symbol** (e.g., `TRADING_SYMBOLS="BTC/USDT:USDT"`).
- API keys are provided via a **local gitignored env file** (recommended) or shell env:
  - `BINGX_KEY`, `BINGX_SECRET`

## Required Flags

PowerShell:

```powershell
# Real execution + production routing
$env:TRADING_MODE="live"
$env:EXECUTION_BACKEND="ccxt"
$env:BINGX_ENV="prod"

# Enable PROD Canary-0 mode (adds preflight + evidence writer)
$env:PROD_CANARY_0="true"
$env:PROD_CANARY_0_MAX_CLOSED_TRADES="1"
# Optional (default false): allow the canary to cancel/close existing symbol state on startup
# $env:PROD_CANARY_0_ALLOW_CLEANUP="true"

# Stage-2: hard stop only
$env:BINGX_NATIVE_HARD_STOP_ENABLED="true"
$env:BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED="false"

# Risk caps (example)
$env:TRADING_SYMBOLS="BTC/USDT:USDT"
$env:DAILY_MAX_TRADES="1"
$env:TRADING_DURATION="900"
```

Run:

```powershell
.\.venv\Scripts\python.exe src\main.py --live
```

## What “Pass” Looks Like (Logs)

- Canary gate:
  - `[PROD-CANARY-0] ENABLED | TRADING_MODE=live EXECUTION_BACKEND=ccxt BINGX_ENV=prod ...`
- Routing guard:
  - `[BINGX-ENV] env=prod ccxt_sandbox=False rest_base_url=https://open-api.bingx.com`
- Native hard stop placed:
  - `[BINGX-NATIVE] HARD_STOP placed position_id=... order_id=...`
- On close, cancel-on-close is attempted (idempotent-ok allowed):
  - `[BINGX-NATIVE] cancel ok (close:...) ...` or `cancel idempotent-ok ...`
- Evidence written:
  - `[PROD-CANARY-0] Wrote summary: diagnostics/prod_canary/prod_canary_summary_YYYYMMDD_HHMMSS.json`

## What “Fail” Looks Like (Immediate Abort)

- Any hard stop placement failure event:
  - `NATIVE_HARD_STOP_PLACE_FAILED {...}`
- Canary abort marker + immediate market close:
  - `[PROD-CANARY-0] Native HARD_STOP missing after entry; closing position immediately. ...`

## Evidence Artifacts

Created in `diagnostics/prod_canary/` (gitignored):

- `diagnostics/prod_canary/prod_canary_summary_latest.json`
- `diagnostics/prod_canary/prod_canary_summary_YYYYMMDD_HHMMSS.json`

Verify in the JSON:

- `env.BINGX_ENV == "prod"` and `bingx.ccxt_sandbox == false`
- `invariants.native_hard_stop_present_for_all_trades == true`
- `abort_reason == null`
- `trades[0].native_hard_stop.order_id` is non-null
- `final_exchange_state` shows no leftover open orders / positions for the symbol

## Rollback (Stop Canary Immediately)

```powershell
$env:BINGX_NATIVE_HARD_STOP_ENABLED="false"
$env:BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED="false"
$env:EXECUTION_BACKEND="simulated"
```

