# Production Canary Runbook: BingX Native Stops (Hard Stop First)

This runbook is for a **minimal-risk** first production canary using:

- Stage-1 real execution surface (`TRADING_MODE=live` + `EXECUTION_BACKEND=ccxt`)
- Stage-2 native conditional stops (feature-flagged; default OFF)

## Preconditions

- BingX account is in **hedge mode** (`dualSidePosition=true`).
- Production API keys are provided via a **local** env file (gitignored) or shell env:
  - `BINGX_KEY`, `BINGX_SECRET`
- Confirm you can restrict symbols to a single canary symbol via `TRADING_SYMBOLS` (example below).

## Canary Phases

### Canary-0 (Hard Stop Only)

**Goal:** prove native hard-stop placement + cancel-on-close + no double-exits.

Set env (PowerShell):

```powershell
$env:TRADING_MODE="live"
$env:EXECUTION_BACKEND="ccxt"
$env:BINGX_ENV="prod"

# Enable Production Canary-0 evidence mode (preflight + JSON summary writer)
$env:PROD_CANARY_0="true"
$env:PROD_CANARY_0_MAX_CLOSED_TRADES="1"

# Stage-2 flags (hard stop only)
$env:BINGX_NATIVE_HARD_STOP_ENABLED="true"
$env:BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED="false"

# Minimize scope/risk
$env:TRADING_SYMBOLS="BTC/USDT:USDT"
$env:DAILY_MAX_TRADES="1"
$env:TRADING_DURATION="900"
```

Run:

```powershell
.\.venv\Scripts\python.exe src\main.py --live
```

**Verify (must see in logs)**

- Endpoint routing:
  - `[BINGX-ENV] env=prod ccxt_sandbox=False rest_base_url=https://open-api.bingx.com`
- Real execution gate:
  - `REAL EXECUTION ENABLED (ccxt, BINGX_ENV=prod)`
- Native hard stop placement after entry:
  - `[BINGX-NATIVE] HARD_STOP placed position_id=... order_id=...`
- Evidence summary written (gitignored):
  - `[PROD-CANARY-0] Wrote summary: diagnostics/prod_canary/prod_canary_summary_....json`
- Cancel-on-close attempts on close paths:
  - `[BINGX-NATIVE] cancel ok (close:...) position_id=... order_id=...`
  - or `cancel idempotent-ok` if the exchange already removed it
- Fail-fast safety (if hard stop is missing):
  - `NATIVE_HARD_STOP_PLACE_FAILED {...}`
  - `[PROD-CANARY-0] Native HARD_STOP missing after entry; closing position immediately. ...`

### Canary-1 (Enable Trailing-on-Activation)

**Goal:** prove trailing is placed only after activation and remains stable.

Change only:

```powershell
$env:BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED="true"
```

**Verify**

- Trailing placement:
  - `[BINGX-NATIVE] TRAILING placed position_id=... order_id=...`
- Synthetic trailing/stop-loss suppression when native ids exist:
  - `[BINGX-NATIVE] TRAILING-STOP-HIT suppressed ...`
  - `[BINGX-NATIVE] STOP-LOSS-HIT suppressed ...`

### Expansion (After Canary-0/1 Pass)

- Increase notional slowly and keep `DAILY_MAX_TRADES` low.
- Add symbols gradually by extending `TRADING_SYMBOLS`.

## Guardrails

- Always set `BINGX_ENV` explicitly when `EXECUTION_BACKEND=ccxt` (fail-fast gate).
- Keep `TRADING_SYMBOLS` limited to canary symbols only.
- Keep Stage-2 flags default OFF outside canary windows.
- Monitor for any unexpected market exits while native order ids exist (should be ~0).

## Metrics / Log Signatures to Monitor

- Native placement success:
  - count of `[BINGX-NATIVE] HARD_STOP placed ...` vs placement failures
  - count of `[BINGX-NATIVE] TRAILING placed ...` vs placement failures
- Cancel success:
  - count of `cancel ok` + `cancel idempotent-ok` vs `cancel failed`
- Double-exit prevention:
  - count of `[BINGX-NATIVE] exchange reports position closed; skipping market exit`
  - count of `preflight_skip_market_exit=true` from `LiveTradingEngine`
- Orphans:
  - any openOrders remaining after positions are flat (should be ~0; still cancel-on-close best-effort)

## Rollback Criteria (Stop Canary Immediately)

- Any evidence of a **flip risk** (position reopened after a close).
- Conditional placement failures above a small threshold (e.g., >1% of attempts).
- Persistent orphan conditional orders after closing positions.
- Any “double-close” attempts (market exits placed while `native_*_order_id` exists).

Rollback action:

- Set `EXECUTION_BACKEND=simulated` (or `TRADING_MODE=paper`) and restart.
- Disable Stage-2 flags: `BINGX_NATIVE_HARD_STOP_ENABLED=false`, `BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED=false`.
