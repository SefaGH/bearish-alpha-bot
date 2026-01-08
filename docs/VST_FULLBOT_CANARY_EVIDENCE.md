# VST Full‑Bot Canary (Stage‑3) — Runbook + Evidence Artifacts

This runbook exercises the **full bot loop** (ProductionCoordinator → StrategyCoordinator → LiveTradingEngine) in **BingX VST** with:

- **real CCXT market execution** (Stage‑1 surface),
- **native hard stop on entry** + **native trailing on activation** (Stage‑2), and
- **Stage‑3 canary guards** (fail‑fast preflight + market‑only enforcement + JSON evidence output).

## Safety prerequisites

- Use **VST keys** only (never production keys).
- Run with **one symbol only**.
- Feature flags are **default OFF**; you must explicitly enable them.

## Required environment

Minimum required (PowerShell):

```powershell
$env:TRADING_MODE="live"
$env:EXECUTION_BACKEND="ccxt"
$env:BINGX_ENV="vst"
$env:TRADING_SYMBOLS="BTC/USDT:USDT"
```

Enable Stage-3 full-bot canary guards:

```powershell
$env:VST_FULLBOT_CANARY="1"
$env:VST_FULLBOT_CANARY_MAX_CLOSED_TRADES="1"
$env:VST_FULLBOT_CANARY_SIDE="long"   # or "short"
```

Optional (recommended for first run): fail‑fast only (no cleanup)

```powershell
$env:VST_FULLBOT_CANARY_ALLOW_CLEANUP="0"
```

Optional (only if you explicitly want the bot to cancel/close leftovers for the canary symbol):

```powershell
$env:VST_FULLBOT_CANARY_ALLOW_CLEANUP="1"
```

Enable Stage‑2 native stops (default OFF):

```powershell
$env:BINGX_NATIVE_HARD_STOP_ENABLED="true"
$env:BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED="true"
```

Recommended runtime cap:

```powershell
$env:TRADING_DURATION="1800"   # 30 minutes (adjust as needed)
```

## Trailing activation configuration

Native trailing is placed **only after activation becomes true**.
Activation threshold precedence is documented in `docs/TRAILING_ACTIVATION_SEMANTICS_AUDIT.md`.

To ensure trailing is eligible during the canary, set trailing enabled + activation threshold in your local config (`config/config.yaml`, gitignored), for example:

- `position_management.trailing_stop.trailing_stop_enabled: true`
- `position_management.trailing_stop.activation_threshold: 0.003`
- `position_management.trailing_stop.trailing_stop_distance: 0.002`

## Run command (full bot)

```powershell
.\.venv\Scripts\python.exe src\main.py --live
```

## Expected log markers

- VST routing:
  - `[BINGX-ENV] env=vst ccxt_sandbox=... rest_base_url=https://open-api-vst.bingx.com`
- Canary enabled + preflight:
  - `[VST-FULLBOT-CANARY] ENABLED | TRADING_MODE=live EXECUTION_BACKEND=ccxt BINGX_ENV=vst ...`
  - If dirty and cleanup disabled: `[VST-FULLBOT-CANARY] Preflight failed; aborting. ...`
- Market‑only enforcement (if a non‑market algo was requested):
  - `[VST-FULLBOT-CANARY] Forcing MARKET execution (requested=limit)`
- Native orders (Stage‑2):
  - `[BINGX-NATIVE] HARD_STOP placed position_id=... order_id=...`
  - `[BINGX-NATIVE] TRAILING placed position_id=... order_id=...` (only after activation)
- Exchange‑close detection (Stage‑2):
  - `Preflight: exchange already flat; skipping market order: ...` (when a conditional already closed)

## Evidence artifacts (gitignored)

After the run ends, a JSON summary is written to:

- `diagnostics/vst/vst_fullbot_canary_summary_latest.json`
- `diagnostics/vst/vst_fullbot_canary_summary_YYYYMMDD_HHMMSS.json`

The summary includes:

- resolved env flags (no secrets),
- BingX routing info (CCXT sandbox + swap URL + REST base URL),
- preflight before/after snapshots,
- closed trades with native order IDs + positionSide/workType/qty,
- final exchange state snapshot (openOrders + positions).
