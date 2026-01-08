# BingX VST Canary (Stage‑1)

This repo includes a minimal **VST canary** that exercises the new Stage‑1 real execution surface:

- CCXT **MARKET** open + close
- REST **positions readback** (must be VST‑routed)
- best‑effort cleanup (cancel any leftover open orders)

## Prereqs

- A Python venv with dependencies installed (`.venv`).
- BingX **VST** API credentials (`BINGX_KEY`, `BINGX_SECRET`) provided via:
  - `bearish-bot.env.local` (recommended; gitignored), or
  - your shell environment (`$env:BINGX_KEY=...`, `$env:BINGX_SECRET=...`)
- Account must be in **hedge mode** (`dualSidePosition=true`). The script fails fast otherwise.

## Run (PowerShell)

```powershell
.venv\Scripts\python.exe diagnostics\bingx_vst_canary.py --symbol BTC/USDT:USDT --notional-usdt 5 --side long --env-file bearish-bot.env.local
```

## Outputs

- `diagnostics/vst/bingx_vst_canary_summary_*.json`
- `diagnostics/vst/bingx_vst_canary_*.log`

## Expected Log Signatures

- `🟢 [REAL EXECUTION]` for CCXT orders/cancels
- `🟢 [CANARY] Positions ... (REST)` showing REST readback after open/close
