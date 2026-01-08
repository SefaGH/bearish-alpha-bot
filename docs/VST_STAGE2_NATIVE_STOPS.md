# Stage-2 VST Validation: Native Hard Stop + Activation-Time Trailing (BingX Swap)

This validates the Stage-2 feature-flagged integration that places **exchange-native conditional close orders** on BingX swap in **VST hedge mode**:

- Native hard stop: `stopLossPrice` + `workingType=MARK_PRICE` + `positionSide`
- Native trailing stop: `create_trailing_percent_order(...)` (placed only after trailing activates)

## Prereqs

- VST keys in `bearish-bot.env.local` (gitignored) or shell env vars.
- Account in **hedge mode** (`dualSidePosition=true`).

## Run (PowerShell)

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_native_stops_smoke.py --symbol BTC/USDT:USDT --notional-usdt 5 --side long --env-file bearish-bot.env.local
.\.venv\Scripts\python.exe diagnostics\bingx_vst_native_stops_smoke.py --symbol BTC/USDT:USDT --notional-usdt 5 --side short --env-file bearish-bot.env.local
```

## Evidence Runner (recommended)

Generates stable evidence artifacts (gitignored) including openOrders snapshots, suppression signals, and skip-market-exit behavior:

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_stage2_native_stops_evidence.py --symbol BTC/USDT:USDT --notional-usdt 5 --side both --env-file bearish-bot.env.local
```

For Stage-4 evidence (exchange-side hard-stop trigger + trailing canaries), see `docs/VST_STAGE4_CANARY_1_2_EVIDENCE.md`.

## Outputs

- `diagnostics/vst/bingx_vst_stage2_native_stops_smoke_*.json`
- `diagnostics/vst/bingx_vst_stage2_native_stops_smoke_*.jsonl`
- `diagnostics/vst/stage2_run_long.json` (plus `.jsonl` / `.log`)
- `diagnostics/vst/stage2_run_short.json` (plus `.jsonl` / `.log`)
- `diagnostics/vst/stage2_evidence.md`

## Expected

- Script prints `PASS | hard_stop_visible=True trailing_visible=True`
- JSON summary includes both order ids and confirms `positions_after_close_rest.data_len == 0`
