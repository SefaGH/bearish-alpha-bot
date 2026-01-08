# VST Stage-4 Evidence Runbook (BingX Native Stops)

This runbook produces **VST evidence** for:

1) exchange-side **hard stop trigger close** + `skip_market_exit` (Canary-1), and  
2) native **trailing placement/activation** evidence (Canary-2).

Artifacts are written to `diagnostics/vst/` (gitignored).

## Preconditions

- BingX account is in **hedge mode** (`dualSidePosition=true`).
- VST API keys are available via shell env or a gitignored env file:
  - `BINGX_KEY`, `BINGX_SECRET`

Recommended: create `bearish-bot.env.local` (already gitignored) with:

```
BINGX_KEY=...
BINGX_SECRET=...
```

## Canary-1 — Exchange-Side Hard Stop Trigger + skip_market_exit

Runs a tight stop and waits for the **exchange** to close the position. If it times out, it will market-close for safety and the run will be `ok=false`.

LONG:

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary1_hard_stop_trigger.py `
  --symbol BTC/USDT:USDT --side long --notional-usdt 5 `
  --stop-distance-pct 0.001 --timeout-s 180 --allow-cleanup
```

SHORT:

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary1_hard_stop_trigger.py `
  --symbol BTC/USDT:USDT --side short --notional-usdt 5 `
  --stop-distance-pct 0.001 --timeout-s 180 --allow-cleanup
```

Evidence files:

- `diagnostics/vst/vst_canary1_hard_stop_trigger_*.json`
- `diagnostics/vst/vst_canary1_hard_stop_trigger_*.jsonl`

Pass checklist (in JSON):

- `ok == true` and `timed_out == false`
- `hard_stop_order_id` is non-null
- `skip_market_exit_signal.skip_market_exit == true`
- `market_close_sent == false`
- `open_orders_at_end` empty (or only unrelated system orders)
- `positions_at_end_rest` shows `positionAmt==0` for the symbol/side

## Canary-2 — Native Trailing Placement/Activation

Places native hard stop on entry, then enables trailing and proves a native trailing order is created. Default activation is `0.0` (immediate) to keep the run deterministic; adjust as needed.

LONG:

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary2_trailing_activation.py `
  --symbol BTC/USDT:USDT --side long --notional-usdt 5 `
  --trailing-activation-threshold-pct 0.0 --trailing-distance-pct 0.002 `
  --timeout-s 60 --allow-cleanup
```

SHORT:

```powershell
.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary2_trailing_activation.py `
  --symbol BTC/USDT:USDT --side short --notional-usdt 5 `
  --trailing-activation-threshold-pct 0.0 --trailing-distance-pct 0.002 `
  --timeout-s 60 --allow-cleanup
```

Evidence files:

- `diagnostics/vst/vst_canary2_trailing_activation_*.json`
- `diagnostics/vst/vst_canary2_trailing_activation_*.jsonl`

Pass checklist (in JSON):

- `ok == true`
- `hard_stop_order_id` and `trailing_order_id` are non-null
- `open_orders_after_entry` shows the hard stop
- `open_orders_after_trailing` shows the trailing order (and hard stop, depending on coexistence)
- `positions_after_close_rest` shows `positionAmt==0` and no orphans remain after close

