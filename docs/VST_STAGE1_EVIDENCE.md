# Stage‑1 VST Canary Evidence (BingX Swap)

This documents reproducible proof that **Stage‑1 real execution** works end‑to‑end in **BingX VST** with unified routing (CCXT + REST) and without simulated fills.

## Artifacts (latest)

- Long canary summary: `diagnostics/vst/bingx_vst_canary_summary_latest_long.json`
- Long canary log: `diagnostics/vst/bingx_vst_canary_latest_long.log`
- Short canary summary: `diagnostics/vst/bingx_vst_canary_summary_latest_short.json`
- Short canary log: `diagnostics/vst/bingx_vst_canary_latest_short.log`

## What “PASS” means (acceptance criteria)

- **Real order ids** returned for open/close (`open_order_id`, `close_order_id`)
- **Positions readback flat** after close (`positions_after_close_rest.data_len == 0`)
- **VST endpoints** in both layers:
  - CCXT: `ccxt_api_swap_url` contains `open-api-vst`
  - REST: `rest_base_url` is `https://open-api-vst.bingx.com`
- **No simulated fills** in logs (no `Order filled (simulated)` / no `[SIMULATED]`)

## Reproduce

1. Put VST API keys in `bearish-bot.env.local` (gitignored; never commit).
2. Run:
   - `.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary.py --symbol BTC/USDT:USDT --notional-usdt 5 --side long --env-file bearish-bot.env.local`
   - `.\.venv\Scripts\python.exe diagnostics\bingx_vst_canary.py --symbol BTC/USDT:USDT --notional-usdt 5 --side short --env-file bearish-bot.env.local`
3. Inspect the "latest" artifacts listed above.

## Log excerpts (proof points)

From `diagnostics/vst/bingx_vst_canary_latest_long.log`:

- `[BINGX-ENV] env=vst ccxt_sandbox=True rest_base_url=https://open-api-vst.bingx.com`
- `🟢 [REAL EXECUTION] Submitting MARKET order via CCXT (bingx)`
- `🟢 [REAL EXECUTION] Market order result: id=...`
- `🟢 [CANARY] Positions after close (REST): {'code': 0, 'data_len': 0}`

From `diagnostics/vst/bingx_vst_canary_latest_short.log`:

- `[BINGX-ENV] env=vst ccxt_sandbox=True rest_base_url=https://open-api-vst.bingx.com`
- `🟢 [REAL EXECUTION] Submitting MARKET order via CCXT (bingx)`
- `🟢 [REAL EXECUTION] Market order result: id=...`
- `🟢 [CANARY] Positions after close (REST): {'code': 0, 'data_len': 0}`
