# VST Run Quickstart (Docker, Minimal-Risk Smoke)

Goal: run the **full bot** in **BingX VST** with the smallest safe surface and clear runtime confirmation.

This profile uses:

- `TRADING_MODE=live` + `EXECUTION_BACKEND=ccxt` + `BINGX_ENV=vst` (real execution in VST)
- Native stops: **hard stop ON**, **trailing OFF**
- Full-bot canary guardrails: preflight + stop after **1** closed trade

## 0) Put VST keys in a gitignored env file

Create `bearish-bot.env.local` (already gitignored) with:

```
BINGX_KEY=...
BINGX_SECRET=...
```

Do not commit or paste the secrets into commands.

## 1) Build the image

```powershell
docker build -f docker/Dockerfile -t bearish-alpha-bot:local .
```

## 2) Run (LONG smoke, 1 trade max)

```powershell
docker run --rm -it --name bearish-vst-smoke-long `
  --env-file .\bearish-bot.env.local `
  -e TRADING_MODE=live `
  -e EXECUTION_BACKEND=ccxt `
  -e BINGX_ENV=vst `
  -e TRADING_SYMBOLS=BTC/USDT:USDT `
  -e VST_FULLBOT_CANARY=true `
  -e VST_FULLBOT_CANARY_SIDE=long `
  -e VST_FULLBOT_CANARY_MAX_CLOSED_TRADES=1 `
  -e VST_FULLBOT_CANARY_FORCE_MARKET=true `
  -e BINGX_NATIVE_HARD_STOP_ENABLED=true `
  -e BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED=false `
  bearish-alpha-bot:local python -u scripts/live_trading_launcher.py --live --duration 900
```

## 3) Run (SHORT smoke, 1 trade max)

```powershell
docker run --rm -it --name bearish-vst-smoke-short `
  --env-file .\bearish-bot.env.local `
  -e TRADING_MODE=live `
  -e EXECUTION_BACKEND=ccxt `
  -e BINGX_ENV=vst `
  -e TRADING_SYMBOLS=BTC/USDT:USDT `
  -e VST_FULLBOT_CANARY=true `
  -e VST_FULLBOT_CANARY_SIDE=short `
  -e VST_FULLBOT_CANARY_MAX_CLOSED_TRADES=1 `
  -e VST_FULLBOT_CANARY_FORCE_MARKET=true `
  -e BINGX_NATIVE_HARD_STOP_ENABLED=true `
  -e BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED=false `
  bearish-alpha-bot:local python -u scripts/live_trading_launcher.py --live --duration 900
```

## 4) What to verify in logs

Look for the single-line startup banner:

- `[MODE-BANNER] TRADING_MODE=live EXECUTION_BACKEND=ccxt BINGX_ENV=vst | CCXT_SANDBOX=true REST_BASE_URL=https://open-api-vst.bingx.com | NATIVE_HARD_STOP=true NATIVE_TRAILING=false`

And the BingX routing line:

- `[BINGX-ENV] env=vst ccxt_sandbox=True rest_base_url=https://open-api-vst.bingx.com`

If you see `TRADING_MODE=paper` or `BINGX_ENV=prod` in the banner, stop immediately (misconfiguration).

## (Optional) VST Demo Balance Auto-Topup (getVst)

BingX VST provides a demo-only endpoint `POST /openApi/swap/v2/trade/getVst` that can return the current VST balance
and optionally adjust it (top-up). Our VST preflight can use it **only when `BINGX_ENV=vst`**.

Enable auto-topup (recommended for VST canaries):

```powershell
$env:BINGX_VST_AUTO_TOPUP_ENABLED="true"
$env:BINGX_VST_TOPUP_THRESHOLD="20000"
$env:BINGX_VST_TOPUP_AMOUNT="100000"
$env:BINGX_VST_RECV_WINDOW_MS="10000"
```

Expected log signature (when a top-up is needed):

- `[VST-TOPUP] balance below threshold: balance=... threshold=...; requesting topup=...`

Notes:
- The getVst endpoint is **VST-host-only**; the client fails fast if the base URL is not `https://open-api-vst.bingx.com`.
- Never log API keys/secrets; use `--env-file bearish-bot.env.local` or a secret manager.

Manual balance query / top-up (optional; makes a real VST API call):

```powershell
.\.venv\Scripts\python.exe -c "import os; from core.ccxt_client import CcxtClient; from core.bingx_vst_balance import BingxVstBalanceClient; c=CcxtClient('bingx', {'apiKey': os.environ['BINGX_KEY'], 'secret': os.environ['BINGX_SECRET']}); v=BingxVstBalanceClient(api_key=os.environ['BINGX_KEY'], secret_key=os.environ['BINGX_SECRET'], base_url=c._bingx_rest_base_url, recv_window_ms=int(os.getenv('BINGX_VST_RECV_WINDOW_MS','5000'))); print('vst_balance=', v.get_vst_balance().balance);"
```
