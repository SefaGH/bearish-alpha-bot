# Bearish-Alpha-Bot

Multi-timeframe bearish scalper with regime filter (4H), 30m/1h/4h EMA/RSI/ATR/ADX, two strategies: Short-the-Rip & Oversold Bounce. 
Auto symbol universe (BingX, Bitget, Binance, KuCoinFutures, AscendEX), risk sizing, Telegram alerts, paper/live, Docker, GitHub Actions.

> Research/Education only. No financial advice.

## Quickstart (GitHub Actions)
1) Add **Repository Secrets**:

- `MODE` (e.g. `paper`)
- `EXCHANGES` (e.g. `bingx,bitget,binance,kucoinfutures,ascendex`)
- `SYM_SOURCE` (`AUTO`)
- `EXECUTION_EXCHANGE` (`bingx`)
- Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional)
- Live keys (optional): `BINGX_*`, `BITGET_*`, `BINANCE_*`, `KUCOIN_*`, `ASCENDEX_*`

2) Push repo; `config/config.yaml` is auto-copied from example on first run.
3) Go to **Actions → Run workflow**.

## Config
See `config/config.example.yaml` for signals, risk, universe, execution.

## Docker
```
docker build -t bearish-alpha-bot:0.1 -f docker/Dockerfile .
docker run --env-file .env bearish-alpha-bot:0.1
```
