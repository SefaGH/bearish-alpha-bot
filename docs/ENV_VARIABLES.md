# Environment Variables Reference

This document lists all environment variables used by the Bearish Alpha Bot.

## Exchange Configuration

### EXCHANGES (Required)
Comma-separated list of exchange names to use for scanning.

**Example:**
```bash
EXCHANGES=bingx,binance,kucoinfutures
```

**Supported exchanges:**
- `binance` - Binance Futures
- `bingx` - BingX
- `bitget` - Bitget
- `kucoin` - KuCoin Spot
- `kucoinfutures` - KuCoin Futures
- `ascendex` - AscendEX
- `bybit` - Bybit
- `okx` - OKX
- `gateio` - Gate.io
- `mexc` - MEXC

**Note:** Both `kucoin` and `kucoinfutures` can share the same `KUCOIN_*` credentials.

### EXECUTION_EXCHANGE
Exchange to use for live order execution (when MODE=live).

**Default:** `kucoinfutures`

**Example:**
```bash
EXECUTION_EXCHANGE=bingx
```

## Exchange API Credentials

For each exchange in EXCHANGES, provide credentials:

### Pattern: {EXCHANGE}_KEY, {EXCHANGE}_SECRET, {EXCHANGE}_PASSWORD

**Example for BingX:**
```bash
BINGX_KEY=your_api_key_here
BINGX_SECRET=your_api_secret_here
```

**Example for Bitget (requires password):**
```bash
BITGET_KEY=your_api_key_here
BITGET_SECRET=your_api_secret_here
BITGET_PASSWORD=your_api_passphrase_here
```

**Example for KuCoin (both spot and futures):**
```bash
# These credentials work for both 'kucoin' and 'kucoinfutures'
KUCOIN_KEY=your_api_key_here
KUCOIN_SECRET=your_api_secret_here
KUCOIN_PASSWORD=your_api_passphrase_here

# Alternative: use separate credentials for futures
KUCOINFUTURES_KEY=your_api_key_here
KUCOINFUTURES_SECRET=your_api_secret_here
KUCOINFUTURES_PASSWORD=your_api_passphrase_here
```

**Required credentials by exchange:**
- Most exchanges: `KEY` and `SECRET`
- Bitget, KuCoin, KuCoin Futures, AscendEX: `KEY`, `SECRET`, and `PASSWORD`

**Note:** KuCoin credentials are shared between `kucoin` (spot) and `kucoinfutures` by default. The system checks `KUCOIN_*` variables first, then falls back to `KUCOINFUTURES_*` for futures.

## Bot Operation Mode

### MODE
Operating mode for the bot.

**Options:**
- `paper` - Scan and generate signals only (no real trades)
- `live` - Execute real trades on EXECUTION_EXCHANGE

**Default:** `paper`

**Example:**
```bash
MODE=paper
```

## Telegram Notifications

### TELEGRAM_BOT_TOKEN
Telegram bot API token for sending notifications.

**Optional** - If not set, notifications are disabled.

**Example:**
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

### TELEGRAM_CHAT_ID
Telegram chat ID to receive notifications.

**Optional** - Required only if TELEGRAM_BOT_TOKEN is set.

**Example:**
```bash
TELEGRAM_CHAT_ID=-1001234567890
```

## Configuration

### CONFIG_PATH
Path to the YAML configuration file.

**Default:** `config/config.example.yaml`

**Example:**
```bash
CONFIG_PATH=config/my_config.yaml
```

## Risk Management (Live Mode Only)

These parameters control position sizing and risk limits when MODE=live.

### RISK_EQUITY_USD
Total account equity in USD for risk calculations.

**Default:** `1000`

**Example:**
```bash
RISK_EQUITY_USD=5000
```

### RISK_PER_TRADE_RISK_PCT
Percentage of equity to risk per trade (as decimal).

**Default:** `0.01` (1%)

**Example:**
```bash
RISK_PER_TRADE_RISK_PCT=0.02  # 2% per trade
```

### RISK_RISK_USD_CAP
Maximum risk amount in USD per trade (hard cap).

**Default:** `50`

**Example:**
```bash
RISK_RISK_USD_CAP=100
```

### RISK_MAX_NOTIONAL_PER_TRADE
Maximum position size (notional value) in USD per trade.

**Default:** `500`

**Example:**
```bash
RISK_MAX_NOTIONAL_PER_TRADE=1000
```

### RISK_MIN_STOP_PCT
Minimum stop loss distance as percentage (prevents over-leveraging).

**Default:** `0.005` (0.5%)

**Example:**
```bash
RISK_MIN_STOP_PCT=0.01  # 1% minimum stop distance
```

### RISK_DAILY_MAX_TRADES
Maximum number of trades allowed per day.

**Default:** `5`

**Example:**
```bash
RISK_DAILY_MAX_TRADES=10
```

### RISK_MIN_AMOUNT_BEHAVIOR
Behavior when calculated position size is below exchange minimum.

**Options:**
- `skip` - Skip the trade
- `scale` - Increase to minimum allowed size

**Default:** `skip`

**Example:**
```bash
RISK_MIN_AMOUNT_BEHAVIOR=scale
```

### RISK_MIN_NOTIONAL_BEHAVIOR
Behavior when position notional value is below exchange minimum.

**Options:**
- `skip` - Skip the trade
- `scale` - Increase to minimum notional

**Default:** `skip`

**Example:**
```bash
RISK_MIN_NOTIONAL_BEHAVIOR=skip
```

## Execution Parameters

### EXEC_FEE_PCT
Trading fee percentage (as decimal) for slippage calculations.

**Default:** `0.0006` (0.06%)

**Example:**
```bash
EXEC_FEE_PCT=0.0004  # 0.04% for VIP tier
```

### EXEC_SLIP_PCT
Expected slippage percentage (as decimal).

**Default:** `0.0005` (0.05%)

**Example:**
```bash
EXEC_SLIP_PCT=0.001  # 0.1% slippage
```

## Backtest Parameters

Used by backtest scripts (param_sweep.py, param_sweep_str.py).

### BT_SYMBOL
Symbol to backtest.

**Example:**
```bash
BT_SYMBOL=BTC/USDT
```

### BT_EXCHANGE
Exchange to use for backtest data.

**Example:**
```bash
BT_EXCHANGE=binance
```

### BT_LIMIT
Number of candles to fetch for backtesting (for OversoldBounce).

**Default:** `1000`

**Example:**
```bash
BT_LIMIT=2000
```

### BT_LIMIT_30M
Number of 30-minute candles for ShortTheRip backtest.

**Default:** `1000`

**Example:**
```bash
BT_LIMIT_30M=1500
```

### BT_LIMIT_1H
Number of 1-hour candles for ShortTheRip backtest.

**Default:** `1000`

**Example:**
```bash
BT_LIMIT_1H=1500
```

## Logging

### LOG_LEVEL
Logging verbosity level.

**Options:**
- `DEBUG` - Very verbose, for development
- `INFO` - Normal operational messages
- `WARNING` - Warning messages only
- `ERROR` - Error messages only
- `CRITICAL` - Critical errors only

**Default:** `INFO`

**Example:**
```bash
LOG_LEVEL=DEBUG
```

## Data Directories

### BT_DIR
Directory for backtest results.

**Default:** `data/backtests`

**Example:**
```bash
BT_DIR=backtests/results
```

### OUT_MD
Output path for backtest summary report.

**Default:** `data/backtests/REPORT.md`

**Example:**
```bash
OUT_MD=reports/summary.md
```

## Complete Example

Here's a complete example `.env` file for paper trading:

```bash
# Exchanges
EXCHANGES=binance,bingx
EXECUTION_EXCHANGE=binance

# Binance credentials
BINANCE_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret

# BingX credentials  
BINGX_KEY=your_bingx_api_key
BINGX_SECRET=your_bingx_secret

# Telegram
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-1001234567890

# Mode
MODE=paper

# Config
CONFIG_PATH=config/config.example.yaml

# Logging
LOG_LEVEL=INFO
```

For live trading, add risk parameters:

```bash
# ... (same as above, but with MODE=live)

# Risk Management
RISK_EQUITY_USD=5000
RISK_PER_TRADE_RISK_PCT=0.01
RISK_RISK_USD_CAP=50
RISK_MAX_NOTIONAL_PER_TRADE=500
RISK_MIN_STOP_PCT=0.005
RISK_DAILY_MAX_TRADES=5
RISK_MIN_AMOUNT_BEHAVIOR=skip
RISK_MIN_NOTIONAL_BEHAVIOR=skip

# Execution
EXEC_FEE_PCT=0.0006
EXEC_SLIP_PCT=0.0005
```
