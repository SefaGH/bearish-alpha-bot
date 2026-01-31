# Min stop-distance simulation (trade_id)

This is an **analysis-only** tool to answer:

> If we enforce a minimum stop distance (hard floor + ATR-based soft floor), what would have changed for specific trades?

It combines:
- `TRADE_CLOSED` (actual fills + actual exit)
- `SIGNAL_BREAKDOWN` (planned entry/stop/target + ATR-bps snapshot)
- Optional: real BTC path via 1m OHLCV fetch

## Script

- `scripts/simulate_min_stop_effect.py`

## Usage

### 1) Compute new stop + R/R only (no OHLCV)

Fast, offline-ish; uses `vol_atr_bps` from the log.

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 45d7611a 73923fd4 --log logs/live_trading_20260130_221520_876109.log --no-ohlcv
```

### 2) Full sim: fetch 1m OHLCV and test which hits first (stop vs target)

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 45d7611a 73923fd4 75e87dec 73395b99 --log logs/live_trading_20260130_221520_876109.log
```

### 3) Smart Entry sim (ATR-based LIMIT fill) + min-stop

This models the concept:

- Long: `limit = current_price - (ATR_price * k)`
- Short: `limit = current_price + (ATR_price * k)`

If the limit price is **not touched** within the timeout window, the trade is treated as **not taken** (`sim_exit_reason=no_fill`).

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 45d7611a 73923fd4 --log logs/live_trading_20260130_221520_876109.log --smart-entry --smart-atr-mult 0.5 --smart-timeout-minutes 15
```

Only apply Smart Entry when volatility is high (threshold in bps from `SIGNAL_BREAKDOWN.vol_atr_bps`):

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 45d7611a 73923fd4 --log logs/live_trading_20260130_221520_876109.log --smart-entry --smart-only-when-atr-bps-gte 12
```

Ignore the R/R gate (useful to purely see path impact):

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 45d7611a 73923fd4 --log logs/live_trading_20260130_221520_876109.log --ignore-rr
```

Scale target to preserve required R/R (models a consistent risk policy):

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/simulate_min_stop_effect.py 75e87dec --log logs/live_trading_20260130_221520_876109.log --scale-target-to-required-rr
```

### Parameters

- `--hard-floor-bps 15` hard minimum stop distance (bps)
- `--atr-mult 1.5` ATR soft floor multiplier
- `--max-signal-lookback-s 90` how far back to search for matching `SIGNAL_BREAKDOWN`
- `--warmup-minutes/--pre-pad-minutes/--post-pad-minutes` OHLCV fetch window
- `--max-sim-minutes 30` extend simulation horizon after entry
- `--cache-dir data/cache/ohlcv` caching for repeated runs
- `--tie-break stop|tp` if stop and target are touched in the same 1m candle
- `--ignore-rr` simulate even if widened stop fails required R/R
- `--scale-target-to-required-rr` move target to preserve required R/R after widening stop
- `--smart-entry` enable Smart Entry limit-fill simulation (requires OHLCV fetch)
- `--smart-atr-mult 0.5` Smart Entry limit offset multiplier
- `--smart-timeout-minutes 15` max minutes to wait for the limit fill
- `--smart-only-when-atr-bps-gte <bps>` only use Smart Entry above a volatility threshold

## Output

The table includes:
- Actual trade pnl% from `TRADE_CLOSED`
- Required R/R from the `R/R Analysis` block (nearest to the matched signal)
- New stop price from the minimum-stop rule
- New R/R, and whether it would still pass
- If OHLCV fetched: which would have hit first (new stop vs original target)

With Smart Entry enabled, it also includes:
- `entry_mode` (`market`, `smart_limit`, `smart_limit_no_fill`)
- `limit_price` (computed Smart Entry limit)
- `sim_entry_time` / `sim_entry_price` (filled limit entry when applicable)
