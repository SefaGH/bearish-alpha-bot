# Trade time windows from log (trade_id)

This repo logs a `TRADE_CLOSED {json}` line that contains `entry_time` and `exit_time` for each trade. You can use that to derive the time range needed to fetch OHLCV (and include indicator warmup).

## Script

- `scripts/extract_trade_time_windows.py`

### Examples

Scan all logs under `logs/` and print a table:

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/extract_trade_time_windows.py 45d7611a 73923fd4 75e87dec
```

Scan a specific log file:

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/extract_trade_time_windows.py 45d7611a --log logs/live_trading_20260130_221520_876109.log
```

JSON output (useful to feed into another script):

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/extract_trade_time_windows.py 45d7611a --format json
```

### Window logic

The recommended OHLCV window is:

- `ohlcv_start = entry_time - warmup_minutes - pre_pad_minutes`
- `ohlcv_end   = exit_time + post_pad_minutes`

Defaults:
- `warmup_minutes=90` (enough for ATR(14) and similar)
- `pre_pad_minutes=5`
- `post_pad_minutes=10`

Override as needed:

```powershell
C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/extract_trade_time_windows.py 45d7611a --warmup-minutes 180 --pre-pad-minutes 10 --post-pad-minutes 15
```
