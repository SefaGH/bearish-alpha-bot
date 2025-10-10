# Bearish-Alpha-Bot v0.2

- One-shot run (cron */5) — no infinite loop
- Only EXECUTION_EXCHANGE sends Telegram
- Debounce/cooldown (default 300s) to prevent spam
- Proper price formatting via ccxt
- Robust universe builder (skips failing exchanges)
- Tightened universe defaults (top_n=8, min_quote_volume=10M)
- No pandas-ta dependency (pure pandas/numpy indicators)
