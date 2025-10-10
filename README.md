# Bearish-Alpha-Bot v0.3.2

### Yeni özellikler
- **Exchange limits**: min/max lot ve min notional kontrolü (ccxt markets.limits)
- **TP/SL/Trail takibi**: `data/state.json` ile açık sinyaller TP/SL'e vurunca Telegram bildirimi + PnL tahmini (paper)
- **Günlük özet**: `data/day_stats.json`'dan günlük TP/SL ve PnL takip
- **Safety**: `daily_max_trades`, `max_notional_per_trade`, `send_all`, `push_trail_updates`

Artifacts: Actions → bot-data içinde `signals.csv`, `state.json`, `day_stats.json`.
