# Bot Rehberi (CI/CD + Çalıştırma)

Bu doküman, botun **GitHub Actions** içinde nasıl çalıştığını ve temel konfigürasyonları özetler.

## Ortam / Sürümler
- **Python:** 3.12
- **Önemli paketler:** ccxt 4.3.88, pandas ≥ 2.2.3, numpy ≥ 2.2.6, pandas-ta 0.4.67b0, python-telegram-bot 21.6

## Secrets
- `EXCHANGES` → `kucoinfutures,binance,bitget` (örnek, varsayılan kucoinfutures)
- Borsa anahtarları: `BINGX_KEY`, `BINGX_SECRET`, `BITGET_*`, `BINANCE_*`, `KUCOIN_*`, …
- (Opsiyonel) Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- (Opsiyonel) `EXECUTION_EXCHANGE`: emir/varsayılan borsa seçimi (varsayılan: kucoinfutures)

## Çalıştırma
- **Run Bot Once (Orchestrated):** manuel tarama ve sinyal üretimi (paper mode)  
  Artefact: `data/RUN_SUMMARY.txt` (+ varsa `data/signals_*.csv`)
- **Run Bot Live:** manuel tarama ve GERÇEK emir girişi (live mode)  
  - Onay gerektirir: `confirm_live` girişi `YES` olmalı
  - Minimal güvenli risk limitleriyle çalışır
  - Yalnızca `EXECUTION_EXCHANGE` üzerinde emir girer
  - Artefact: `data/**`, `state.json`, `day_stats.json`
- **Backtest’ler:**  
  - OB: `.github/workflows/backtest.yml` → `src/backtest/param_sweep.py`  
  - STR: `.github/workflows/backtest_str.yml` → `src/backtest/param_sweep_str.py`
- **Nightly:** `.github/workflows/nightly_backtests.yml` → OB+STR sweep, `REPORT.md` üretimi  
  - Semboller bash döngüsü ile (matrix yerine) virgüllü string’den okunur.

## Mimaride Önemli Noktalar
- `core/ccxt_client.py` → `ohlcv()` 3 denemeli, son hatayı **aynıyla** yeniden fırlatır.
- `core/indicators.py` → `add_indicators(df, cfg)` RSI/EMA/ATR sağlar (pandas-ta’ya bağımlı değil).
- `src/main.py` → veri yeterlilik guard’ı (min bar), enrich sonrası `dropna()`, RUN_SUMMARY yazımı.

## Test İpuçları
- Rejim filtresi kapalı test: `signals.*.ignore_regime: true`
- OB sinyali artırmak için: `rsi_max` ↑, `tp_pct` ↓
- STR sinyali artırmak için: `rsi_min` ↓
