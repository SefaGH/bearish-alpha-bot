# Backtest & Param Sweep (MVP)

Bu klasör, **OversoldBounce (30m)** stratejisi için hızlı parametre taraması yapmanızı sağlar.

## Nasıl Çalışır?
- **Veri**: 30m OHLCV ccxt üzerinden çekilir.
- **İndikatörler**: `core/indicators.enrich()` ile RSI/EMA/ATR hesaplanır.
- **Sinyal**: `rsi <= rsi_max` durumlarında giriş kabul edilir (long).
- **Simülasyon**: Bir sonraki mumun `high/low` değerleri kullanılarak TP/SL test edilir.
- **Grid**: `rsi_max`, `tp_pct`, `sl_atr_mult` kombinasyonları taranır.
- **Çıktı**: `data/backtests/<symbol>_<timestamp>.csv`

## Çalıştırma (lokal)
```bash
export EXCHANGES=bingx,bitget,binance,kucoinfutures
export EXECUTION_EXCHANGE=bingx
export TELEGRAM_BOT_TOKEN=...   # opsiyonel
export TELEGRAM_CHAT_ID=...     # opsiyonel

# Opsiyonel: config dosyası
export CONFIG_PATH=config/config.example.yaml

# Sembol ve sınır
export BT_SYMBOL=BTC/USDT
export BT_EXCHANGE=bingx
export BT_LIMIT=1000

python -u src/backtest/param_sweep.py
```

## Çalıştırma (GitHub Actions içi)
- `EXCHANGES`, ilgili borsa anahtarları ve `EXECUTION_EXCHANGE` Secrets olarak tanımlanmış olmalı.
- “Run workflow” ile manuel tetikleyebilir veya cron tanımlayabilirsiniz.

## Sonuçların Yorumlanması
CSV sütunları:
- `rsi_max`, `tp_pct`, `sl_atr_mult` → denenen parametreler
- `trades` → işlem sayısı
- `win_rate` → kazanma oranı (0–1)
- `avg_pnl` → işlem başı ortalama getiri (oransal)
- `rr` → ortalama kazanç / ortalama kayıp
- `net_pnl` → toplam getiri (oransal toplam)

> Notlar:
> - Bu MVP, **OversoldBounce** için hazırdır. `ShortTheRip` eklemesi için 1h EMA bant hizalaması ve zaman eşlemesi gerekecek; ikinci adımda genişleteceğiz.
> - Fill mantığı “bir sonraki bar içinde TP/SL görülürse” varsayımına dayanır; kayma/ücret eklemek isterseniz kolayca genişletilebilir.