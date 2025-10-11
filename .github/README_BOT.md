# Bearish Alpha Bot – Detaylı Manifesto

## 1) Neden Bu Bot?
- Bearish (düşüş) rejimde **panik yerine sistematik fırsat** yakalamak.
- **Veri odaklı**, kurallı, tekrarlanabilir kısa vadeli işlemler (intraday).
- Küçük ama **sürdürülebilir** getiriler; risk her zaman önce.

## 2) Genel Mimari
- **Çoklu borsa**: BingX, Bitget, KuCoin Futures, Binance (opsiyonel: Ascendex)
- **Çoklu zaman dilimi**: 30m / 1h / 4h
- **İndikatör katmanı**: RSI, EMA(21/50/200), ATR (EMA/SMA)
- **Rejim filtresi (4h)**: Düşüş trendi teyit edilmeden sinyal yok
- **Stratejiler**:
  - **ShortTheRip**: 1h EMA bandına dokunan, RSI yüksek → short
  - **OversoldBounce**: RSI düşük (oversold) → kısa rebound long
- **Risk & Boyutlama**:
  - İşlem başı sabit **risk USD**
  - **ATR tabanlı SL/TP/Trailing**
  - Borsa **precision**, **min amount/notional** kurallarına otomatik uyum
  - **Sınıf limitleri** (ör. meme: max notional & risk cap)
  - Günlük işlem/kayıp limitleri, cooldown
- **Bildirim**: Telegram (sinyaller, debug, error, PnL özetleri)
- **Durum/İstatistik**: `state.json`, `day_stats.json`, `data/signals.csv`
- **CI/CD**: GitHub Actions (cron + manual), config fallback step

## 3) Evren (Universe) Mantığı
- **USDT-quoted** marketler taranır.
- `prefer_perps: true` → linear perps (swap)
- `prefer_perps: false` → **spot + linear perps** (önerilen)
- Hacim sıralaması: `quoteVolume` yoksa **`baseVolume` fallback**
- Filtreler: `min_quote_volume_usdt`, `max_symbols_per_exchange`,
  `exclude_stables`, `include/exclude/blacklist`
- Telegram/Log’a evren özeti basılabilir:
  - `[universe] bingx:80 | bitget:80 | kucoinfutures:80 | binance:80`

## 4) Strateji Kuralları (Özet)
### ShortTheRip
- **Koşullar** (örnek konfig):
  - 4h bearish rejimi teyit
  - 1h fiyat **EMA bandına dokunma/üstüne taşma**
  - RSI ≥ `rsi_min` (örn. 61–64 arası)
- **Çıkış**: `tp_pct`, `sl_atr_mult`, `trail_atr_mult` ile yönetilir.

### OversoldBounce
- **Koşullar** (örnek konfig):
  - 4h bearish rejimi teyit
  - 30m **RSI ≤ rsi_min** (örn. 25–27 arası)
- **Çıkış**: `tp_pct`, `sl_atr_mult`, trailing opsiyonu, `cool_down_min`

> Değerler **konfigden** gelir; küçük oynamalar ile sinyal frekansı ayarlanır.

## 5) Risk Yönetimi
- `equity_usd`, `per_trade_risk_pct` → **risk USD** hesaplanır.
- ATR tabanlı stop mesafesi ile **miktar (qty)** bulunur.
- **Min stop mesafesi** ve **exchange precision** kontrolleri yapılır.
- **Global limitler** ve **class_limits (meme)** devrede:
  - Örn. `meme.max_notional_per_trade: 20000`, `meme.risk_usd_cap: 300`
- Günlük kontrol: `daily_max_trades`, `daily_loss_limit_pct`, `cool_down_min`

## 6) Bildirim & Format
- **Sinyal mesajı** örneği:
```
🔴 [kucoinfutures] SHORT BTC/USDT @ 126062.8
TP~122911.2 SL~126920 Trail~126598.5 Qty~317
RSI 68.4 & touch 1h EMA band
Notional≈$40.0k  Risk≈$271.73
```
- **No signals** durumunda:
```
ℹ️ No signals this run. scanned=24 bearish_ok=6 signals_found=0 sent=0 open=0
```
- **Hata** durumunda traceback özeti Telegram’a düşer.

## 7) Kurulum ve Çalıştırma (Detay)
### Secrets
- Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Borsalar: `EXCHANGES`, `EXECUTION_EXCHANGE`, borsa API key/secret/password’leri
- (Opsiyonel) `MAX_SCAN_PER_EXCHANGE` (0 veya tanımlama → tüm semboller)

### Workflow (özet)
- Python kurulumu → `pip install -r requirements.txt`
- **Prepare config**: `config.yaml` yoksa `config.example.yaml` kopyalanır
- `python -u src/main.py`
- Artifacts: `data/`, `state.json`, `day_stats.json`

### Config (öneri)
```yaml
universe:
  min_quote_volume_usdt: 1_000_000
  max_symbols_per_exchange: 40
  exclude_stables: true
  prefer_perps: false
  include: []
  exclude: []
  blacklist: [USDT/USDT, USDC/USDT, FDUSD/USDT]

signals:
  short_the_rip: { enable: true, rsi_min: 61, sl_atr_mult: 1.2, tp_pct: 0.012, trail_atr_mult: 0.9, touch_mid_ema: true, confirm_mid_trend: true }
  oversold_bounce: { enable: true, rsi_min: 25, sl_atr_mult: 1.0, tp_pct: 0.016, trail_atr_mult: 1.0, cool_down_min: 5 }

risk:
  equity_usd: 10000
  per_trade_risk_pct: 0.004
  daily_loss_limit_pct: 0.05
  daily_max_trades: 25
  min_stop_pct: 0.003
  min_amount_behavior: "skip"
  min_notional_behavior: "skip"
  max_notional_per_trade: 100000
  risk_usd_cap: 1200

class_limits:
  meme: { max_notional_per_trade: 20000, risk_usd_cap: 300 }

notify:
  send_all: true
  min_cooldown_sec: 180
  push_no_signal: true
  push_debug: true
```

## 8) Teşhis İpuçları
- `[universe]` mesajı → borsa başına sembol sayısı
- `scanned/bearish_ok/signals_found/sent` → boru hattı nerede tıkanıyor?
- “No signals” uzarsa: önce **evren**i genişlet; sonra **eşikleri** ılımlı gevşet (RSI ±1 vs.).

---

> Bu manifesto, repo içindeki strateji ve mimariyi tek yerde toplamayı amaçlar. PR’larda değişiklikleri bu belgeye işlemeniz önerilir.
