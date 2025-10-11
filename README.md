# Bearish Alpha Bot

Kısa vadeli (intraday) fırsatları **bearish** piyasa koşullarında tespit eden, çok borsalı ve çok zaman dilimli bir **sinyal üretim botu**.

## 🎯 Amaç
- 30m/1h/4h verileriyle **trend-takip + mean-reversion** setuplarını bulmak.
- **RSI/EMA/ATR** gibi indikatörlerle **Short the Rip** (short fırsatı) ve **Oversold Bounce** (oversold long) sinyalleri üretmek.
- Risk-öncelikli: küçük ama **istikrarlı** getiriler (%2–5 hedef aralığı).

## ⚙️ Nasıl Çalışır (Kısa)
1. **Universe**: Borsalardan (BingX, Bitget, KuCoin Futures, Binance; istersen Ascendex) USDT-quoted semboller seçilir (spot+perps konfige bağlı).
2. **Rejim filtresi (4h)**: Düşüş koşulu sağlanmazsa sinyal üretilmez.
3. **Sinyaller**: 30m/1h RSI/EMA/ATR ile ShortTheRip & OversoldBounce kuralları.
4. **Risk**: İşlem başı risk USD, ATR tabanlı SL/TP/Trailing, meme sınıfı limitleri.
5. **Bildirim**: Sinyaller Telegram’a gönderilir. Paper modda PnL/istatistik saklanır.

## 🚀 Hızlı Kurulum
> Lokal test gerekmiyor; GitHub Actions ile çalışır.

1. **Secrets** (GitHub → Settings → Secrets → Actions):
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - `EXCHANGES` (örn: `bingx,bitget,kucoinfutures,binance`)
   - `EXECUTION_EXCHANGE` (örn: `bingx`)
   - Borsa API’leri: `BINGX_API_KEY/SECRET_KEY`, `BITGET_API_KEY/SECRET_KEY/PASSWORD`,
     `KUCOIN_API_KEY/SECRET_KEY/PASSWORD`, `BINANCE_API_KEY/SECRET_KEY`
2. **Workflow**: `.github/workflows/bot.yml` dosyasını ekle (bizde hazır).
3. **Config**: `config/config.yaml` yoksa workflow otomatik `config.example.yaml`’dan kopyalar.
4. **Çalıştır**: GitHub → Actions → **Run workflow**.

## 📲 Örnek Telegram Çıktısı
```
[universe] bingx:80 | bitget:80 | kucoinfutures:80 | binance:80
🔴 [kucoinfutures] SHORT BTC/USDT @ 126062.8
TP~122911.2  SL~126920  Trail~126598.5  Qty~317
RSI 68.4 & touch 1h EMA band
Notional≈$40.0k  Risk≈$271.73
```
_No signals_ koşularında:
```
ℹ️ No signals this run. scanned=24 bearish_ok=6 signals_found=0 sent=0 open=0
```

## 🧩 Dizin Yapısı
```
bearish-alpha-bot/
 ├─ README.md                 # Bu dosya (özet & hızlı kurulum)
 ├─ .github/
 │   ├─ README_BOT.md         # Detaylı manifesto/dökümantasyon
 │   └─ workflows/
 │        └─ bot.yml          # CI/CD (cron + manual run)
 ├─ src/                      # Kodlar (main, universe, indicators, notify...)
 ├─ config/                   # config.yaml & config.example.yaml
 ├─ data/                     # sinyaller (csv) – artifacts
 ├─ state.json, day_stats.json# PnL/pozisyon istatistikleri – artifacts
```

---

> Repo kökü için bu özet yeter. Ayrıntılar için `.github/README_BOT.md` dosyasına bak.
