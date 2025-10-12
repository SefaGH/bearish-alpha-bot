# Nightly Backtests + Report

Bu iş akışı, her gece (veya manuel) **OversoldBounce** ve **ShortTheRip** taramalarını koşturur, sonuç CSV’lerini birleştirip **Markdown rapor** üretir ve (opsiyonel) Telegram’a kısa özet yollar.

## Kurulum
1. Dosyaları repo’ya ekleyin:
   - `scripts/summarize_backtests.py`
   - `.github/workflows/nightly_backtests.yml`
2. Secrets (Settings ▸ Secrets and variables ▸ Actions):
   - `EXCHANGES`: `bingx,bitget,binance,kucoinfutures` vb.
   - Borsa anahtarları: `BINGX_*`, `BITGET_*`, `BINANCE_*`, `KUCOIN_*`, (kullanacaklarınız)
   - (Opsiyonel) `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## Çalıştırma
- **Actions ▸ Nightly Backtests + Report ▸ Run workflow** ile manuel koşturun, ya da cron zamanı gelmesini bekleyin.
- Koşu sonunda “Artifacts” kısmından `nightly-backtest-report` paketini indirin. İçinde:
  - `data/backtests/REPORT.md` (özet rapor)
  - tüm CSV sonuçları (OB+STR)

## Notlar
- Sembol listesi `workflow_dispatch` girdi alanından virgüllü verilebilir (örn. `BTC/USDT,ETH/USDT,SOL/USDT`). Cron’da varsayılan `BTC/USDT,ETH/USDT` kullanılır.
- `REPORT.md` içinde her strateji ve sembol için **ilk 5** kombinasyon tablo olarak gelir (sıralama: `avg_pnl`, `win_rate`, `trades`, `rr`). Telegram özeti kısa başlık listesi gönderir.