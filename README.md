# Bearish Alpha Bot

Kripto türev piyasalarında (özellikle USDT margined perpetual) **ayı piyasası odaklı** fırsatları tarayıp sinyal üreten, GitHub Actions üzerinden tamamen **tarayıcıdan** çalıştırılabilen bot.

## Özellikler
- **Çoklu borsa**: BingX, Binance, KuCoin Futures, Bitget (CCXT)
- **Sinyaller**:
  - Oversold Bounce (30m)
  - Short The Rip (30m + 1h bağlam)
- **Rejim filtresi** (4h bearish) – test amaçlı kapatılıp açılabilir
- **Telegram bildirimi**
- **CSV çıktı** (artefact)
- **Backtest & Param tarama**: OB ve STR için Actions ile tek tık
- **Nightly raporlama**: OB+STR sweep + Markdown rapor + (opsiyonel) Telegram özet

## Hızlı Başlangıç (sadece GitHub)
1. **Secrets ayarla** (Repo → Settings → Secrets and variables → Actions)
   - `EXCHANGES`: örn. `bingx,binance,kucoinfutures`
   - Kullandığın borsa anahtarları: `BINGX_KEY`, `BINGX_SECRET`, … (spot/derivatives izinleri açık olmalı)
   - (Opsiyonel) `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - (Opsiyonel) `EXECUTION_EXCHANGE`: örn. `bingx`
2. **Python 3.12** ile çalıştır
   - Tüm workflow dosyalarında:
     ```yaml
     - uses: actions/setup-python@v5
       with:
         python-version: "3.12"
     ```
3. **requirements.txt** (3.12 uyumlu)
   ```text
   ccxt==4.3.88
   pandas>=2.2.3,<3
   numpy>=2.2.6
   python-dotenv==1.0.1
   pyyaml==6.0.2
   requests==2.32.3
   python-telegram-bot==21.6
   pandas-ta==0.4.67b0
   ```
4. **Botu bir kez çalıştır**  
   Actions → **Run Bot Once (Orchestrated)** → Run  
   - Telegram: “tarama başlıyor” + sinyal/uyarı mesajları  
   - Artefact: `bot-run` içinde `RUN_SUMMARY.txt` ve varsa `data/signals_*.csv`

## Çalışma Akışı (MVP)
```
ENV/Secrets → CCXT veri çekimi (30m/1h/4h) → indikatörler (RSI/EMA/ATR) →
4h regime (opsiyonel) → OB/STR stratejileri → Telegram → CSV artefact
```

## Yapı
```
src/
  core/
    ccxt_client.py      # ccxt sarmalayıcı (retry’li OHLCV)
    indicators.py       # add_indicators(...) → ema21/50/200, rsi, atr
    multi_exchange.py   # ENV’den borsa client’ları
    notify.py           # Telegram
    regime.py           # 4h bearish kontrolü
  strategies/
    oversold_bounce.py
    short_the_rip.py
  backtest/
    param_sweep.py      # OB param tarama (Actions)
    param_sweep_str.py  # STR param tarama (Actions)
main.py                 # Orkestrasyon (RUN_SUMMARY yazıyor)
```

## Sıkça Sorulanlar
- “Artefact yok uyarısı” → `RUN_SUMMARY.txt` her koşuda oluşturulur.  
- “Sinyal yok” → test için `ignore_regime: true` ve RSI eşiklerini gevşet; EXCHANGES’i genişlet; `min_bars` eşiğini düşür.  
- “IndexError iloc[-1]” → `main.py` veri yeterlilik ve `dropna()` guard’larıyla giderildi.

Daha fazla ayrıntı için `docs/` klasörüne bak.


## Test Update
- Manual PR test: 2025-10-12 21:57 UTC
