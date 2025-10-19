# Bearish Alpha Bot

Kripto türev piyasalarında (özellikle USDT margined perpetual) **ayı piyasası odaklı** fırsatları tarayıp sinyal üreten, GitHub Actions üzerinden tamamen **tarayıcıdan** çalıştırılabilen bot.

## ✅ Son Güncellemeler (2025-10)

Bu bot ChatGPT ile oluşturulmuş, ancak önemli hatalar ve eksiklikler tespit edilip düzeltilmiştir:

- ✅ **KRİTİK: Pozisyon büyüklüğü hesaplama hatası düzeltildi** (10x hata yapıyordu!)
- ✅ Loglama sistemi eklendi
- ✅ Gelişmiş hata yönetimi
- ✅ Kapsamlı testler (9 test, hepsi geçiyor)
- ✅ Detaylı dokümantasyon

**📖 Detaylı değişiklikler için:** [docs/IYILESTIRMELER.md](docs/IYILESTIRMELER.md)

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
- **🆕 Monitoring & Alerting**: Real-time web dashboard, multi-channel alerts, performance analytics
  - Web-based dashboard with live WebSocket updates
  - Advanced alert management (Telegram, Discord, Webhook)
  - Performance metrics (Sharpe ratio, win rate, drawdown, etc.)
  - 📊 [Monitoring System Documentation](docs/MONITORING_SYSTEM.md)

## Hızlı Başlangıç (sadece GitHub)
1. **Secrets ayarla** (Repo → Settings → Secrets and variables → Actions)
   - `EXCHANGES`: örn. `bingx,binance,kucoinfutures`
   - Kullandığın borsa anahtarları: `BINGX_KEY`, `BINGX_SECRET`, … (spot/derivatives izinleri açık olmalı)
   - (Opsiyonel) `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - (Opsiyonel) `EXECUTION_EXCHANGE`: örn. `bingx`
2. **Python 3.11** ile çalıştır
   - Tüm workflow dosyalarında:
     ```yaml
     - uses: actions/setup-python@v5
       with:
         python-version: "3.11"
     ```
3. **requirements.txt** (3.11 uyumlu)
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
  monitoring/           # 🆕 Real-time monitoring & alerting
    dashboard.py        # Web dashboard with WebSocket
    alert_manager.py    # Multi-channel alerts
    performance_analytics.py  # Performance metrics
main.py                 # Orkestrasyon (RUN_SUMMARY yazıyor)
```

## Import Patterns & Usage

This project supports **both package-style and script-style execution** through a dual import strategy:

### Package-Style Execution (Recommended for Production)
```bash
# Run as a package module
python -m src.main

# Import in Python/Jupyter
import src.core.risk_manager
from src.utils.pnl_calculator import calculate_unrealized_pnl
```

### Script-Style Execution (For Development/Scripts)
```bash
# Add src to path (done automatically in scripts/)
export PYTHONPATH=/path/to/bearish-alpha-bot/src:$PYTHONPATH

# Run scripts directly
python scripts/live_trading_launcher.py

# Import without src prefix
from core.risk_manager import RiskManager
from utils.pnl_calculator import calculate_unrealized_pnl
```

### Technical Details
The core modules (`risk_manager.py`, `position_manager.py`, `realtime_risk.py`, `production_coordinator.py`) use a dual import strategy:
```python
try:
    # Absolute import for script execution
    from src.utils.pnl_calculator import calculate_unrealized_pnl
except ImportError:
    # Relative import for package context
    from ..utils.pnl_calculator import calculate_unrealized_pnl
```

This ensures compatibility across different execution contexts without breaking existing workflows.

## Sıkça Sorulanlar
- “Artefact yok uyarısı” → `RUN_SUMMARY.txt` her koşuda oluşturulur.  
- “Sinyal yok” → test için `ignore_regime: true` ve RSI eşiklerini gevşet; EXCHANGES’i genişlet; `min_bars` eşiğini düşür.  
- “IndexError iloc[-1]” → `main.py` veri yeterlilik ve `dropna()` guard’larıyla giderildi.

Daha fazla ayrıntı için `docs/` klasörüne bak.

## Dokümantasyon

### Genel Dokümantasyon
- 📘 [İyileştirmeler ve Değişiklikler](docs/IYILESTIRMELER.md) - Son yapılan düzeltmeler
- 📗 [Environment Variables](docs/ENV_VARIABLES.md) - Tüm environment variable'lar
- 📙 [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Sorun giderme kılavuzu
- 📕 [Workflows](docs/WORKFLOWS.md) - GitHub Actions kullanımı
- 📓 [Config Reference](docs/CONFIG_REFERENCE.md) - Config dosyası ayarları

### Phase 2.1: Market Data Pipeline (YENİ! ✨)
- 🔷 [**Phase 2.1 Comprehensive Guide**](docs/PHASE2_MARKET_DATA.md) - Tam dokümantasyon
- 🔷 [Market Data Pipeline Usage](docs/market_data_pipeline_usage.md) - Detaylı kullanım kılavuzu
- 🔷 [Implementation Details](IMPLEMENTATION_DATA_AGGREGATOR.md) - Teknik uygulama detayları

**Phase 2.1 Özellikleri:**
- ✅ Çoklu borsa veri toplama ve otomatik yedekleme
- ✅ Otomatik bellek yönetimi (circular buffers)
- ✅ Entegre göstergeler (RSI, ATR, EMA21/50/200)
- ✅ Sağlık izleme ve durum takibi
- ✅ Veri kalite skorlaması ve konsensüs oluşturma
- ✅ Üretim ortamı için hazır (16 test geçiyor ✅)

**Örnek Kullanım:**
```python
from core.multi_exchange import build_clients_from_env
from core.market_data_pipeline import MarketDataPipeline

# Borsalardan veri topla
clients = build_clients_from_env()
pipeline = MarketDataPipeline(clients)

# Veri akışlarını başlat
pipeline.start_feeds(['BTC/USDT:USDT', 'ETH/USDT:USDT'], ['30m', '1h'])

# Göstergelerle zenginleştirilmiş veri al
df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '30m')
```

**Daha Fazla Örnek:** `examples/market_data_pipeline_example.py`

### Task 7: Pipeline Mode Integration (YENİ! 🚀)

Pipeline mode, market data pipeline'ı ana bot'a entegre eder ve **60x daha hızlı** sinyal üretimi sağlar:

**Kullanım:**
```bash
# Pipeline mode (optimize edilmiş, sürekli çalışma)
python src/main.py --pipeline

# Geleneksel mode (tek seferlik)
python src/main.py

# Live trading mode
python src/main.py --live
```

**Avantajlar:**
- ⚡ **60x daha hızlı**: 30 saniyede bir kontrol (geleneksel: 30 dakika)
- 💾 **5x daha az API çağrısı**: Veri cache'leniyor
- 🔄 **Otomatik failover**: Bir borsa çökerse diğerlerinden veri alınır
- 🧠 **Bellek yönetimi**: Circular buffers ile kontrol

**GitHub Actions:**
- Workflow: `.github/workflows/bot_pipeline.yml`
- Otomatik çalışma: Her 15 dakikada bir
- Manuel tetikleme: Actions → Run Bot with Pipeline

**Dokümantasyon:**
- 📘 [Pipeline Mode Kullanım Kılavuzu](docs/PIPELINE_MODE.md)
- 💻 Örnek: `examples/pipeline_mode_example.py`
- 🧪 Test: `scripts/test_pipeline_integration.py`

## Test Etme

Bot çalışır durumda mı kontrol etmek için:

```bash
# Smoke test (önerilen)
python tests/smoke_test.py

# Tüm testler
pytest tests/ -v

# Sonuç: 9 passed ✅
```
