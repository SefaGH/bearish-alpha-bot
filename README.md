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

### 🆕 Phase 2: Multi-Symbol Trading & Signal Acceptance Enhancement (2025-10-20)

Phase 2 introduces major improvements to signal generation and multi-symbol support:

- ✅ **Optimized Duplicate Prevention**: Reduced thresholds (0.05%, 20s) for >70% signal acceptance
- ✅ **Multi-Symbol Trading**: BTC, ETH, SOL with symbol-specific RSI thresholds
- ✅ **Enhanced Debug Logging**: Comprehensive [STR-DEBUG] format for all symbols
- ✅ **Symbol Independence**: Each symbol has independent duplicate prevention tracking
- ✅ **Price Delta Bypass**: Signals bypass cooldown when price moves >0.05%

**📖 Complete Phase 2 Documentation:** [docs/PHASE2_MULTI_SYMBOL_TRADING.md](docs/PHASE2_MULTI_SYMBOL_TRADING.md)

## Özellikler
- **Çoklu borsa**: BingX, Binance, KuCoin Futures, Bitget (CCXT)
- **Sinyaller**:
  - Oversold Bounce (30m)
  - Short The Rip (30m + 1h bağlam)
  - **🆕 Symbol-Specific Thresholds**: BTC/ETH/SOL with custom RSI levels
- **Rejim filtresi** (4h bearish) – test amaçlı kapatılıp açılabilir
- **Telegram bildirimi**
- **CSV çıktı** (artefact)
- **Backtest & Param tarama**: OB ve STR için Actions ile tek tık
- **Nightly raporlama**: OB+STR sweep + Markdown rapor + (opsiyonel) Telegram özet
- **🆕 Multi-Symbol Trading**: Trade BTC, ETH, SOL simultaneously with optimized signal acceptance
  - Symbol-independent duplicate prevention
  - Strategy-independent tracking
  - Price delta bypass (0.05% threshold)
  - 📊 [Phase 2 Documentation](docs/PHASE2_MULTI_SYMBOL_TRADING.md)
- **🆕 Monitoring & Alerting**: Real-time web dashboard, multi-channel alerts, performance analytics
  - Web-based dashboard with live WebSocket updates
  - Advanced alert management (Telegram, Discord, Webhook)
  - Performance metrics (Sharpe ratio, win rate, drawdown, etc.)
  - 📊 [Monitoring System Documentation](docs/MONITORING_SYSTEM.md)

## 📊 Exit Logic Validation & Session Summaries

The bot provides comprehensive exit event logging to validate Stop Loss (SL), Take Profit (TP), and Trailing Stop functionality (Issue #134).

### Exit Event Logging

All position exits are logged with clear indicators and detailed P&L information:

```
🛑 [STOP-LOSS-HIT] pos_BTC_1234567890
   Symbol: BTC/USDT:USDT
   Entry: $110000.00, Exit: $109500.00
   P&L: $-0.50 (-0.45%)
   Reason: STOP-LOSS

🎯 [TAKE-PROFIT-HIT] pos_ETH_1234567891
   Symbol: ETH/USDT:USDT
   Entry: $3500.00, Exit: $3552.50
   P&L: $+1.20 (+1.50%)
   Reason: TAKE-PROFIT

🚦 [TRAILING-STOP-HIT] pos_SOL_1234567892
   Symbol: SOL/USDT:USDT
   Entry: $145.00, Exit: $148.15
   P&L: $+0.70 (+2.17%)
   Reason: TRAILING-STOP
```

### Session Summary

At the end of each trading session, a comprehensive exit summary is logged:

```
======================================================================
📊 EXIT SUMMARY - Session Statistics
======================================================================
Total Exits: 8

Exits by Reason:
  🛑 Stop Loss:     3
  🎯 Take Profit:   4
  🚦 Trailing Stop: 1

Win/Loss Breakdown:
  ✅ Winning Trades: 5
  ❌ Losing Trades:  3
  📈 Win Rate:       62.50%

P&L Summary:
  Total P&L:    $+125.50
  Total Wins:   $+180.00
  Total Losses: $-54.50
  Avg Win:      $+36.00
  Avg Loss:     $-18.17
======================================================================
```

### Running Extended Sessions

To validate exit logic, run extended paper trading sessions:

```bash
# 30-minute session (1800 seconds)
python scripts/live_trading_launcher.py --paper --duration 1800

# 60-minute session (3600 seconds)
python scripts/live_trading_launcher.py --paper --duration 3600

# Indefinite paper trading (until manually stopped)
python scripts/live_trading_launcher.py --paper
```

### Validation Criteria

The system validates that:
- ✅ Stop Loss exits trigger when price reaches SL level
- ✅ Take Profit exits trigger when price reaches TP level
- ✅ Trailing Stop updates dynamically and triggers correctly
- ✅ Exit events are logged with clear reasons and P&L
- ✅ Session summaries provide win rate and overall statistics

### Testing Exit Logic

Comprehensive unit tests are available to validate exit logic:

```bash
# Run exit logic tests
python -m pytest tests/test_phase3_low_priority.py::TestExitLogicValidation -v

# Run all Phase 3 tests (exit logic + WebSocket performance)
python -m pytest tests/test_phase3_low_priority.py -v
```

**Tests Include:**
- Stop Loss hit detection and logging
- Take Profit hit detection and logging
- Trailing Stop exit detection
- Exit statistics calculation (win rate, P&L breakdown)
- Session summary generation

## 📡 WebSocket Performance Monitoring

The bot continuously monitors WebSocket performance and logs metrics every 60 seconds during trading sessions (Issue #135).

### Performance Logging

Real-time WebSocket performance is logged with the following metrics:

```
[WS-PERFORMANCE]
  Usage Ratio: 97.8%
  WS Latency: 18.3ms
  REST Latency: 234.7ms
  Improvement: 92.2%
```

**Metrics Explained:**
- **Usage Ratio**: Percentage of market data fetches served by WebSocket vs REST API
- **WS Latency**: Average latency for WebSocket data fetches in milliseconds
- **REST Latency**: Average latency for REST API data fetches in milliseconds
- **Improvement**: Performance improvement percentage of WebSocket over REST API

These metrics help validate that WebSocket optimization is working correctly and providing expected performance gains during live trading sessions.

### Testing WebSocket Performance

Unit tests verify WebSocket performance metrics:

```bash
# Run WebSocket performance tests
python -m pytest tests/test_phase3_low_priority.py::TestWebSocketPerformanceLogging -v
```

**Tests Include:**
- WebSocket fetch success/failure tracking
- REST API fallback tracking
- Latency metrics calculation
- Usage ratio calculation
- Performance improvement percentage
- Log format validation

## ⚙️ Duplicate Prevention Configuration

The bot includes intelligent duplicate signal prevention to avoid spam trades while remaining responsive to market movements. 

### Configuration (config/config.example.yaml)

```yaml
signals:
  duplicate_prevention:
    min_price_change_pct: 0.05  # Accept signals when price moves ≥0.05% (more sensitive)
    cooldown_seconds: 20        # Minimum 20s between signals for same symbol+strategy
```

### How It Works

The duplicate prevention system uses a **combined key approach** (`symbol:strategy`) that:

✅ **Allows**: Different strategies on same symbol (BTC+strategy1 → BTC+strategy2)  
✅ **Allows**: Same strategy on different symbols (BTC+strategy1 → ETH+strategy1)  
❌ **Blocks**: Repeated signals for same symbol+strategy within cooldown period

**Price-Based Bypass**: If price moves ≥ threshold (0.05%), the cooldown is bypassed automatically.

### Tuning Recommendations

| Trading Style | `min_price_change_pct` | `cooldown_seconds` | Description |
|--------------|----------------------|-------------------|-------------|
| **Scalping** (current) | 0.05 | 20 | Fast reaction, catches small moves |
| **Conservative** | 0.15 | 30 | Less noise, only significant moves |
| **Aggressive** | 0.03 | 15 | Maximum sensitivity, more signals |

**Current Setting (Issue #129)**: Optimized for better signal acceptance (70%+ acceptance rate) while preventing spam trades.

### Monitoring

Check duplicate prevention statistics in logs:
- Signal acceptance rate
- Bypass events (when price movement triggers bypass)
- Rejection reasons (cooldown vs. insufficient price delta)

## 🎯 Symbol-Specific Configuration

The bot supports **symbol-specific RSI thresholds** to optimize signal generation for different assets (Issue #131).

### Configuration

Add symbol-specific overrides in `config/config.example.yaml`:

```yaml
signals:
  short_the_rip:
    # Default parameters
    adaptive_rsi_base: 55
    adaptive_rsi_range: 10
    
    # Symbol-specific RSI threshold overrides
    symbols:
      "BTC/USDT:USDT":
        rsi_threshold: 55  # BTC: More selective
      "ETH/USDT:USDT":
        rsi_threshold: 50  # ETH: More sensitive
      "SOL/USDT:USDT":
        rsi_threshold: 50  # SOL: More sensitive
```

### How It Works

1. **Default Behavior**: All symbols use `adaptive_rsi_base` (e.g., 55 for shorts)
2. **Symbol Override**: If a symbol is configured in `symbols`, its specific threshold is used instead
3. **Debug Logging**: The bot logs which threshold is being applied for each symbol

### Tuning Guidelines

| Asset Type | Recommended RSI Threshold (Short) | Reasoning |
|------------|-----------------------------------|-----------|
| **Large Cap** (BTC) | 55-60 | More selective, wait for stronger overbought signals |
| **Mid Cap** (ETH) | 50-55 | Balanced approach |
| **Small Cap** (SOL, etc.) | 45-50 | More sensitive, catch earlier moves |

**For Long Strategies** (OversoldBounce): Use inverse logic (lower threshold = more selective)

### Debug Mode

Enable comprehensive debug logging to see why signals are/aren't generated:

```bash
# The bot automatically logs for each symbol:
[STR-DEBUG] ETH/USDT:USDT
  RSI: 52.3 (threshold: 50.0)
  ✅ RSI check passed: 52.3 >= 50.0
  EMA Align: ✅ (21=3890.45, 50=3905.23, 200=3920.12)
  Volume: 125430.50
  ATR: 45.2300
  ✅ Signal: SELL (RSI 52.3 >= 50.0, regime=neutral)
  Entry: $3895.20, Target: $3759.64, Stop: $3963.05, R/R: 2.00
```

### Troubleshooting

If a symbol is not generating signals:

1. **Check RSI values**: Look at debug logs to see current RSI vs. threshold
2. **Adjust threshold**: Lower for shorts (more signals), higher for longs
3. **Check EMA alignment**: Ensure EMA filters aren't too strict
4. **Verify data**: Ensure the symbol has sufficient historical data (120+ bars)

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
# Using relative path:
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
# Or using absolute path:
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run scripts directly
python scripts/live_trading_launcher.py

# Import without src prefix
from core.risk_manager import RiskManager
from utils.pnl_calculator import calculate_unrealized_pnl
```

### Technical Details
The core modules (`risk_manager.py`, `position_manager.py`, `realtime_risk.py`, `production_coordinator.py`) use a triple-fallback import strategy:
```python
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import calculate_unrealized_pnl
except ModuleNotFoundError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import calculate_unrealized_pnl
    except ModuleNotFoundError as e:
        # Option 3: Relative import (package context)
        if e.name in ('src', 'src.utils', 'src.utils.pnl_calculator'):
            from ..utils.pnl_calculator import calculate_unrealized_pnl
        else:
            raise
```

This ensures compatibility across different execution contexts without breaking existing workflows.

**Note:** The try/except pattern has no runtime performance impact - the ModuleNotFoundError only occurs once during module import, and the correct import path is cached by Python's import system.

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
