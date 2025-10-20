🎨 **Bearish Alpha Bot - Comprehensive Architecture & Flow Diagrams**

---

## 📊 **MASTER ARCHITECTURE DIAGRAM**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     BEARISH ALPHA TRADING BOT                                ║
║                          Production v1.0                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENTRY POINT: main.py                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Load Configuration (config/config.yaml)                            │ │
│  │  2. Initialize Logging System                                          │ │
│  │  3. Print Startup Banner (with headers)                                │ │
│  │  4. Create LiveTradingEngine instance                                  │ │
│  │  5. Run engine: await engine.run()                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CORE ENGINE: LiveTradingEngine                             │
│                   (src/core/live_trading_engine.py)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ INITIALIZATION (__init__)                                              │ │
│  │ • Load config, set paper_mode                                          │ │
│  │ • Initialize components:                                               │ │
│  │   - ExchangeManager (Bybit API)                                        │ │
│  │   - PortfolioManager (capital tracking)                                │ │
│  │   - RiskManager (position sizing)                                      │ │
│  │   - SignalGenerator (strategy logic)                                   │ │
│  │   - AdvancedPositionManager (exit management)                          │ │
│  │   - WebSocketManager (real-time data)                                  │ │
│  │ • Setup statistics: ws_stats, ws_config                                │ │
│  │ • State: STOPPED                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ MAIN RUN METHOD (run())                                                │ │
│  │  1. State → STARTING                                                   │ │
│  │  2. Log startup message with headers                                   │ │
│  │  3. Start WebSocket streams (OHLCV for all symbols/timeframes)         │ │
│  │  4. Start position monitoring loop (background task)                   │ │
│  │  5. Start strategy execution loop (main task)                          │ │
│  │  6. State → RUNNING                                                    │ │
│  │  7. Wait for shutdown signal                                           │ │
│  │  8. State → STOPPING → STOPPED                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
         │                           │                           │
         │                           │                           │
         ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  WebSocket Loop  │    │  Strategy Loop       │    │  Position Monitor    │
│  (Background)    │    │  (Main Loop)         │    │  (Background)        │
└──────────────────┘    └──────────────────────┘    └──────────────────────┘
```

---

## 🔄 **DETAILED FLOW DIAGRAMS**

### **1. WEBSOCKET DATA FLOW**

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     WebSocket Manager (src/core/websocket_manager.py)     │
└───────────────────────────────────────────────────────────────────────────┘

    START
      │
      ▼
┌─────────────────────────────────────┐
│ start_ohlcv_stream(symbol, tf)     │
│ • Connect to Bybit WebSocket        │
│ • Subscribe to kline.<interval>     │
│ • Register callback handler         │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Message Received (Callback)         │
│ • Parse kline data                  │
│ • Convert to OHLCV format           │
│ • Store in self.data_cache          │
│ • Increment message_count           │
│ • Update last_message_time          │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ DATA CACHE STRUCTURE                │
│ {                                   │
│   "BTC/USDT:USDT": {                │
│     "1m": {                         │
│       "ohlcv": [                    │
│         [timestamp, o, h, l, c, v], │
│         [timestamp, o, h, l, c, v], │
│         ...                         │
│       ],                            │
│       "symbol": "BTC/USDT:USDT",    │
│       "timeframe": "1m"             │
│     }                               │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ HEALTH MONITORING                   │
│ • get_connection_health()           │
│   - status: healthy/warming_up/     │
│     connecting/disconnected         │
│   - active_streams: count           │
│   - total_messages: count           │
│   - uptime_seconds: duration        │
│                                     │
│ • is_data_fresh(symbol, tf)         │
│   - Check last candle age           │
│   - Return True if < 60s            │
│                                     │
│ • get_data_quality_score()          │
│   - Freshness: 40 pts               │
│   - Completeness: 30 pts            │
│   - Update frequency: 30 pts        │
│   - Total: 0-100                    │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ get_latest_data(symbol, tf)         │
│ • Return cached OHLCV data          │
│ • Used by strategy loop             │
└─────────────────────────────────────┘
```

---

### **2. STRATEGY EXECUTION FLOW**

```
┌───────────────────────────────────────────────────────────────────────────┐
│              Strategy Execution Loop (_strategy_execution_loop)           │
│                     Interval: 5 seconds (configurable)                    │
└───────────────────────────────────────────────────────────────────────────┘

    LOOP START (Every 5s)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ FOR each symbol in active_symbols:                          │
│   FOR each timeframe in active_timeframes:                  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Fetch OHLCV Data (WITH WEBSOCKET PRIORITY)         │
│ _get_ohlcv_with_priority(symbol, timeframe, limit=100)     │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ TRY WEBSOCKET FIRST:                                │   │
│ │ • Check ws_config['priority_enabled']               │   │
│ │ • Call ws_manager.is_data_fresh(symbol, tf, 60s)    │   │
│ │ • If fresh:                                         │   │
│ │   - Get ws_manager.get_latest_data()                │   │
│ │   - Validate len(ohlcv) >= limit                    │   │
│ │   - Record latency (ms)                             │   │
│ │   - _record_ws_fetch(latency, success=True)         │   │
│ │   - Log: 📡 [WS-DATA] symbol tf - X candles, Yms    │   │
│ │   - Return DataFrame                                │   │
│ │ • If fails: _record_ws_fetch(0, success=False)      │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ FALLBACK TO REST:                                   │   │
│ │ • _fetch_ohlcv_rest(symbol, tf, limit, start_time)  │   │
│ │ • For each exchange_client:                         │   │
│ │   - Try fetch_ohlcv()                               │   │
│ │   - Record latency                                  │   │
│ │   - _record_rest_fetch(latency, success=True)       │   │
│ │   - Log: 🌐 [REST-SUCCESS] symbol - Xms             │   │
│ │   - Return DataFrame                                │   │
│ │ • If all fail: return None                          │   │
│ └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Check Data Validity                                │
│ if ohlcv_df is None or ohlcv_df.empty:                     │
│   log warning, continue to next symbol                     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Calculate Indicators                               │
│ signal_generator.calculate_indicators(ohlcv_df)            │
│ • EMA calculations                                         │
│ • Volume analysis                                          │
│ • Price action patterns                                    │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Generate Signal                                    │
│ signal = signal_generator.generate_signal(                 │
│   symbol, timeframe, ohlcv_df                              │
│ )                                                           │
│                                                             │
│ Signal Structure:                                           │
│ {                                                           │
│   'action': 'long' | 'short' | None,                       │
│   'symbol': 'BTC/USDT:USDT',                               │
│   'timeframe': '15m',                                      │
│   'entry_price': 50000.00,                                 │
│   'stop_loss': 49000.00,                                   │
│   'take_profit': 55000.00,                                 │
│   'confidence': 0.85,                                      │
│   'timestamp': 1729425147000                               │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Check Signal Validity                              │
│ if signal is None or signal['action'] is None:             │
│   continue to next                                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: DUPLICATE PREVENTION CHECK (NEW!)                  │
│ position_manager.is_duplicate_signal(signal)               │
│                                                             │
│ Checks:                                                     │
│ • Recent signals cache (last 5 minutes)                    │
│ • Same symbol, side, timeframe                             │
│ • Entry price within 0.1% tolerance                        │
│                                                             │
│ if is_duplicate:                                            │
│   log: ⚠️ [DUPLICATE-SIGNAL] Ignoring...                   │
│   continue to next                                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: PORTFOLIO CAPITAL CHECK (NEW!)                     │
│ portfolio_manager.check_capital_limit()                    │
│                                                             │
│ • Get total_capital_used from active positions             │
│ • Check if total_capital_used >= max_capital              │
│                                                             │
│ if capital_limit_reached:                                  │
│   log: 💰 [CAPITAL-LIMIT] Cannot open new position        │
│   continue to next                                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: Risk Management & Position Sizing                  │
│ position_size = risk_manager.calculate_position_size(      │
│   signal, current_balance                                  │
│ )                                                           │
│                                                             │
│ Calculations:                                               │
│ • Risk per trade: 1% of balance                            │
│ • Risk amount = balance * 0.01                             │
│ • Price distance = |entry - stop_loss|                     │
│ • Position size = risk_amount / price_distance             │
│ • Apply min/max limits                                     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 9: Execute Trade                                      │
│ if paper_mode:                                             │
│   • Simulate order execution                               │
│   • Record to paper trading ledger                         │
│ else:                                                       │
│   • Place real order via exchange API                      │
│                                                             │
│ Create position record:                                    │
│ {                                                           │
│   'position_id': 'pos_123abc',                             │
│   'symbol': 'BTC/USDT:USDT',                               │
│   'side': 'long',                                          │
│   'entry_price': 50000.00,                                 │
│   'current_price': 50000.00,                               │
│   'amount': 0.01,                                          │
│   'stop_loss': 49000.00,                                   │
│   'take_profit': 55000.00,                                 │
│   'entry_time': timestamp,                                 │
│   'status': 'open'                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 10: Register Position                                 │
│ position_manager.add_position(position)                    │
│ • Add to active_positions dict                             │
│ • Add signal to recent_signals cache                       │
│ • Start monitoring for this position                       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
    NEXT SYMBOL/TIMEFRAME
      │
      ▼
    SLEEP (5 seconds)
      │
      ▼
    LOOP AGAIN
```

---

### **3. POSITION MONITORING FLOW**

```
┌───────────────────────────────────────────────────────────────────────────┐
│          Position Monitoring Loop (_position_monitoring_loop)             │
│                     Interval: 10 seconds (configurable)                   │
└───────────────────────────────────────────────────────────────────────────┘

    LOOP START (Every 10s)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Get Active Positions                                        │
│ positions = position_manager.get_active_positions()        │
│                                                             │
│ if no positions:                                            │
│   log: "No active positions to monitor"                    │
│   sleep 10s, continue                                      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Initialize Loop Variables                                   │
│ • total_unrealized_pnl = 0.0                               │
│ • positions_closed_count = 0                               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ FOR each position in active_positions:                     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Get Current Price (WEBSOCKET PRIORITY)             │
│ current_price = _get_current_price(symbol)                 │
│                                                             │
│ ┌───────────────────────────────────────────────────┐     │
│ │ TRY WEBSOCKET:                                    │     │
│ │ • Get ws_manager.get_latest_data(symbol, '1m')    │     │
│ │ • Extract last close price                        │     │
│ │ • If available: return price                      │     │
│ └───────────────────────────────────────────────────┘     │
│                                                             │
│ ┌───────────────────────────────────────────────────┐     │
│ │ FALLBACK TO REST:                                 │     │
│ │ • Fetch ticker from exchange                      │     │
│ │ • Return last price                               │     │
│ └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Update Position Price                              │
│ position['current_price'] = current_price                  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Calculate P&L                                      │
│ entry_price = position['entry_price']                      │
│ amount = position['amount']                                │
│ side = position['side']                                    │
│                                                             │
│ if side == 'long':                                         │
│   unrealized_pnl = (current - entry) * amount              │
│ else: # short                                              │
│   unrealized_pnl = (entry - current) * amount              │
│                                                             │
│ pnl_pct = (unrealized_pnl / (entry * amount)) * 100       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Enhanced P&L Logging (NEW!)                        │
│ logger.info(                                                │
│   f"💰 [P&L-UPDATE] {position_id}\n"                       │
│   f"   Symbol: {symbol}\n"                                 │
│   f"   Entry: ${entry_price:.2f}\n"                        │
│   f"   Current: ${current_price:.2f}\n"                    │
│   f"   Unrealized P&L: ${unrealized_pnl:.2f}               │
│      ({pnl_pct:+.2f}%)"                                    │
│ )                                                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Check Exit Conditions                              │
│ exit_check = position_manager.manage_position_exits(       │
│   position_id                                              │
│ )                                                           │
│                                                             │
│ ┌───────────────────────────────────────────────────┐     │
│ │ Exit Conditions Checked:                          │     │
│ │                                                    │     │
│ │ 1. STOP LOSS:                                     │     │
│ │    • Long: current_price <= stop_loss             │     │
│ │    • Short: current_price >= stop_loss            │     │
│ │    • Log: 🛑 [STOP-LOSS-HIT]                      │     │
│ │                                                    │     │
│ │ 2. TAKE PROFIT:                                   │     │
│ │    • Long: current_price >= take_profit           │     │
│ │    • Short: current_price <= take_profit          │     │
│ │    • Log: 🎯 [TAKE-PROFIT-HIT]                    │     │
│ │                                                    │     │
│ │ 3. TRAILING STOP (NEW!):                          │     │
│ │    • Track highest_price (long) / lowest (short)  │     │
│ │    • Calculate trailing_stop_level                │     │
│ │    • Exit if pullback exceeds distance            │     │
│ │    • Log: 📉 [TRAILING-STOP-HIT]                  │     │
│ └───────────────────────────────────────────────────┘     │
│                                                             │
│ Returns:                                                    │
│ {                                                           │
│   'should_exit': True/False,                               │
│   'exit_reason': 'stop_loss'|'take_profit'|                │
│                  'trailing_stop',                          │
│   'exit_price': current_price                              │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Handle Exit if Triggered                           │
│ if exit_check['should_exit']:                              │
│   ┌─────────────────────────────────────────────────┐     │
│   │ Log Enhanced Exit (NEW!):                       │     │
│   │ exit_emoji = 🛑 if SL else 🎯                   │     │
│   │ logger.warning(                                 │     │
│   │   f"{exit_emoji} [EXIT-SIGNAL] {pos_id}\n"     │     │
│   │   f"   Symbol: {symbol}\n"                      │     │
│   │   f"   Reason: {exit_reason}\n"                 │     │
│   │   f"   Entry: ${entry:.2f}\n"                   │     │
│   │   f"   Exit: ${current:.2f}\n"                  │     │
│   │   f"   P&L: ${pnl:.2f} ({pct:+.2f}%)"          │     │
│   │ )                                               │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ Execute Exit:                                   │     │
│   │ • Close position (API or paper)                 │     │
│   │ • Remove from active_positions                  │     │
│   │ • Update portfolio                              │     │
│   │ • Increment positions_closed_count              │     │
│   └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Accumulate Total P&L                               │
│ total_unrealized_pnl += unrealized_pnl                     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
    NEXT POSITION
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: Summary Logging (NEW!)                             │
│ if active_positions:                                        │
│   logger.info(                                              │
│     f"📊 [MONITORING-SUMMARY]\n"                           │
│     f"   Active Positions: {len(active_positions)}\n"      │
│     f"   Total Unrealized P&L: ${total_pnl:.2f}\n"        │
│     f"   Positions Closed: {positions_closed_count}"       │
│   )                                                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
    SLEEP (10 seconds)
      │
      ▼
    LOOP AGAIN
```

---

## 📁 **FILE STRUCTURE & RESPONSIBILITIES**

```
bearish-alpha-bot/
│
├── main.py ⭐ ENTRY POINT
│   ├── Load config/config.yaml
│   ├── Initialize logging
│   ├── Print startup banner (with headers) 🆕
│   ├── Create LiveTradingEngine
│   └── Run engine
│
├── config/
│   ├── config.yaml                    # Main configuration
│   └── config.example.yaml 🆕         # With WebSocket settings
│       ├── exchange settings
│       ├── trading parameters
│       ├── risk management
│       ├── portfolio limits 🆕
│       └── websocket settings 🆕
│           ├── priority_enabled: true
│           ├── max_data_age: 60
│           └── fallback_threshold: 3
│
├── src/
│   ├── core/
│   │   │
│   │   ├── live_trading_engine.py ⭐ CORE ENGINE
│   │   │   │
│   │   │   ├── CLASS: LiveTradingEngine
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Load config
│   │   │   │   │   ├── Initialize all managers
│   │   │   │   │   ├── Setup ws_config 🆕
│   │   │   │   │   └── Setup ws_stats 🆕
│   │   │   │   │
│   │   │   │   ├── run() - Main entry
│   │   │   │   │   ├── Log startup headers 🆕
│   │   │   │   │   ├── Start WebSocket streams
│   │   │   │   │   ├── Start position monitoring
│   │   │   │   │   └── Start strategy execution
│   │   │   │   │
│   │   │   │   ├── _strategy_execution_loop() 🔄
│   │   │   │   │   ├── For each symbol/timeframe:
│   │   │   │   │   ├── _get_ohlcv_with_priority() 🆕
│   │   │   │   │   ├── Calculate indicators
│   │   │   │   │   ├── Generate signal
│   │   │   │   │   ├── Check duplicate 🆕
│   │   │   │   │   ├── Check capital limit 🆕
│   │   │   │   │   ├── Risk management
│   │   │   │   │   └── Execute trade
│   │   │   │   │
│   │   │   │   ├── _position_monitoring_loop() 🔄
│   │   │   │   │   ├── For each active position:
│   │   │   │   │   ├── Get current price (WS priority)
│   │   │   │   │   ├── Calculate P&L
│   │   │   │   │   ├── Enhanced logging 🆕
│   │   │   │   │   ├── Check exit conditions
│   │   │   │   │   └── Summary stats 🆕
│   │   │   │   │
│   │   │   │   ├── 🆕 _get_ohlcv_with_priority()
│   │   │   │   │   ├── Try WebSocket first
│   │   │   │   │   │   ├── Check is_data_fresh()
│   │   │   │   │   │   ├── Get cached data
│   │   │   │   │   │   ├── Record metrics
│   │   │   │   │   │   └── Return DataFrame
│   │   │   │   │   └── Fallback to REST
│   │   │   │   │       ├── _fetch_ohlcv_rest()
│   │   │   │   │       └── Record metrics
│   │   │   │   │
│   │   │   │   ├── 🆕 _fetch_ohlcv_rest()
│   │   │   │   │   ├── For each exchange client
│   │   │   │   │   ├── Try fetch_ohlcv()
│   │   │   │   │   ├── Record latency
│   │   │   │   │   └── Return DataFrame
│   │   │   │   │
│   │   │   │   ├── 🆕 _record_ws_fetch()
│   │   │   │   │   ├── Update ws_stats
│   │   │   │   │   ├── Calculate avg latency
│   │   │   │   │   └── Calculate success rate
│   │   │   │   │
│   │   │   │   ├── 🆕 _record_rest_fetch()
│   │   │   │   │   ├── Update rest stats
│   │   │   │   │   └── Calculate avg latency
│   │   │   │   │
│   │   │   │   ├── 🆕 get_websocket_stats()
│   │   │   │   │   ├── Return ws_stats copy
│   │   │   │   │   ├── Add connection_health
│   │   │   │   │   ├── Calculate usage_ratio
│   │   │   │   │   └── Calculate latency_improvement
│   │   │   │   │
│   │   │   │   └── shutdown()
│   │   │   │       ├── Stop all loops
│   │   │   │       ├── Close positions
│   │   │   │       └── Disconnect
│   │   │
│   │   ├── websocket_manager.py 🔌 REAL-TIME DATA
│   │   │   │
│   │   │   ├── CLASS: WebSocketManager
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Initialize ccxt.pro exchange
│   │   │   │   │   ├── Setup data_cache
│   │   │   │   │   ├── Setup streams dict
│   │   │   │   │   ├── 🆕 Setup health tracking
│   │   │   │   │   │   ├── start_time
│   │   │   │   │   │   ├── message_count
│   │   │   │   │   │   ├── last_message_time
│   │   │   │   │   │   └── reconnection_count
│   │   │   │   │
│   │   │   │   ├── start_ohlcv_stream()
│   │   │   │   │   ├── Connect WebSocket
│   │   │   │   │   ├── Subscribe to kline
│   │   │   │   │   ├── Register callback
│   │   │   │   │   └── Start listening
│   │   │   │   │
│   │   │   │   ├── _handle_ohlcv_message()
│   │   │   │   │   ├── Parse kline data
│   │   │   │   │   ├── Convert to OHLCV
│   │   │   │   │   ├── Update cache
│   │   │   │   │   ├── Increment message_count 🆕
│   │   │   │   │   └── Update last_message_time 🆕
│   │   │   │   │
│   │   │   │   ├── get_latest_data()
│   │   │   │   │   ├── Lookup cache by symbol/tf
│   │   │   │   │   └── Return OHLCV data
│   │   │   │   │
│   │   │   │   ├── 🆕 get_connection_health()
│   │   │   │   │   ├── Calculate uptime
│   │   │   │   │   ├── Determine status
│   │   │   │   │   │   ├── disconnected (no streams)
│   │   │   │   │   │   ├── connecting (0 messages)
│   │   │   │   │   │   ├── warming_up (<10 messages)
│   │   │   │   │   │   └── healthy (>=10 messages)
│   │   │   │   │   └── Return health dict
│   │   │   │   │
│   │   │   │   ├── 🆕 is_data_fresh()
│   │   │   │   │   ├── Get latest data
│   │   │   │   │   ├── Check last candle timestamp
│   │   │   │   │   ├── Calculate age in seconds
│   │   │   │   │   └── Return age < max_age_seconds
│   │   │   │   │
│   │   │   │   ├── 🆕 get_data_quality_score()
│   │   │   │   │   ├── Freshness score (40%)
│   │   │   │   │   ├── Completeness score (30%)
│   │   │   │   │   ├── Update frequency score (30%)
│   │   │   │   │   └── Return total score (0-100)
│   │   │   │   │
│   │   │   │   └── stop_all_streams()
│   │   │   │       ├── Disconnect all WebSockets
│   │   │   │       └── Clear cache
│   │   │
│   │   ├── position_manager.py 📊 POSITION MANAGEMENT
│   │   │   │
│   │   │   ├── CLASS: AdvancedPositionManager
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Initialize positions dict
│   │   │   │   │   ├── Initialize recent_signals list 🆕
│   │   │   │   │   └── Set signal_expiry_minutes 🆕
│   │   │   │   │
│   │   │   │   ├── add_position()
│   │   │   │   │   ├── Generate position_id
│   │   │   │   │   ├── Store in positions dict
│   │   │   │   │   ├── Add to recent_signals 🆕
│   │   │   │   │   └── Log creation
│   │   │   │   │
│   │   │   │   ├── 🆕 is_duplicate_signal()
│   │   │   │   │   ├── Clean expired signals
│   │   │   │   │   ├── For each recent signal:
│   │   │   │   │   │   ├── Check symbol match
│   │   │   │   │   │   ├── Check side match
│   │   │   │   │   │   ├── Check timeframe match
│   │   │   │   │   │   └── Check entry price (±0.1%)
│   │   │   │   │   ├── If duplicate found:
│   │   │   │   │   │   └── Log warning
│   │   │   │   │   └── Return True/False
│   │   │   │   │
│   │   │   │   ├── manage_position_exits()
│   │   │   │   │   ├── Get position data
│   │   │   │   │   ├── Check STOP LOSS
│   │   │   │   │   │   ├── Long: current <= SL
│   │   │   │   │   │   ├── Short: current >= SL
│   │   │   │   │   │   └── 🆕 Enhanced logging
│   │   │   │   │   ├── Check TAKE PROFIT
│   │   │   │   │   │   ├── Long: current >= TP
│   │   │   │   │   │   ├── Short: current <= TP
│   │   │   │   │   │   └── 🆕 Enhanced logging
│   │   │   │   │   ├── 🆕 Check TRAILING STOP
│   │   │   │   │   │   ├── Track highest_price (long)
│   │   │   │   │   │   ├── Track lowest_price (short)
│   │   │   │   │   │   ├── Calculate trailing_stop_level
│   │   │   │   │   │   ├── Check if triggered
│   │   │   │   │   │   └── Log trailing stop hit
│   │   │   │   │   └── Return exit_check dict
│   │   │   │   │
│   │   │   │   ├── get_active_positions()
│   │   │   │   │   └── Return positions.values()
│   │   │   │   │
│   │   │   │   └── close_position()
│   │   │   │       ├── Remove from positions
│   │   │   │       └── Log closure
│   │   │
│   │   ├── portfolio_manager.py 💰 PORTFOLIO TRACKING
│   │   │   │
│   │   │   ├── CLASS: PortfolioManager
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Set initial_balance
│   │   │   │   │   ├── Set current_balance
│   │   │   │   │   ├── 🆕 Set max_capital_pct (80%)
│   │   │   │   │   ├── 🆕 Calculate max_capital
│   │   │   │   │   └── Initialize positions tracking
│   │   │   │   │
│   │   │   │   ├── 🆕 check_capital_limit()
│   │   │   │   │   ├── Sum capital_used from positions
│   │   │   │   │   ├── Calculate total_capital_used
│   │   │   │   │   ├── Check vs max_capital
│   │   │   │   │   ├── If exceeded:
│   │   │   │   │   │   └── Log capital limit warning
│   │   │   │   │   └── Return can_open_position
│   │   │   │   │
│   │   │   │   ├── update_balance()
│   │   │   │   │   ├── Add/subtract from balance
│   │   │   │   │   └── Log update
│   │   │   │   │
│   │   │   │   ├── get_available_balance()
│   │   │   │   │   └── Return current_balance
│   │   │   │   │
│   │   │   │   └── get_portfolio_summary()
│   │   │   │       ├── Total value
│   │   │   │       ├── Total P&L
│   │   │   │       ├── 🆕 Capital utilization
│   │   │   │       └── Position count
│   │   │
│   │   ├── risk_manager.py ⚠️ RISK MANAGEMENT
│   │   │   │
│   │   │   ├── CLASS: RiskManager
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Set risk_per_trade (1%)
│   │   │   │   │   ├── Set max_position_size
│   │   │   │   │   └── Set min_position_size
│   │   │   │   │
│   │   │   │   ├── calculate_position_size()
│   │   │   │   │   ├── risk_amount = balance * 0.01
│   │   │   │   │   ├── price_distance = |entry - SL|
│   │   │   │   │   ├── position_size = risk / distance
│   │   │   │   │   ├── Apply min/max limits
│   │   │   │   │   └── Return size
│   │   │   │   │
│   │   │   │   └── validate_trade()
│   │   │   │       ├── Check position size
│   │   │   │       ├── Check risk/reward ratio
│   │   │   │       └── Return valid/invalid
│   │   │
│   │   ├── signal_generator.py 📈 STRATEGY LOGIC
│   │   │   │
│   │   │   ├── CLASS: SignalGenerator
│   │   │   │   ├── calculate_indicators()
│   │   │   │   │   ├── EMA calculations
│   │   │   │   │   ├── Volume analysis
│   │   │   │   │   └── Price patterns
│   │   │   │   │
│   │   │   │   └── generate_signal()
│   │   │   │       ├── Analyze indicators
│   │   │   │       ├── Identify entry conditions
│   │   │   │       ├── Calculate SL/TP levels
│   │   │   │       └── Return signal dict
│   │   │
│   │   └── exchange_manager.py 🔌 EXCHANGE API
│   │       │
│   │       ├── CLASS: ExchangeManager
│   │       │   ├── __init__()
│   │       │   │   ├── Load API credentials
│   │       │   │   ├── Initialize ccxt exchange
│   │       │   │   └── Set testnet if paper_mode
│   │       │   │
│   │       │   ├── fetch_ohlcv()
│   │       │   │   ├── Call exchange.fetch_ohlcv()
│   │       │   │   └── Return data
│   │       │   │
│   │       │   ├── place_order()
│   │       │   │   ├── Create order params
│   │       │   │   ├── Submit to exchange
│   │       │   │   └── Return order result
│   │       │   │
│   │       │   └── get_account_balance()
│   │       │       ├── Fetch from exchange
│   │       │       └── Return balance
│   │
│   └── utils/
│       ├── logger.py 📝 LOGGING SYSTEM
│       │   ├── Setup logging config
│       │   ├── Console handler
│       │   ├── File handler
│       │   └── 🆕 Startup headers formatter
│       │
│       └── helpers.py 🛠️ UTILITY FUNCTIONS
│           ├── Format price
│           ├── Calculate percentage
│           └── Time conversions
│
├── tests/ 🧪 TEST SUITE
│   ├── test_position_exit_simple.py
│   │   ├── test_stop_loss_trigger
│   │   ├── test_take_profit_trigger
│   │   └── test_basic_monitoring
│   │
│   ├── 🆕 test_position_monitoring_phase2.py
│   │   ├── test_short_position_stop_loss
│   │   └── test_trailing_stop_activation
│   │
│   └── 🆕 test_websocket_integration_full.py
│       ├── test_websocket_priority_when_available
│       ├── test_rest_fallback_when_unavailable
│       ├── test_statistics_tracking
│       ├── test_data_freshness_check
│       ├── test_connection_health_monitoring
│       ├── test_connection_health_disconnected
│       └── test_data_quality_score
│
├── logs/ 📁 LOG FILES
│   ├── trading_YYYY-MM-DD.log
│   └── error_YYYY-MM-DD.log
│
└── data/ 📊 DATA STORAGE
    ├── positions.json
    ├── signals.json
    └── portfolio.json
```

---

## 🔄 **DATA FLOW DIAGRAM**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │   Bybit Exchange    │
                        │   (WebSocket API)   │
                        └─────────────────────┘
                                  │
                                  │ Real-time
                                  │ kline data
                                  ▼
                        ┌─────────────────────┐
                        │  WebSocketManager   │
                        │  • Receive messages │
                        │  • Parse data       │
                        │  • Cache OHLCV      │
                        │  • Track health 🆕  │
                        └─────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Strategy Loop    │      │ Monitoring Loop  │
          │ (Every 5s)       │      │ (Every 10s)      │
          └──────────────────┘      └──────────────────┘
                    │                           │
                    │ WS Priority 🆕            │ WS Priority
                    ▼                           ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Get OHLCV Data   │      │ Get Current Price│
          │ • Try WS first   │      │ • Try WS first   │
          │ • Fallback REST  │      │ • Fallback REST  │
          │ • Record metrics │      │ • Update position│
          └──────────────────┘      └──────────────────┘
                    │                           │
                    ▼                           ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ SignalGenerator  │      │ Calculate P&L    │
          │ • Calculate EMA  │      │ • Track profit   │
          │ • Volume check   │      │ • Enhanced log 🆕│
          │ • Pattern detect │      └──────────────────┘
          └──────────────────┘                  │
                    │                           ▼
                    ▼                  ┌──────────────────┐
          ┌──────────────────┐         │ Exit Manager     │
          │ Generate Signal  │         │ • Check SL/TP    │
          │ {action, price,  │         │ • Trailing stop🆕│
          │  SL, TP}         │         │ • Execute exit   │
          └──────────────────┘         └──────────────────┘
                    │                           │
                    ▼                           │
          ┌──────────────────┐                  │
          │ Duplicate Check🆕│                  │
          │ • Recent signals │                  │
          │ • 5 min expiry   │                  │
          └──────────────────┘                  │
                    │                           │
                    ▼                           │
          ┌──────────────────┐                  │
          │ Capital Check 🆕 │                  │
          │ • 80% max limit  │                  │
          │ • Available cap  │                  │
          └──────────────────┘                  │
                    │                           │
                    ▼                           │
          ┌──────────────────┐                  │
          │ RiskManager      │                  │
          │ • 1% risk/trade  │                  │
          │ • Position size  │                  │
          └──────────────────┘                  │
                    │                           │
                    ▼                           │
          ┌──────────────────┐                  │
          │ Execute Trade    │                  │
          │ • Paper/Live     │                  │
          │ • Create position│                  │
          └──────────────────┘                  │
                    │                           │
                    ├───────────────────────────┘
                    ▼
          ┌──────────────────┐
          │ PositionManager  │
          │ • Active positions│
          │ • Track signals 🆕│
          │ • Monitor exits  │
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ PortfolioManager │
          │ • Update balance │
          │ • Track capital🆕│
          │ • Calculate P&L  │
          └──────────────────┘
```

---

## 📊 **STATISTICS & MONITORING**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      MONITORING DASHBOARD (Conceptual)                    │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ WEBSOCKET STATISTICS (get_websocket_stats()) 🆕                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Connection Health:                                                       │
│   • Status: healthy                                                      │
│   • Active Streams: 4                                                    │
│   • Total Messages: 1,247                                                │
│   • Uptime: 2h 34m                                                       │
│   • Reconnections: 0                                                     │
│                                                                          │
│ Data Fetching:                                                           │
│   • WebSocket Fetches: 523                                               │
│   • REST Fetches: 12                                                     │
│   • WebSocket Usage Ratio: 97.8%                                         │
│   • WebSocket Success Rate: 99.2%                                        │
│                                                                          │
│ Performance:                                                             │
│   • Avg Latency (WS): 18.3ms                                             │
│   • Avg Latency (REST): 234.7ms                                          │
│   • Latency Improvement: 92.2%                                           │
│                                                                          │
│ Data Quality (per symbol):                                               │
│   • BTC/USDT:USDT 1m: 95/100                                             │
│   • ETH/USDT:USDT 1m: 92/100                                             │
│   • SOL/USDT:USDT 5m: 88/100                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PORTFOLIO STATISTICS 🆕                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Capital Management:                                                      │
│   • Initial Balance: $10,000.00                                          │
│   • Current Balance: $10,245.50                                          │
│   • Total P&L: +$245.50 (+2.45%)                                         │
│                                                                          │
│ Capital Utilization:                                                     │
│   • Max Capital: $8,000.00 (80% limit)                                   │
│   • Used Capital: $5,200.00                                              │
│   • Available: $2,800.00                                                 │
│   • Utilization: 65%                                                     │
│                                                                          │
│ Positions:                                                               │
│   • Active: 3                                                            │
│   • Total Opened: 47                                                     │
│   • Total Closed: 44                                                     │
│   • Win Rate: 63.6%                                                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ACTIVE POSITIONS 🆕                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Position 1:                                                              │
│   • Symbol: BTC/USDT:USDT                                                │
│   • Side: LONG                                                           │
│   • Entry: $50,000.00                                                    │
│   • Current: $50,850.00                                                  │
│   • Amount: 0.02 BTC                                                     │
│   • Unrealized P&L: +$17.00 (+1.70%)                                     │
│   • Stop Loss: $49,000.00                                                │
│   • Take Profit: $55,000.00                                              │
│   • Trailing Stop: ACTIVE (highest: $50,900)                             │
│                                                                          │
│ Position 2:                                                              │
│   • Symbol: ETH/USDT:USDT                                                │
│   • Side: SHORT                                                          │
│   • Entry: $2,800.00                                                     │
│   • Current: $2,750.00                                                   │
│   • Amount: 1.5 ETH                                                      │
│   • Unrealized P&L: +$75.00 (+1.79%)                                     │
│   • Stop Loss: $2,900.00                                                 │
│   • Take Profit: $2,500.00                                               │
│   • Trailing Stop: ACTIVE (lowest: $2,740)                               │
│                                                                          │
│ Position 3:                                                              │
│   • Symbol: SOL/USDT:USDT                                                │
│   • Side: LONG                                                           │
│   • Entry: $145.00                                                       │
│   • Current: $147.50                                                     │
│   • Amount: 10 SOL                                                       │
│   • Unrealized P&L: +$25.00 (+1.72%)                                     │
│   • Stop Loss: $142.00                                                   │
│   • Take Profit: $155.00                                                 │
│   • Trailing Stop: INACTIVE                                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ RECENT SIGNALS (Duplicate Prevention) 🆕                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Last 5 minutes:                                                          │
│   • BTC/USDT:USDT LONG @ $50,000 (2m ago) ✅ EXECUTED                    │
│   • ETH/USDT:USDT SHORT @ $2,800 (4m ago) ✅ EXECUTED                    │
│   • BTC/USDT:USDT LONG @ $50,020 (1m ago) ❌ DUPLICATE (ignored)        │
│   • SOL/USDT:USDT LONG @ $145 (3m ago) ✅ EXECUTED                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 **KEY FEATURES SUMMARY**

### **✅ Implemented Features**

**1. WebSocket Integration (PR #125) 🆕**
- Priority data fetching (WebSocket → REST fallback)
- Health monitoring (status, uptime, messages)
- Data freshness validation (<60s)
- Quality scoring (0-100)
- Performance metrics (latency, success rate)

**2. Position Monitoring (PR #123, #124) 🆕**
- Real-time P&L tracking
- Stop loss detection
- Take profit detection
- Trailing stop support
- Enhanced logging with emojis
- Summary statistics

**3. Duplicate Prevention (PR #122) 🆕**
- Signal deduplication (5-minute window)
- Entry price tolerance (±0.1%)
- Same symbol/side/timeframe check
- Recent signals cache

**4. Portfolio Capital Limit (PR #120) 🆕**
- 80% max capital utilization
- Dynamic capital tracking
- Position sizing limits
- Capital availability check

**5. Professional Logging (PR #121) 🆕**
- Startup banner with headers
- Emoji-based indicators
- Multi-line formatted output
- Structured log messages

**6. Core Trading Logic**
- Signal generation (EMA, volume, patterns)
- Risk management (1% per trade)
- Position sizing
- Trade execution (paper/live)

---

## 📝 **LOG OUTPUT EXAMPLES**

### **Startup Logs:**

```
════════════════════════════════════════════════════════════════════════════
                    BEARISH ALPHA TRADING BOT v1.0                          
════════════════════════════════════════════════════════════════════════════
🕐 Start Time: 2025-10-20 11:32:27 UTC
👤 User: SefaGH
🔧 Mode: PAPER TRADING
📊 Exchange: Bybit (Testnet)
💰 Initial Balance: $10,000.00
════════════════════════════════════════════════════════════════════════════

[11:32:27] ✅ Exchange connection established
[11:32:27] ✅ Portfolio manager initialized
[11:32:27] ✅ Risk manager initialized (1% risk per trade)
[11:32:27] ✅ Signal generator initialized
[11:32:27] ✅ Position manager initialized
[11:32:28] 🔌 Starting WebSocket streams...
[11:32:28]    └─ BTC/USDT:USDT (1m, 5m, 15m)
[11:32:28]    └─ ETH/USDT:USDT (1m, 5m, 15m)
[11:32:29] ✅ WebSocket streams active (6 streams)
[11:32:29] 🎯 Starting position monitoring loop (interval: 10s)
[11:32:29] 🚀 Starting strategy execution loop (interval: 5s)
[11:32:29] ✅ TRADING BOT RUNNING
```

### **Strategy Execution Logs:**

```
[11:32:34] 📡 [WS-DATA] BTC/USDT:USDT 1m - 100 candles, 14.2ms
[11:32:34] 📈 [SIGNAL-GENERATED] BTC/USDT:USDT
           Action: LONG
           Entry: $50,000.00
           Stop Loss: $49,000.00 (-2.0%)
           Take Profit: $55
