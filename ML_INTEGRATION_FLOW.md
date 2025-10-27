# ML Pipeline Integration Flow

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                         PRODUCTION COORDINATOR                                │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                       │    │
│  │  async def process_symbol(symbol: str):                              │    │
│  │                                                                       │    │
│  │    1. Fetch Market Data (WebSocket)                                  │    │
│  │       ├─ 1m, 5m, 30m, 1h, 4h timeframes                             │    │
│  │       └─ OHLCV + indicators                                          │    │
│  │                                                                       │    │
│  │    2. Generate ML Context ────────────────────────────────────┐      │    │
│  │       │                                                        │      │    │
│  │       ▼                                                        │      │    │
│  │    ┌──────────────────────────────────────┐                  │      │    │
│  │    │  MLStrategyIntegrationManager        │                  │      │    │
│  │    │                                      │                  │      │    │
│  │    │  get_ml_context(symbol, data):      │                  │      │    │
│  │    │    ├─ validate_ml_data() ───────────┼──────┐           │      │    │
│  │    │    │                                 │      │           │      │    │
│  │    │    ├─ RegimePredictor ──────────────┼──┐   │           │      │    │
│  │    │    │   predict_regime_transition()  │  │   │           │      │    │
│  │    │    │   → 'bullish'/'bearish'        │  │   │           │      │    │
│  │    │    │   → confidence: 0.85           │  │   │           │      │    │
│  │    │    │                                 │  │   │           │      │    │
│  │    │    ├─ PriceEngine ───────────────────┼──┼───┼──┐        │      │    │
│  │    │    │   get_price_forecast()         │  │   │  │        │      │    │
│  │    │    │   → direction: 'up'            │  │   │  │        │      │    │
│  │    │    │   → uncertainty: 0.02          │  │   │  │        │      │    │
│  │    │    │                                 │  │   │  │        │      │    │
│  │    │    └─ Calculate Consensus ──────────┼──┼───┼──┼─┐      │      │    │
│  │    │        agreement_score: 0.78        │  │   │  │ │      │      │    │
│  │    │                                      │  │   │  │ │      │      │    │
│  │    │  return MLContext:                  │  │   │  │ │      │      │    │
│  │    │    ├─ is_healthy: True              │  │   │  │ │      │      │    │
│  │    │    ├─ regime: 'bullish' (0.85)      │◄─┘   │  │ │      │      │    │
│  │    │    ├─ price: 'up' (0.75)            │◄─────┘  │ │      │      │    │
│  │    │    └─ consensus: 0.78                │◄────────┘ │      │      │    │
│  │    └──────────────────────────────────────┘◄──────────┘      │      │    │
│  │                     │                                         │      │    │
│  │                     └─────────────────────────────────────────┘      │    │
│  │                     │                                                │    │
│  │    3. Pass to Strategies                                            │    │
│  │       │                                                              │    │
│  │       ▼                                                              │    │
│  │    ┌──────────────────────────────────────────────────────┐         │    │
│  │    │  AdaptiveOversoldBounce / AdaptiveShortTheRip        │         │    │
│  │    │                                                       │         │    │
│  │    │  signal(df_30m, ml_context):                         │         │    │
│  │    │                                                       │         │    │
│  │    │    Base Signal: LONG (RSI oversold)                  │         │    │
│  │    │                                                       │         │    │
│  │    │    ┌─────────────────────────────────────┐           │         │    │
│  │    │    │  ML-AWARE DECISION LOGIC             │           │         │    │
│  │    │    │                                      │           │         │    │
│  │    │    │  IF ml_context.is_healthy:          │           │         │    │
│  │    │    │                                      │           │         │    │
│  │    │    │  VETO CHECK:                         │           │         │    │
│  │    │    │    if regime='bearish' & conf>0.7:  │           │         │    │
│  │    │    │      ❌ REJECT signal                │           │         │    │
│  │    │    │      return None                     │           │         │    │
│  │    │    │                                      │           │         │    │
│  │    │    │  CONFIRM CHECK:                      │           │         │    │
│  │    │    │    if regime='bullish' & conf>0.6:  │           │         │    │
│  │    │    │      ✅ BOOST position +25%          │           │         │    │
│  │    │    │      modifier = 1.25                 │           │         │    │
│  │    │    │                                      │           │         │    │
│  │    │    │  CAUTION CHECK:                      │           │         │    │
│  │    │    │    if consensus < 0.5:              │           │         │    │
│  │    │    │      ⚠️ REDUCE position -25%         │           │         │    │
│  │    │    │      modifier = 0.75                 │           │         │    │
│  │    │    └─────────────────────────────────────┘           │         │    │
│  │    │                                                       │         │    │
│  │    │  return enhanced_signal {                            │         │    │
│  │    │    side: 'buy',                                      │         │    │
│  │    │    position_multiplier: 1.25,  ◄─ ML enhanced!      │         │    │
│  │    │    ml_enhanced: True,                                │         │    │
│  │    │    ml_regime: 'bullish',                             │         │    │
│  │    │    ml_confidence: 0.85                               │         │    │
│  │    │  }                                                    │         │    │
│  │    └──────────────────────────────────────────────────────┘         │    │
│  │                     │                                                │    │
│  │    4. Execute Trade                                                 │    │
│  │       │                                                              │    │
│  └───────┼──────────────────────────────────────────────────────────────┘    │
│          │                                                                   │
└──────────┼───────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         POSITION MANAGER                                      │
│                                                                               │
│  async def close_position(position_id, exit_price, exit_reason):            │
│                                                                               │
│    1. Calculate PnL                                                          │
│       realized_pnl = calculate_realized_pnl(...)                             │
│       return_pct = (pnl / position_value) * 100                             │
│                                                                               │
│    2. RL Feedback Loop ────────────────────────────────────┐                 │
│       │                                                     │                 │
│       ▼                                                     │                 │
│    ┌───────────────────────────────────────┐              │                 │
│    │  Calculate Reward                     │              │                 │
│    │                                        │              │                 │
│    │  reward = return_pct / 10.0           │              │                 │
│    │                                        │              │                 │
│    │  Modifiers:                            │              │                 │
│    │    TP hit    → +0.2 bonus             │              │                 │
│    │    SL hit    → -0.1 penalty           │              │                 │
│    │    Trail hit → +0.1 bonus             │              │                 │
│    │                                        │              │                 │
│    │  reward = clip(reward, -2.0, 2.0)    │              │                 │
│    └───────────────────────────────────────┘              │                 │
│                     │                                      │                 │
│                     ▼                                      │                 │
│    3. Feed to RL Agent ─────────────────────────────────┐ │                 │
│       │                                                  │ │                 │
│       ▼                                                  │ │                 │
│    ┌──────────────────────────────────────┐            │ │                 │
│    │  TradingRLAgent                      │            │ │                 │
│    │                                       │            │ │                 │
│    │  learn_from_experience(              │            │ │                 │
│    │    state=entry_state,                │            │ │                 │
│    │    action='buy',                     │            │ │                 │
│    │    reward=0.85,          ◄───────────┼────────────┘ │                 │
│    │    next_state=exit_state,            │              │                 │
│    │    done=True                         │              │                 │
│    │  )                                    │              │                 │
│    │                                       │              │                 │
│    │  → Updates Q-network                 │              │                 │
│    │  → Stores in experience replay       │              │                 │
│    │  → Returns {loss, q_value}           │              │                 │
│    │                                       │              │                 │
│    │  🧠 Learning from outcomes!          │              │                 │
│    └──────────────────────────────────────┘              │                 │
│                     │                                      │                 │
│    4. Log Metrics  ◄────────────────────────────────────┘                 │
│       🧠 [RL-FEEDBACK] Reward=0.85, Loss=0.023, Q=1.25                       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


                        STARTUP HEALTH CHECKS
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  _ml_preflight_health_check():                                               │
│                                                                               │
│  1. Test Regime Predictor                                                    │
│     ├─ Generate dummy OHLCV data (100 candles)                              │
│     ├─ Call predict_regime_transition()                                      │
│     └─ ✅ Verify prediction returned                                         │
│                                                                               │
│  2. Check ML Integration Manager                                             │
│     ├─ Call get_integration_status()                                         │
│     └─ ✅ Verify active=True                                                 │
│                                                                               │
│  3. Verify RL Agent                                                          │
│     ├─ Check memory buffer exists                                            │
│     ├─ Check epsilon value                                                   │
│     └─ ✅ Verify ready for learning                                          │
│                                                                               │
│  Output:                                                                      │
│    🧠 [ML-HEALTH-CHECK] ✅ All critical ML components are healthy           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Legend

- `─────►` Data/Control flow
- `◄─────` Return value
- `✅` Success condition
- `❌` Rejection/Veto
- `⚠️` Warning/Caution
- `🧠` ML/AI component

## Key Integration Points

1. **Validation Gateway**: Prevents bad data from entering ML pipeline
2. **ML Context**: Unified data structure for all ML predictions
3. **Strategy Enhancement**: ML can veto, confirm, or adjust signals
4. **RL Feedback**: Continuous learning from trade outcomes
5. **Health Checks**: Ensures system reliability on startup
