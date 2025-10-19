graph TD
    Start([GitHub Actions: live_trading_launcher.yml]) --> Launch[scripts/live_trading_launcher.py]
    
    Launch --> Init{Initialization Phase}
    
    Init --> Config[1. Config Loader<br/>config/config.example.yaml]
    Config --> |3 symbols<br/>BTC, ETH, SOL| Env[2. Environment Setup<br/>BINGX_KEY/SECRET<br/>TELEGRAM credentials]
    
    Env --> Exchange[3. Exchange Init<br/>core/bingx_authenticator.py<br/>core/ccxt_client.py]
    Exchange --> |Test connection<br/>Verify pairs| Risk[4. Risk Manager<br/>core/risk_manager.py<br/>$100 capital]
    
    Risk --> AI[5. Phase 4 AI Components<br/>ml/regime_predictor.py<br/>ml/strategy_optimizer.py<br/>ml/price_predictor.py]
    
    AI --> Strat[6. Strategy Init<br/>strategies/adaptive_ob.py<br/>strategies/adaptive_str.py]
    
    Strat --> Prod[7. Production System<br/>core/production_coordinator.py]
    
    Prod --> WS[8. WebSocket Manager<br/>core/websocket_manager.py<br/>core/websocket_client.py]
    
    WS --> |1m streams<br/>BTC, ETH, SOL| Preflight{Pre-Flight Checks}
    
    Preflight --> |ALL PASSED| Start_Trading[START LIVE TRADING]
    
    Start_Trading --> MainLoop[Main Trading Loop<br/>Duration: 300s]
    
    MainLoop --> Coord[Production Coordinator Loop<br/>Every 30s]
    
    Coord --> FetchData[Fetch OHLCV Data<br/>REST API]
    
    FetchData --> BTC_Data[BTC/USDT:USDT<br/>30m, 1h, 4h<br/>200 candles each]
    FetchData --> ETH_Data[ETH/USDT:USDT<br/>30m, 1h, 4h<br/>200 candles each]
    FetchData --> SOL_Data[SOL/USDT:USDT<br/>30m, 1h, 4h<br/>200 candles each]
    
    BTC_Data --> BTC_Regime{Market Regime<br/>4h Analysis}
    ETH_Data --> ETH_Process[Process ETH]
    SOL_Data --> SOL_Process[Process SOL]
    
    BTC_Regime --> |Bearish trend<br/>Weak momentum<br/>Low volatility| BTC_Strat[Execute Strategies<br/>adaptive_ob<br/>adaptive_str]
    
    ETH_Process --> ETH_Strat[Execute Strategies]
    SOL_Process --> SOL_Strat[Execute Strategies]
    
    BTC_Strat --> |No signal| BTC_Done[BTC: No signals]
    ETH_Strat --> Signal_Check{Check Conditions}
    SOL_Strat --> |No signal| SOL_Done[SOL: No signals]
    
    Signal_Check --> |adaptive_str<br/>RSI >= 62.0| Signal_Gen[SIGNAL GENERATED!<br/>strategies/adaptive_str.py]
    
    Signal_Gen --> Signal_Data[Signal Details:<br/>- Side: SELL<br/>- Entry: $3905.47<br/>- Stop: $3923.03<br/>- Target: $3874.22<br/>- RSI: 62.8]
    
    Signal_Data --> Strategy_Coord[StrategyCoordinator<br/>core/strategy_coordinator.py]
    
    Strategy_Coord --> Conflict{Signal Conflict?}
    
    Conflict --> |Yes: same_direction| Resolve[Conflict Resolution<br/>highest_priority strategy]
    Conflict --> |No| Risk_Check[Risk Validation]
    
    Resolve --> Risk_Check
    
    Risk_Check --> Risk_Calc[core/risk_manager.py<br/>Calculate position size:<br/>0.0026 ETH = $10<br/>Risk: $0.045]
    
    Risk_Calc --> Risk_Pass{Validation?}
    
    Risk_Pass --> |PASSED| Queue_SC[Add to StrategyCoordinator Queue]
    Risk_Pass --> |FAILED| Reject[Signal Rejected]
    
    Queue_SC --> |Signal ID created| Lifecycle[Signal Lifecycle Tracking]
    
    Lifecycle --> Stage1[STAGE: GENERATED ✓]
    Stage1 --> Stage2[STAGE: VALIDATED ✓]
    Stage2 --> Stage3[STAGE: QUEUED ✓]
    Stage3 --> Stage4[STAGE: FORWARDED ✓]
    
    Stage4 --> LTE_Queue[LiveTradingEngine Queue<br/>core/live_trading_engine.py]
    
    LTE_Queue --> |❌ PROBLEM HERE| Signal_Loop[Signal Processing Loop<br/>_signal_processing_loop]
    
    Signal_Loop --> Loop_Check{Queue Empty?}
    
    Loop_Check --> |Yes| Wait[await asyncio.sleep]
    Loop_Check --> |❌ Should be No<br/>But acts like Yes| No_Execute[NO EXECUTION!]
    
    Wait --> |30s later| Coord
    
    No_Execute --> Problem[🔴 CRITICAL ISSUE:<br/>Signals stuck in queue<br/>Never executed!]
    
    Problem --> Monitor[Queue Monitor<br/>Every 30s]
    
    Monitor --> Status[StrategyCoordinator: 6 signals<br/>LiveTradingEngine: 6 signals<br/>❌ None executed]
    
    Status --> Continue{Duration<br/>Reached?}
    
    Continue --> |No| Coord
    Continue --> |Yes: 300s| Shutdown[Shutdown Sequence]
    
    Shutdown --> Stop_WS[Stop WebSocket Loops<br/>BTC: 555 iterations<br/>ETH: 602 iterations<br/>SOL: 182 iterations]
    
    Stop_WS --> Final_Report[Final Report:<br/>Total signals: 0 ❌<br/>Total trades: 0 ❌<br/>Win rate: 0% ❌]
    
    Final_Report --> End([Workflow Complete])
    
    style Signal_Gen fill:#90EE90
    style Signal_Data fill:#90EE90
    style Queue_SC fill:#90EE90
    style Stage4 fill:#90EE90
    style No_Execute fill:#FF6B6B
    style Problem fill:#FF6B6B
    style Status fill:#FF6B6B
    style Final_Report fill:#FF6B6B
    style Signal_Loop fill:#FFD700
    style Loop_Check fill:#FFD700
    style LTE_Queue fill:#FFD700
