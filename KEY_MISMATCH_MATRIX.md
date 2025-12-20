# KEY MISMATCH MATRIX (RESOLVED)

## 1. MISPLACED / Alias Keys (AppConfig -> Canonical YAML)

**Status: CLEARED**
All "Env-style" keys (e.g., `ML_PRED_TIMEFRAMES`, `VOLUME_ANALYZER_BUCKETS`) have been successfully migrated to their canonical "Dot Notation" equivalents (e.g., `ml.price_prediction.timeframes`, `volume_analyzer.buckets`) and deleted from App Configuration.

## 2. DIFF (Alias exists but Canonical might be different)

**Status: CLEARED**
No alias keys remain.

## 3. MISSING

**Status: CLEARED**
All required canonical keys are now present in App Configuration.

## 4. OK (Correctly Configured)

The following keys are now correctly configured in App Configuration using the canonical dot-notation structure:

### Adaptive Strategies
- `adaptive_strategies.enable`
- `adaptive_strategies.monitoring.enabled`
- `adaptive_strategies.performance.max_position_multiplier`
- `adaptive_strategies.performance.min_position_multiplier`
- `adaptive_strategies.performance.min_volatility_for_adjustment`

### ML (General & Gemma)
- `ml.enabled`
- `ml.gemma.enabled`
- `models.active_bundle`

### ML (Price Prediction)
- `ml.price.min_confidence`
- `ml.price_prediction.feature_size`
- `ml.price_prediction.timeframes`
- `ml.price_prediction.model_params.lstm.dropout`
- `ml.price_prediction.model_params.lstm.hidden_size`
- `ml.price_prediction.model_params.lstm.num_layers`
- `ml.regime_prediction.min_confidence_threshold`

### ML (Features)
- `ml.features.atr_period`
- `ml.features.bb_period`
- `ml.features.macd_fast`
- `ml.features.macd_slow`
- `ml.features.momentum_windows`
- `ml.features.rsi_period`
- `ml.features.volatility_windows`

### ML (Reinforcement Learning)
- `ml.reinforcement_learning.epsilon_inference`
- `ml.reinforcement_learning.gamma`
- `ml.reinforcement_learning.hold_confidence_threshold`
- `ml.reinforcement_learning.learning_rate`
- `ml.reinforcement_learning.ppo_enabled`
- `ml.reinforcement_learning.training_mode`

### ML (Scoring)
- `ml.signal_scoring.min_score_to_trade`
- `ml.signal_scoring.weights.ml_price`
- `ml.signal_scoring.weights.risk_reward`
- `ml.signal_scoring.weights.strategy`

### Risk Management
- `risk.daily_max_trades`
- `risk.equity_usd`
- `risk.max_notional_pct_per_trade`
- `risk.max_position_size_pct`
- `risk.min_stop_pct`
- `risk.per_trade_risk_pct`
- `risk.position_size_policy`
- `risk.queue.max_pending_scale_in_per_symbol`
- `risk.rr_dynamic.base_target_rr`
- `risk.size_planner_enabled`

### Signals & Strategies (General)
- `signals.bypass.enabled`
- `signals.bypass.rsi_overbought_threshold`
- `signals.bypass.rsi_oversold_threshold`
- `signals.duplicate_prevention.cooldown_seconds`
- `signals.duplicate_prevention.min_price_change_pct`
- `signals.duplicate_prevention.price_delta_bypass_enabled`
- `signals.duplicate_prevention.price_delta_bypass_threshold`

### Signals (Oversold Bounce)
- `signals.oversold_bounce.adaptive_rsi_base`
- `signals.oversold_bounce.adaptive_rsi_range`
- `signals.oversold_bounce.enable`
- `signals.oversold_bounce.min_rr_ratio`
- `signals.oversold_bounce.rsi_max`
- `signals.oversold_bounce.sl_atr_mult`
- `signals.oversold_bounce.tp_atr_mult`

### Signals (Short The Rip)
- `signals.short_the_rip.adaptive_rsi_base`
- `signals.short_the_rip.adaptive_rsi_range`
- `signals.short_the_rip.enable`
- `signals.short_the_rip.min_rr_ratio`
- `signals.short_the_rip.mtf_confirmation.enabled`
- `signals.short_the_rip.mtf_confirmation.require_15m`
- `signals.short_the_rip.mtf_confirmation.require_1h`
- `signals.short_the_rip.rsi_min`
- `signals.short_the_rip.sl_atr_mult`
- `signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold`
- `signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold`
- `signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold`
- `signals.short_the_rip.tp_atr_mult`

### Universe & Websocket
- `universe.fixed_symbols`
- `websocket.max_streams_per_exchange.bingx`

### Volume Analyzer
- `volume_analyzer.buckets`

### Pyramiding
- `pyramiding.enabled`
- `pyramiding.max_layers_per_symbol`
- `pyramiding.min_scale_in_distance_pct`
- `pyramiding.min_scale_in_quality`
- `pyramiding.min_scale_in_unrealized_pnl_pct`

## 5. System / Environment Variables (Retained)

The following keys are retained in App Configuration as they represent system-level or environment-specific configurations that are typically injected as environment variables:

- `BINGX_REST_DEBUG`
- `CCXT_TIMEOUT_MS`
- `DEBUG_MODE`
- `EXCHANGES`
- `LOG_LEVEL`
- `PYTHONPATH`
- `PYTHONUNBUFFERED`
- `TELEGRAM_CHAT_ID`
- `TICKER_CACHE_TTL_S`
- `TICKER_MAX_ATTEMPTS`
- `TICKER_RETRY_BASE_DELAY_S`
- `TRADING_DURATION`
- `TRADING_MODE`
- `DOCKER_IMAGE_TAG`
