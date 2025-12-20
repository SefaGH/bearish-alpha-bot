# Azure App Configuration Content-Type Hygiene Audit

## Executive Summary
- **Total Keys Scanned**: 90
- **Complex JSON Objects**: 1
- **Empty Content-Type**: 79
- **Issues/Suggestions**: 79

## Detailed Inventory
| Key | Value Preview | Content-Type | Status | Recommendation |
|---|---|---|---|---|
| `BearishAlphaBot/BINGX_REST_DEBUG` | `1` | `text/plain` | OK | None |
| `BearishAlphaBot/CCXT_TIMEOUT_MS` | `10000` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/DEBUG_MODE` | `false` | `text/plain` | OK | None |
| `BearishAlphaBot/EXCHANGES` | `bingx` | `text/plain` | OK | None |
| `BearishAlphaBot/LOG_LEVEL` | `INFO` | `text/plain` | OK | None |
| `BearishAlphaBot/PYTHONPATH` | `/home/site/wwwroot:/home/site/wwwroot/sr...` | `text/plain` | OK | None |
| `BearishAlphaBot/PYTHONUNBUFFERED` | `1` | `text/plain` | OK | None |
| `BearishAlphaBot/TELEGRAM_CHAT_ID` | `1359128753` | `text/plain` | OK | None |
| `BearishAlphaBot/TICKER_CACHE_TTL_S` | `1.0` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/TICKER_MAX_ATTEMPTS` | `2` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/TICKER_RETRY_BASE_DELAY_S` | `0.4` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/TRADING_DURATION` | `7200` | `text/plain` | OK | None |
| `BearishAlphaBot/TRADING_MODE` | `paper` | `text/plain` | OK | None |
| `BearishAlphaBot/adaptive_strategies.enable` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/adaptive_strategies.monitoring.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/adaptive_strategies.performance.max_position_multiplier` | `2.0` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/adaptive_strategies.performance.min_position_multiplier` | `0.5` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/adaptive_strategies.performance.min_volatility_for_adjustment` | `0.02` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.atr_period` | `14` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.bb_period` | `20` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.macd_fast` | `12` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.macd_slow` | `26` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.momentum_windows` | `5,10,20,50` | `` | POTENTIAL_JSON_ARRAY | Consider converting to JSON Array [...] |
| `BearishAlphaBot/ml.features.rsi_period` | `14` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.features.volatility_windows` | `5,10,20,50` | `` | POTENTIAL_JSON_ARRAY | Consider converting to JSON Array [...] |
| `BearishAlphaBot/ml.gemma.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price.min_confidence` | `0.55` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price_prediction.feature_size` | `42` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.dropout` | `0.6` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.hidden_size` | `64` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.num_layers` | `2` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.price_prediction.timeframes` | `5m,15m` | `` | POTENTIAL_JSON_ARRAY | Consider converting to JSON Array [...] |
| `BearishAlphaBot/ml.regime_prediction.min_confidence_threshold` | `0.6` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.epsilon_inference` | `0.01` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.gamma` | `0.95` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.hold_confidence_threshold` | `0.60` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.learning_rate` | `0.00003` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.ppo_enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.reinforcement_learning.training_mode` | `false` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.signal_scoring.min_score_to_trade` | `62` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.signal_scoring.weights.ml_price` | `0.35` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.signal_scoring.weights.risk_reward` | `0.10` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/ml.signal_scoring.weights.strategy` | `0.35` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/models.active_bundle` | `artifacts/gemma/final` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/pyramiding.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/pyramiding.max_layers_per_symbol` | `2` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/pyramiding.min_scale_in_distance_pct` | `0.003` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/pyramiding.min_scale_in_quality` | `0.65` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/pyramiding.min_scale_in_unrealized_pnl_pct` | `0.003` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.daily_max_trades` | `8` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.equity_usd` | `500` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.max_notional_pct_per_trade` | `0.25` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.max_position_size_pct` | `0.25` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.min_stop_pct` | `0.005` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.per_trade_risk_pct` | `0.003` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.position_size_policy` | `clip` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.queue.max_pending_scale_in_per_symbol` | `1` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.rr_dynamic.base_target_rr` | `1.3` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/risk.size_planner_enabled` | `true` | `text/plain` | OK | None |
| `BearishAlphaBot/signals.bypass.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.bypass.rsi_overbought_threshold` | `88` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.bypass.rsi_oversold_threshold` | `12` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.duplicate_prevention.cooldown_seconds` | `20` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.duplicate_prevention.min_price_change_pct` | `0.0005` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_threshold` | `0.0015` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_base` | `28` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_range` | `8` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.enable` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.min_rr_ratio` | `1.5` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.rsi_max` | `45` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.sl_atr_mult` | `1.0` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.oversold_bounce.tp_atr_mult` | `1.8` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_base` | `72` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_range` | `8` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.enable` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.min_rr_ratio` | `1.5` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.enabled` | `true` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_15m` | `false` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_1h` | `false` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.rsi_min` | `55` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.sl_atr_mult` | `1.0` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | `50` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | `50` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | `50` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/signals.short_the_rip.tp_atr_mult` | `1.8` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/universe.fixed_symbols` | `BTC/USDT:USDT` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |
| `BearishAlphaBot/volume_analyzer.buckets` | `[[0.0, "LOW"], [4.96, "NORMAL"], [6.23, ...` | `application/json` | OK (JSON) | Keep as is |
| `BearishAlphaBot/websocket.max_streams_per_exchange.bingx` | `10` | `` | MISSING_METADATA | Set to text/plain (or application/json if typed) |