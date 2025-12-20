## Executive Summary
- Canonical schema extracted from `config/config.example.yaml` (460 dot-paths; env mappings embedded in Section 1 table with line refs).
- AppConfig snapshot uses 90 keys from `APPCONFIG_CONTENTTYPE_JSON_AUDIT.md` (lines 12-101).
- Alignment results: MATCH 74, ALIAS 3 (symbol-case), UNKNOWN 13 (system/env keys); see Section 3 tables.
- Legacy/env-style consumption still present via direct env reads (e.g., TRADING_MODE, TRADING_SYMBOLS) while `ml_rl_training_mode` has no code hits; evidence in Section 4.
- AppConfig load/flatten has no casting (load/flatten: `src/config/live_trading_config.py:722,815`; env casting only: `src/config/live_trading_config.py:251,399`).
- bool/truthy checks remain for MTF and pyramiding (`src/core/production_coordinator.py:406`, `src/strategies/adaptive_str.py:422`, `src/core/risk_manager.py:373`).
- Symbol-specific AppConfig keys lowercased by `_flatten_to_nested` can miss canonical YAML symbol casing (see ALIAS entries + `src/config/live_trading_config.py:815`).
- Debug mode is DISABLED in a recent run (`live_trading_latest.log:439`).
- RL training_mode forced to False in paper mode (`logs/live_trading_20251117_022805_680296.log:362-365`).
- Pyramiding settings logged as enabled in live config (`logs/live_trading_20251218_110812_154401.log:40-44`).
- per_trade_risk_pct normalized from env in recent run (`live_trading_latest.log:7,17`).
- No MTF-specific log lines found in current *.log set (see Section 5 note).
- Allowlist-first central coercion recommended for high-risk bool/list/numeric paths (Section 6).

## 1. Canonical YAML Key-Path & Tip Şeması
- Note: Dict/list default values are truncated in this table for readability; leaf keys carry full defaults.
| Dot Path | Default Value | Type | Override Env | Source |
| --- | --- | --- | --- | --- |
| `execution` | {'enable_live': True, 'order_type': 'market', 'time_in_force': 'IOC', 'fee_pct': 0.0006, 'max_slippage_pct': 0.001, 'leverage': {'default': 5, 'overrides': {... | dict |  | config/config.example.yaml:38 |
| `execution.enable_live` | True | bool | `ENABLE_LIVE_TRADING` | config/config.example.yaml:39 |
| `execution.order_type` | 'market' | str | `ORDER_TYPE` | config/config.example.yaml:40 |
| `execution.time_in_force` | 'IOC' | str | `TIME_IN_FORCE` | config/config.example.yaml:41 |
| `execution.fee_pct` | 0.0006 | float | `FEE_PCT` | config/config.example.yaml:42 |
| `execution.max_slippage_pct` | 0.001 | float | `MAX_SLIPPAGE_PCT` | config/config.example.yaml:43 |
| `execution.leverage` | {'default': 5, 'overrides': {'BTC/USDT:USDT': 10}} | dict |  | config/config.example.yaml:44 |
| `execution.leverage.default` | 5 | int | `LEVERAGE_DEFAULT` | config/config.example.yaml:45 |
| `execution.leverage.overrides` | {'BTC/USDT:USDT': 10} | dict |  | config/config.example.yaml:46 |
| `execution.leverage.overrides.BTC/USDT:USDT` | 10 | int | `LEVERAGE_BTC_USDT` | config/config.example.yaml:47 |
| `risk` | {'size_planner_enabled': True, 'equity_usd': 500, 'per_trade_risk_pct': 0.003, 'daily_loss_limit_pct': 0.02, 'risk_usd_cap': 1, 'max_position_size_pct': 0.25... | dict |  | config/config.example.yaml:52 |
| `risk.size_planner_enabled` | True | bool | `RISK_SIZE_PLANNER_ENABLED` | config/config.example.yaml:56 |
| `risk.equity_usd` | 500 | int | `CAPITAL_USDT` | config/config.example.yaml:59 |
| `risk.per_trade_risk_pct` | 0.003 | float | `PER_TRADE_RISK_PCT` | config/config.example.yaml:60 |
| `risk.daily_loss_limit_pct` | 0.02 | float | `DAILY_LOSS_LIMIT_PCT` | config/config.example.yaml:61 |
| `risk.risk_usd_cap` | 1 | int | `RISK_USD_CAP` | config/config.example.yaml:62 |
| `risk.max_position_size_pct` | 0.25 | float | `MAX_POSITION_SIZE_PCT` | config/config.example.yaml:65 |
| `risk.max_notional_pct_per_trade` | 0.25 | float | `MAX_NOTIONAL_PCT_PER_TRADE` | config/config.example.yaml:66 |
| `risk.max_margin_pct_per_trade` | 0.3 | float | `MAX_MARGIN_PCT_PER_TRADE` | config/config.example.yaml:67 |
| `risk.position_size_policy` | 'clip' | str | `POSITION_SIZE_POLICY` | config/config.example.yaml:68 |
| `risk.min_position_size_usd` | 10 | int | `MIN_POSITION_SIZE_USD` | config/config.example.yaml:71 |
| `risk.min_stop_pct` | 0.005 | float | `MIN_STOP_PCT` | config/config.example.yaml:72 |
| `risk.min_amount_behavior` | 'skip' | str | `MIN_AMOUNT_BEHAVIOR` | config/config.example.yaml:77 |
| `risk.min_notional_behavior` | 'skip' | str | `MIN_NOTIONAL_BEHAVIOR` | config/config.example.yaml:78 |
| `risk.daily_max_trades` | 8 | int | `DAILY_MAX_TRADES` | config/config.example.yaml:79 |
| `risk.rr_dynamic` | {'enabled': True, 'base_target_rr': 2.0, 'lower_bound_rr': 1.2, 'upper_bound_rr': 3.0, 'weights': {'ml_confidence': 0.35, 'rl_agreement': 0.35, 'regime_clari... | dict |  | config/config.example.yaml:82 |
| `risk.rr_dynamic.enabled` | True | bool | `RR_DYNAMIC_ENABLED` | config/config.example.yaml:83 |
| `risk.rr_dynamic.base_target_rr` | 2.0 | float | `RR_BASE_TARGET` | config/config.example.yaml:86 |
| `risk.rr_dynamic.lower_bound_rr` | 1.2 | float | `RR_LOWER_BOUND` | config/config.example.yaml:87 |
| `risk.rr_dynamic.upper_bound_rr` | 3.0 | float | `RR_UPPER_BOUND` | config/config.example.yaml:88 |
| `risk.rr_dynamic.weights` | {'ml_confidence': 0.35, 'rl_agreement': 0.35, 'regime_clarity': 0.2, 'volume_strength': 0.05, 'momentum_strength': 0.05} | dict |  | config/config.example.yaml:91 |
| `risk.rr_dynamic.weights.ml_confidence` | 0.35 | float | `RR_WEIGHT_ML` | config/config.example.yaml:92 |
| `risk.rr_dynamic.weights.rl_agreement` | 0.35 | float | `RR_WEIGHT_RL` | config/config.example.yaml:93 |
| `risk.rr_dynamic.weights.regime_clarity` | 0.2 | float | `RR_WEIGHT_REGIME` | config/config.example.yaml:94 |
| `risk.rr_dynamic.weights.volume_strength` | 0.05 | float | `RR_WEIGHT_VOLUME` | config/config.example.yaml:95 |
| `risk.rr_dynamic.weights.momentum_strength` | 0.05 | float | `RR_WEIGHT_MOMENTUM` | config/config.example.yaml:96 |
| `risk.rr_dynamic.fallback` | {'missing_ml_default': 0.5, 'missing_rl_default': 0.5, 'missing_regime_default': 0.3} | dict |  | config/config.example.yaml:99 |
| `risk.rr_dynamic.fallback.missing_ml_default` | 0.5 | float | `RR_FALLBACK_ML` | config/config.example.yaml:100 |
| `risk.rr_dynamic.fallback.missing_rl_default` | 0.5 | float | `RR_FALLBACK_RL` | config/config.example.yaml:101 |
| `risk.rr_dynamic.fallback.missing_regime_default` | 0.3 | float | `RR_FALLBACK_REGIME` | config/config.example.yaml:102 |
| `risk.rr_dynamic.regime_multipliers` | {'bullish': 0.9, 'bearish': 0.9, 'neutral': 1.0, 'volatile': 1.2} | dict |  | config/config.example.yaml:105 |
| `risk.rr_dynamic.regime_multipliers.bullish` | 0.9 | float | `RR_MULT_BULLISH` | config/config.example.yaml:106 |
| `risk.rr_dynamic.regime_multipliers.bearish` | 0.9 | float | `RR_MULT_BEARISH` | config/config.example.yaml:107 |
| `risk.rr_dynamic.regime_multipliers.neutral` | 1.0 | float | `RR_MULT_NEUTRAL` | config/config.example.yaml:108 |
| `risk.rr_dynamic.regime_multipliers.volatile` | 1.2 | float | `RR_MULT_VOLATILE` | config/config.example.yaml:109 |
| `risk.rr_dynamic.strategy_overrides` | {'scalper': {'base_target_rr': 1.5, 'lower_bound_rr': 1.0, 'weights': {'ml_confidence': 0.25, 'regime_clarity': 0.1}}, 'mean_reversion': {'upper_bound_rr': 2... | dict |  | config/config.example.yaml:112 |
| `risk.rr_dynamic.strategy_overrides.scalper` | {'base_target_rr': 1.5, 'lower_bound_rr': 1.0, 'weights': {'ml_confidence': 0.25, 'regime_clarity': 0.1}} | dict |  | config/config.example.yaml:113 |
| `risk.rr_dynamic.strategy_overrides.scalper.base_target_rr` | 1.5 | float | `RR_SCALPER_BASE_TARGET` | config/config.example.yaml:114 |
| `risk.rr_dynamic.strategy_overrides.scalper.lower_bound_rr` | 1.0 | float | `RR_SCALPER_LOWER_BOUND` | config/config.example.yaml:115 |
| `risk.rr_dynamic.strategy_overrides.scalper.weights` | {'ml_confidence': 0.25, 'regime_clarity': 0.1} | dict |  | config/config.example.yaml:116 |
| `risk.rr_dynamic.strategy_overrides.scalper.weights.ml_confidence` | 0.25 | float | `RR_SCALPER_WEIGHT_ML` | config/config.example.yaml:117 |
| `risk.rr_dynamic.strategy_overrides.scalper.weights.regime_clarity` | 0.1 | float | `RR_SCALPER_WEIGHT_REGIME` | config/config.example.yaml:118 |
| `risk.rr_dynamic.strategy_overrides.mean_reversion` | {'upper_bound_rr': 2.2, 'regime_multipliers': {'neutral': 0.9, 'volatile': 1.3}} | dict |  | config/config.example.yaml:119 |
| `risk.rr_dynamic.strategy_overrides.mean_reversion.upper_bound_rr` | 2.2 | float | `RR_MEANREV_UPPER_BOUND` | config/config.example.yaml:120 |
| `risk.rr_dynamic.strategy_overrides.mean_reversion.regime_multipliers` | {'neutral': 0.9, 'volatile': 1.3} | dict |  | config/config.example.yaml:121 |
| `risk.rr_dynamic.strategy_overrides.mean_reversion.regime_multipliers.neutral` | 0.9 | float | `RR_MEANREV_MULT_NEUTRAL` | config/config.example.yaml:122 |
| `risk.rr_dynamic.strategy_overrides.mean_reversion.regime_multipliers.volatile` | 1.3 | float | `RR_MEANREV_MULT_VOLATILE` | config/config.example.yaml:123 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter` | {'base_target_rr': 2.2, 'lower_bound_rr': 1.4, 'weights': {'ml_confidence': 0.45, 'volume_strength': 0.15}} | dict |  | config/config.example.yaml:124 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter.base_target_rr` | 2.2 | float | `RR_BREAKOUT_BASE_TARGET` | config/config.example.yaml:125 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter.lower_bound_rr` | 1.4 | float | `RR_BREAKOUT_LOWER_BOUND` | config/config.example.yaml:126 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter.weights` | {'ml_confidence': 0.45, 'volume_strength': 0.15} | dict |  | config/config.example.yaml:127 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter.weights.ml_confidence` | 0.45 | float | `RR_BREAKOUT_WEIGHT_ML` | config/config.example.yaml:128 |
| `risk.rr_dynamic.strategy_overrides.breakout_hunter.weights.volume_strength` | 0.15 | float | `RR_BREAKOUT_WEIGHT_VOLUME` | config/config.example.yaml:129 |
| `risk.volume_bucket_risk_matrix` | {'LOW': {'position_size_multiplier': 0.5, 'stop_loss_multiplier': 1.3, 'take_profit_multiplier': 0.9}, 'NORMAL': {'position_size_multiplier': 1.0, 'stop_loss... | dict | `RISK_VOLUME_BUCKET_MATRIX` | config/config.example.yaml:132 |
| `risk.volume_bucket_risk_matrix.LOW` | {'position_size_multiplier': 0.5, 'stop_loss_multiplier': 1.3, 'take_profit_multiplier': 0.9} | dict |  | config/config.example.yaml:133 |
| `risk.volume_bucket_risk_matrix.LOW.position_size_multiplier` | 0.5 | float |  | config/config.example.yaml:134 |
| `risk.volume_bucket_risk_matrix.LOW.stop_loss_multiplier` | 1.3 | float |  | config/config.example.yaml:135 |
| `risk.volume_bucket_risk_matrix.LOW.take_profit_multiplier` | 0.9 | float |  | config/config.example.yaml:136 |
| `risk.volume_bucket_risk_matrix.NORMAL` | {'position_size_multiplier': 1.0, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 1.0} | dict |  | config/config.example.yaml:137 |
| `risk.volume_bucket_risk_matrix.NORMAL.position_size_multiplier` | 1.0 | float |  | config/config.example.yaml:138 |
| `risk.volume_bucket_risk_matrix.NORMAL.stop_loss_multiplier` | 1.0 | float |  | config/config.example.yaml:139 |
| `risk.volume_bucket_risk_matrix.NORMAL.take_profit_multiplier` | 1.0 | float |  | config/config.example.yaml:140 |
| `risk.volume_bucket_risk_matrix.HIGH` | {'position_size_multiplier': 1.2, 'stop_loss_multiplier': 0.9, 'take_profit_multiplier': 1.1} | dict |  | config/config.example.yaml:141 |
| `risk.volume_bucket_risk_matrix.HIGH.position_size_multiplier` | 1.2 | float |  | config/config.example.yaml:142 |
| `risk.volume_bucket_risk_matrix.HIGH.stop_loss_multiplier` | 0.9 | float |  | config/config.example.yaml:143 |
| `risk.volume_bucket_risk_matrix.HIGH.take_profit_multiplier` | 1.1 | float |  | config/config.example.yaml:144 |
| `risk.volume_bucket_risk_matrix.EXTREME` | {'position_size_multiplier': 1.5, 'stop_loss_multiplier': 0.8, 'take_profit_multiplier': 1.3} | dict |  | config/config.example.yaml:145 |
| `risk.volume_bucket_risk_matrix.EXTREME.position_size_multiplier` | 1.5 | float |  | config/config.example.yaml:146 |
| `risk.volume_bucket_risk_matrix.EXTREME.stop_loss_multiplier` | 0.8 | float |  | config/config.example.yaml:147 |
| `risk.volume_bucket_risk_matrix.EXTREME.take_profit_multiplier` | 1.3 | float |  | config/config.example.yaml:148 |
| `risk.queue` | {'ttl_seconds': 60, 'max_queue_depth': 50, 'batch_dequeue': 3, 'max_pending_per_symbol': 1, 'max_pending_scale_in_per_symbol': 2, 'priority_weights': {'expli... | dict |  | config/config.example.yaml:151 |
| `risk.queue.ttl_seconds` | 60 | int | `SIGNAL_QUEUE_TTL_SECONDS` | config/config.example.yaml:152 |
| `risk.queue.max_queue_depth` | 50 | int | `SIGNAL_QUEUE_MAX_DEPTH` | config/config.example.yaml:153 |
| `risk.queue.batch_dequeue` | 3 | int | `SIGNAL_QUEUE_BATCH_DEQUEUE` | config/config.example.yaml:154 |
| `risk.queue.max_pending_per_symbol` | 1 | int | `SIGNAL_QUEUE_MAX_PENDING_PER_SYMBOL` | config/config.example.yaml:155 |
| `risk.queue.max_pending_scale_in_per_symbol` | 2 | int | `SIGNAL_QUEUE_MAX_PENDING_SCALE_IN_PER_SYMBOL` | config/config.example.yaml:156 |
| `risk.queue.priority_weights` | {'explicit_priority': 0.4, 'risk_reward': 0.3, 'ml_confidence': 0.2, 'urgency': 0.1, 'regime_alignment': 0.05, 'strategy_urgency': 0.05} | dict |  | config/config.example.yaml:157 |
| `risk.queue.priority_weights.explicit_priority` | 0.4 | float | `SIGNAL_QUEUE_WEIGHT_PRIORITY` | config/config.example.yaml:158 |
| `risk.queue.priority_weights.risk_reward` | 0.3 | float | `SIGNAL_QUEUE_WEIGHT_RR` | config/config.example.yaml:159 |
| `risk.queue.priority_weights.ml_confidence` | 0.2 | float | `SIGNAL_QUEUE_WEIGHT_ML` | config/config.example.yaml:160 |
| `risk.queue.priority_weights.urgency` | 0.1 | float | `SIGNAL_QUEUE_WEIGHT_URGENCY` | config/config.example.yaml:161 |
| `risk.queue.priority_weights.regime_alignment` | 0.05 | float | `SIGNAL_QUEUE_WEIGHT_REGIME_ALIGNMENT` | config/config.example.yaml:162 |
| `risk.queue.priority_weights.strategy_urgency` | 0.05 | float | `SIGNAL_QUEUE_WEIGHT_STRATEGY_URGENCY` | config/config.example.yaml:163 |
| `risk.concurrent_limits` | {'max_open_positions': 3, 'max_positions_per_symbol': 1, 'max_total_risk_pct': 0.06, 'correlation_bucket_threshold': 0.8, 'dynamic_scaling': {'enabled': True... | dict |  | config/config.example.yaml:166 |
| `risk.concurrent_limits.max_open_positions` | 3 | int | `MAX_OPEN_POSITIONS` | config/config.example.yaml:167 |
| `risk.concurrent_limits.max_positions_per_symbol` | 1 | int | `MAX_POSITIONS_PER_SYMBOL` | config/config.example.yaml:168 |
| `risk.concurrent_limits.max_total_risk_pct` | 0.06 | float | `MAX_TOTAL_RISK_PCT` | config/config.example.yaml:169 |
| `risk.concurrent_limits.correlation_bucket_threshold` | 0.8 | float | `CORRELATION_BUCKET_THRESHOLD` | config/config.example.yaml:170 |
| `risk.concurrent_limits.dynamic_scaling` | {'enabled': True, 'quality_threshold': 0.8, 'min_unrealized_pnl_pct': 0.005, 'max_additional_positions': 2} | dict |  | config/config.example.yaml:171 |
| `risk.concurrent_limits.dynamic_scaling.enabled` | True | bool | `DYNAMIC_SCALING_ENABLED` | config/config.example.yaml:172 |
| `risk.concurrent_limits.dynamic_scaling.quality_threshold` | 0.8 | float | `DYNAMIC_SCALING_QUALITY_THRESHOLD` | config/config.example.yaml:173 |
| `risk.concurrent_limits.dynamic_scaling.min_unrealized_pnl_pct` | 0.005 | float | `DYNAMIC_SCALING_MIN_PNL_PCT` | config/config.example.yaml:174 |
| `risk.concurrent_limits.dynamic_scaling.max_additional_positions` | 2 | int | `DYNAMIC_SCALING_MAX_EXTRA` | config/config.example.yaml:175 |
| `risk.volatility_sizing` | {'enabled': True, 'atr_window': 14, 'atr_floor_pct': 0.005, 'atr_ceiling_pct': 0.02, 'low_vol_multiplier': 1.2, 'baseline_multiplier': 1.0, 'high_vol_multipl... | dict |  | config/config.example.yaml:178 |
| `risk.volatility_sizing.enabled` | True | bool | `VOL_SIZING_ENABLED` | config/config.example.yaml:179 |
| `risk.volatility_sizing.atr_window` | 14 | int | `VOL_SIZING_ATR_WINDOW` | config/config.example.yaml:180 |
| `risk.volatility_sizing.atr_floor_pct` | 0.005 | float | `VOL_SIZING_ATR_FLOOR` | config/config.example.yaml:181 |
| `risk.volatility_sizing.atr_ceiling_pct` | 0.02 | float | `VOL_SIZING_ATR_CEILING` | config/config.example.yaml:182 |
| `risk.volatility_sizing.low_vol_multiplier` | 1.2 | float | `VOL_SIZING_LOW_MULT` | config/config.example.yaml:183 |
| `risk.volatility_sizing.baseline_multiplier` | 1.0 | float | `VOL_SIZING_BASE_MULT` | config/config.example.yaml:184 |
| `risk.volatility_sizing.high_vol_multiplier` | 0.6 | float | `VOL_SIZING_HIGH_MULT` | config/config.example.yaml:185 |
| `risk.volatility_sizing.min_position_size_pct` | 0.01 | float | `VOL_SIZING_MIN_POSITION_PCT` | config/config.example.yaml:186 |
| `class_limits` | {'meme': {'max_notional_per_trade': 10, 'risk_usd_cap': 0.5}, 'microcap': {'max_notional_per_trade': 15, 'risk_usd_cap': 0.75}, 'bluechip': {'max_notional_pe... | dict |  | config/config.example.yaml:189 |
| `class_limits.meme` | {'max_notional_per_trade': 10, 'risk_usd_cap': 0.5} | dict |  | config/config.example.yaml:190 |
| `class_limits.meme.max_notional_per_trade` | 10 | int | `CLASS_MEME_MAX_NOTIONAL` | config/config.example.yaml:191 |
| `class_limits.meme.risk_usd_cap` | 0.5 | float | `CLASS_MEME_RISK_CAP` | config/config.example.yaml:192 |
| `class_limits.microcap` | {'max_notional_per_trade': 15, 'risk_usd_cap': 0.75} | dict |  | config/config.example.yaml:193 |
| `class_limits.microcap.max_notional_per_trade` | 15 | int | `CLASS_MICROCAP_MAX_NOTIONAL` | config/config.example.yaml:194 |
| `class_limits.microcap.risk_usd_cap` | 0.75 | float | `CLASS_MICROCAP_RISK_CAP` | config/config.example.yaml:195 |
| `class_limits.bluechip` | {'max_notional_per_trade': 25, 'risk_usd_cap': 1.0} | dict |  | config/config.example.yaml:196 |
| `class_limits.bluechip.max_notional_per_trade` | 25 | int | `CLASS_BLUECHIP_MAX_NOTIONAL` | config/config.example.yaml:197 |
| `class_limits.bluechip.risk_usd_cap` | 1.0 | float | `CLASS_BLUECHIP_RISK_CAP` | config/config.example.yaml:198 |
| `notify` | {'send_all': True, 'push_no_signal': True, 'push_debug': True, 'min_cooldown_sec': 300, 'push_trail_updates': False} | dict |  | config/config.example.yaml:203 |
| `notify.send_all` | True | bool | `NOTIFY_SEND_ALL` | config/config.example.yaml:204 |
| `notify.push_no_signal` | True | bool | `NOTIFY_PUSH_NO_SIGNAL` | config/config.example.yaml:205 |
| `notify.push_debug` | True | bool | `NOTIFY_PUSH_DEBUG` | config/config.example.yaml:206 |
| `notify.min_cooldown_sec` | 300 | int | `NOTIFY_MIN_COOLDOWN_SEC` | config/config.example.yaml:207 |
| `notify.push_trail_updates` | False | bool | `NOTIFY_PUSH_TRAIL_UPDATES` | config/config.example.yaml:208 |
| `regime` | {'min_slow_candles': 90} | dict |  | config/config.example.yaml:213 |
| `regime.min_slow_candles` | 90 | int | `REGIME_MIN_SLOW_CANDLES` | config/config.example.yaml:214 |
| `websocket` | {'enabled': True, 'priority_enabled': True, 'max_data_age': 60, 'fallback_threshold': 3, 'buffer_size': 300, 'max_streams_per_exchange': {'bingx': 10, 'binan... | dict |  | config/config.example.yaml:219 |
| `websocket.enabled` | True | bool | `WEBSOCKET_ENABLED` | config/config.example.yaml:220 |
| `websocket.priority_enabled` | True | bool | `WEBSOCKET_PRIORITY_ENABLED` | config/config.example.yaml:221 |
| `websocket.max_data_age` | 60 | int | `WEBSOCKET_MAX_DATA_AGE` | config/config.example.yaml:222 |
| `websocket.fallback_threshold` | 3 | int | `WEBSOCKET_FALLBACK_THRESHOLD` | config/config.example.yaml:223 |
| `websocket.buffer_size` | 300 | int | `WEBSOCKET_BUFFER_SIZE` | config/config.example.yaml:224 |
| `websocket.max_streams_per_exchange` | {'bingx': 10, 'binance': 20, 'kucoinfutures': 15, 'default': 10} | dict |  | config/config.example.yaml:225 |
| `websocket.max_streams_per_exchange.bingx` | 10 | int | `WS_MAX_STREAMS_BINGX` | config/config.example.yaml:226 |
| `websocket.max_streams_per_exchange.binance` | 20 | int | `WS_MAX_STREAMS_BINANCE` | config/config.example.yaml:227 |
| `websocket.max_streams_per_exchange.kucoinfutures` | 15 | int | `WS_MAX_STREAMS_KUCOIN` | config/config.example.yaml:228 |
| `websocket.max_streams_per_exchange.default` | 10 | int | `WS_MAX_STREAMS_DEFAULT` | config/config.example.yaml:229 |
| `websocket.stream_timeframes` | '1m,5m,15m,30m,1h,4h' | str | `WS_STREAM_TIMEFRAMES` | config/config.example.yaml:230 |
| `websocket.reconnect_delay` | 5 | int | `WS_RECONNECT_DELAY` | config/config.example.yaml:231 |
| `websocket.max_reconnect_attempts` | 3 | int | `WS_MAX_RECONNECT_ATTEMPTS` | config/config.example.yaml:232 |
| `universe` | {'prefetch': {'enabled': True, 'startup_candle_count': 500}, 'fixed_symbols': 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT', 'auto_select': False, 'priority_or... | dict |  | config/config.example.yaml:237 |
| `universe.prefetch` | {'enabled': True, 'startup_candle_count': 500} | dict |  | config/config.example.yaml:238 |
| `universe.prefetch.enabled` | True | bool | `UNIVERSE_PREFETCH_ENABLED` | config/config.example.yaml:239 |
| `universe.prefetch.startup_candle_count` | 500 | int | `UNIVERSE_STARTUP_CANDLE_COUNT` | config/config.example.yaml:240 |
| `universe.fixed_symbols` | 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT' | str | `TRADING_SYMBOLS` | config/config.example.yaml:241 |
| `universe.auto_select` | False | bool | `UNIVERSE_AUTO_SELECT` | config/config.example.yaml:242 |
| `universe.priority_order` | 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT' | str | `TRADING_SYMBOLS_PRIORITY` | config/config.example.yaml:243 |
| `indicators` | {'rsi_period': 14, 'atr_period': 14, 'ema_fast': 21, 'ema_mid': 50, 'ema_slow': 200} | dict |  | config/config.example.yaml:248 |
| `indicators.rsi_period` | 14 | int | `INDICATOR_RSI_PERIOD` | config/config.example.yaml:249 |
| `indicators.atr_period` | 14 | int | `INDICATOR_ATR_PERIOD` | config/config.example.yaml:250 |
| `indicators.ema_fast` | 21 | int | `INDICATOR_EMA_FAST` | config/config.example.yaml:251 |
| `indicators.ema_mid` | 50 | int | `INDICATOR_EMA_MID` | config/config.example.yaml:252 |
| `indicators.ema_slow` | 200 | int | `INDICATOR_EMA_SLOW` | config/config.example.yaml:253 |
| `validator` | {'volume_analyzer_required': False} | dict |  | config/config.example.yaml:258 |
| `validator.volume_analyzer_required` | False | bool | `VALIDATOR_VOLUME_ANALYZER_REQUIRED` | config/config.example.yaml:262 |
| `volume_analyzer` | {'enabled': True, 'baseline_short_tf': '1h', 'baseline_medium_tf': '4h', 'short_lookback': 168, 'medium_lookback': 180, 'window_bars': 3, 'weight_short': 0.6... | dict |  | config/config.example.yaml:267 |
| `volume_analyzer.enabled` | True | bool | `VOLUME_ANALYZER_ENABLED` | config/config.example.yaml:268 |
| `volume_analyzer.baseline_short_tf` | '1h' | str | `VOLUME_ANALYZER_BASELINE_SHORT_TF` | config/config.example.yaml:269 |
| `volume_analyzer.baseline_medium_tf` | '4h' | str | `VOLUME_ANALYZER_BASELINE_MEDIUM_TF` | config/config.example.yaml:270 |
| `volume_analyzer.short_lookback` | 168 | int | `VOLUME_ANALYZER_SHORT_LOOKBACK` | config/config.example.yaml:271 |
| `volume_analyzer.medium_lookback` | 180 | int | `VOLUME_ANALYZER_MEDIUM_LOOKBACK` | config/config.example.yaml:272 |
| `volume_analyzer.window_bars` | 3 | int | `VOLUME_ANALYZER_WINDOW_BARS` | config/config.example.yaml:273 |
| `volume_analyzer.weight_short` | 0.6 | float |  | config/config.example.yaml:274 |
| `volume_analyzer.weight_medium` | 0.4 | float |  | config/config.example.yaml:275 |
| `volume_analyzer.sigmoid_alpha` | 1.2 | float | `VOLUME_ANALYZER_SIGMOID_ALPHA` | config/config.example.yaml:276 |
| `volume_analyzer.min_ratio` | 0.1 | float | `VOLUME_ANALYZER_MIN_RATIO` | config/config.example.yaml:277 |
| `volume_analyzer.max_ratio` | 10.0 | float | `VOLUME_ANALYZER_MAX_RATIO` | config/config.example.yaml:278 |
| `volume_analyzer.buckets` | ['[0.0, "LOW"]', '[0.3, "NORMAL"]', '[0.6, "HIGH"]', '[0.85, "EXTREME"]'] | list | `VOLUME_ANALYZER_BUCKETS` | config/config.example.yaml:279 |
| `adaptive_strategies` | {'enable': True, 'monitoring': {'enabled': True, 'report_interval': 300, 'track_performance': True}, 'performance': {'min_volatility_for_adjustment': 0.02, '... | dict |  | config/config.example.yaml:288 |
| `adaptive_strategies.enable` | True | bool | `ADAPTIVE_STRATEGIES_ENABLE` | config/config.example.yaml:289 |
| `adaptive_strategies.monitoring` | {'enabled': True, 'report_interval': 300, 'track_performance': True} | dict |  | config/config.example.yaml:290 |
| `adaptive_strategies.monitoring.enabled` | True | bool | `ADAPTIVE_MONITORING_ENABLED` | config/config.example.yaml:291 |
| `adaptive_strategies.monitoring.report_interval` | 300 | int | `ADAPTIVE_REPORT_INTERVAL` | config/config.example.yaml:292 |
| `adaptive_strategies.monitoring.track_performance` | True | bool | `ADAPTIVE_TRACK_PERFORMANCE` | config/config.example.yaml:293 |
| `adaptive_strategies.performance` | {'min_volatility_for_adjustment': 0.02, 'max_position_multiplier': 2.0, 'min_position_multiplier': 0.5} | dict |  | config/config.example.yaml:294 |
| `adaptive_strategies.performance.min_volatility_for_adjustment` | 0.02 | float | `ADAPTIVE_MIN_VOLATILITY` | config/config.example.yaml:295 |
| `adaptive_strategies.performance.max_position_multiplier` | 2.0 | float | `ADAPTIVE_MAX_POS_MULT` | config/config.example.yaml:296 |
| `adaptive_strategies.performance.min_position_multiplier` | 0.5 | float | `ADAPTIVE_MIN_POS_MULT` | config/config.example.yaml:297 |
| `signals` | {'bypass': {'enabled': True, 'rsi_oversold_threshold': 12, 'rsi_overbought_threshold': 88, 'force_swap_enabled': True}, 'duplicate_prevention': {'min_price_c... | dict |  | config/config.example.yaml:302 |
| `signals.bypass` | {'enabled': True, 'rsi_oversold_threshold': 12, 'rsi_overbought_threshold': 88, 'force_swap_enabled': True} | dict |  | config/config.example.yaml:305 |
| `signals.bypass.enabled` | True | bool | `SIGNAL_BYPASS_ENABLED` | config/config.example.yaml:306 |
| `signals.bypass.rsi_oversold_threshold` | 12 | int | `SIGNAL_BYPASS_RSI_OVERSOLD` | config/config.example.yaml:307 |
| `signals.bypass.rsi_overbought_threshold` | 88 | int | `SIGNAL_BYPASS_RSI_OVERBOUGHT` | config/config.example.yaml:308 |
| `signals.bypass.force_swap_enabled` | True | bool | `SIGNAL_FORCE_SWAP_ENABLED` | config/config.example.yaml:309 |
| `signals.duplicate_prevention` | {'min_price_change_pct': 0.0005, 'cooldown_seconds': 20, 'price_delta_bypass_threshold': 0.0015, 'price_delta_bypass_enabled': False, 'scale_in_min_price_cha... | dict |  | config/config.example.yaml:311 |
| `signals.duplicate_prevention.min_price_change_pct` | 0.0005 | float | `DUPLICATE_PREVENTION_THRESHOLD` | config/config.example.yaml:312 |
| `signals.duplicate_prevention.cooldown_seconds` | 20 | int | `DUPLICATE_PREVENTION_COOLDOWN` | config/config.example.yaml:313 |
| `signals.duplicate_prevention.price_delta_bypass_threshold` | 0.0015 | float | `PRICE_DELTA_BYPASS_THRESHOLD` | config/config.example.yaml:314 |
| `signals.duplicate_prevention.price_delta_bypass_enabled` | False | bool | `PRICE_DELTA_BYPASS_ENABLED` | config/config.example.yaml:315 |
| `signals.duplicate_prevention.scale_in_min_price_change_pct` | 0.0005 | float |  | config/config.example.yaml:317 |
| `signals.duplicate_prevention.scale_in_cooldown_seconds` | 20 | int |  | config/config.example.yaml:318 |
| `signals.duplicate_prevention.ml_duplicate_detection_enabled` | True | bool | `DUPLICATE_ML_DETECTION_ENABLED` | config/config.example.yaml:319 |
| `signals.duplicate_prevention.dynamic_cooldown` | {'enabled': True, 'high_delta_threshold': 15, 'medium_delta_threshold': 8, 'fast_cooldown_seconds': 15, 'medium_cooldown_seconds': 45, 'slow_cooldown_seconds... | dict |  | config/config.example.yaml:320 |
| `signals.duplicate_prevention.dynamic_cooldown.enabled` | True | bool | `DUPLICATE_DYNAMIC_COOLDOWN_ENABLED` | config/config.example.yaml:321 |
| `signals.duplicate_prevention.dynamic_cooldown.high_delta_threshold` | 15 | int | `DUPLICATE_DYNAMIC_HIGH_DELTA` | config/config.example.yaml:322 |
| `signals.duplicate_prevention.dynamic_cooldown.medium_delta_threshold` | 8 | int | `DUPLICATE_DYNAMIC_MEDIUM_DELTA` | config/config.example.yaml:323 |
| `signals.duplicate_prevention.dynamic_cooldown.fast_cooldown_seconds` | 15 | int | `DUPLICATE_DYNAMIC_FAST_SECONDS` | config/config.example.yaml:324 |
| `signals.duplicate_prevention.dynamic_cooldown.medium_cooldown_seconds` | 45 | int | `DUPLICATE_DYNAMIC_MEDIUM_SECONDS` | config/config.example.yaml:325 |
| `signals.duplicate_prevention.dynamic_cooldown.slow_cooldown_seconds` | 120 | int | `DUPLICATE_DYNAMIC_SLOW_SECONDS` | config/config.example.yaml:326 |
| `signals.oversold_bounce` | {'enable': True, 'ignore_regime': False, 'min_rr_ratio': 1.5, 'rsi_max': 45, 'adaptive_rsi_base': 32, 'adaptive_rsi_range': 8, 'adaptive_mode': 'dynamic', 'v... | dict |  | config/config.example.yaml:327 |
| `signals.oversold_bounce.enable` | True | bool | `STRATEGY_OB_ENABLED` | config/config.example.yaml:328 |
| `signals.oversold_bounce.ignore_regime` | False | bool | `STRATEGY_OB_IGNORE_REGIME` | config/config.example.yaml:329 |
| `signals.oversold_bounce.min_rr_ratio` | 1.5 | float | `STRATEGY_OB_MIN_RR` | config/config.example.yaml:330 |
| `signals.oversold_bounce.rsi_max` | 45 | int | `STRATEGY_OB_RSI_MAX` | config/config.example.yaml:331 |
| `signals.oversold_bounce.adaptive_rsi_base` | 32 | int | `RSI_BASE_OB` | config/config.example.yaml:332 |
| `signals.oversold_bounce.adaptive_rsi_range` | 8 | int | `RSI_RANGE_OB` | config/config.example.yaml:333 |
| `signals.oversold_bounce.adaptive_mode` | 'dynamic' | str | `STRATEGY_OB_ADAPTIVE_MODE` | config/config.example.yaml:334 |
| `signals.oversold_bounce.volatility_sensitivity` | 'medium' | str | `STRATEGY_OB_VOL_SENSITIVITY` | config/config.example.yaml:335 |
| `signals.oversold_bounce.tp_atr_mult` | 2.5 | float | `STRATEGY_OB_TP_ATR_MULT` | config/config.example.yaml:336 |
| `signals.oversold_bounce.sl_atr_mult` | 1.2 | float | `STRATEGY_OB_SL_ATR_MULT` | config/config.example.yaml:337 |
| `signals.oversold_bounce.min_tp_pct` | 0.008 | float | `STRATEGY_OB_MIN_TP_PCT` | config/config.example.yaml:338 |
| `signals.oversold_bounce.max_sl_pct` | 0.015 | float | `STRATEGY_OB_MAX_SL_PCT` | config/config.example.yaml:339 |
| `signals.short_the_rip` | {'enable': True, 'ignore_regime': False, 'min_rr_ratio': 1.5, 'rsi_min': 55, 'adaptive_rsi_base': 68, 'adaptive_rsi_range': 8, 'adaptive_mode': 'dynamic', 'v... | dict |  | config/config.example.yaml:340 |
| `signals.short_the_rip.enable` | True | bool | `STRATEGY_STR_ENABLED` | config/config.example.yaml:341 |
| `signals.short_the_rip.ignore_regime` | False | bool | `STRATEGY_STR_IGNORE_REGIME` | config/config.example.yaml:342 |
| `signals.short_the_rip.min_rr_ratio` | 1.5 | float | `STRATEGY_STR_MIN_RR` | config/config.example.yaml:343 |
| `signals.short_the_rip.rsi_min` | 55 | int | `STRATEGY_STR_RSI_MIN` | config/config.example.yaml:344 |
| `signals.short_the_rip.adaptive_rsi_base` | 68 | int | `RSI_BASE_STR` | config/config.example.yaml:345 |
| `signals.short_the_rip.adaptive_rsi_range` | 8 | int | `RSI_RANGE_STR` | config/config.example.yaml:346 |
| `signals.short_the_rip.adaptive_mode` | 'dynamic' | str | `STRATEGY_STR_ADAPTIVE_MODE` | config/config.example.yaml:347 |
| `signals.short_the_rip.volatility_sensitivity` | 'medium' | str | `STRATEGY_STR_VOL_SENSITIVITY` | config/config.example.yaml:348 |
| `signals.short_the_rip.tp_atr_mult` | 3.0 | float | `STRATEGY_STR_TP_ATR_MULT` | config/config.example.yaml:349 |
| `signals.short_the_rip.sl_atr_mult` | 1.5 | float | `STRATEGY_STR_SL_ATR_MULT` | config/config.example.yaml:350 |
| `signals.short_the_rip.min_tp_pct` | 0.01 | float | `STRATEGY_STR_MIN_TP_PCT` | config/config.example.yaml:351 |
| `signals.short_the_rip.max_sl_pct` | 0.02 | float | `STRATEGY_STR_MAX_SL_PCT` | config/config.example.yaml:352 |
| `signals.short_the_rip.mtf_confirmation` | {'enabled': True, 'require_15m': False, 'require_1h': False, 'rsi_15m_min': 62.0, 'min_15m_close_over_ema50_pct': 0.0015, 'require_1h_bearish_ema_stack': Tru... | dict |  | config/config.example.yaml:354 |
| `signals.short_the_rip.mtf_confirmation.enabled` | True | bool |  | config/config.example.yaml:355 |
| `signals.short_the_rip.mtf_confirmation.require_15m` | False | bool |  | config/config.example.yaml:356 |
| `signals.short_the_rip.mtf_confirmation.require_1h` | False | bool |  | config/config.example.yaml:357 |
| `signals.short_the_rip.mtf_confirmation.rsi_15m_min` | 62.0 | float |  | config/config.example.yaml:359 |
| `signals.short_the_rip.mtf_confirmation.min_15m_close_over_ema50_pct` | 0.0015 | float |  | config/config.example.yaml:360 |
| `signals.short_the_rip.mtf_confirmation.require_1h_bearish_ema_stack` | True | bool |  | config/config.example.yaml:362 |
| `signals.short_the_rip.mtf_confirmation.rsi_1h_max` | 60.0 | float |  | config/config.example.yaml:363 |
| `signals.short_the_rip.mtf_confirmation.on_missing_15m` | 'skip' | str |  | config/config.example.yaml:365 |
| `signals.short_the_rip.mtf_confirmation.on_missing_1h` | 'skip' | str |  | config/config.example.yaml:366 |
| `signals.short_the_rip.mtf_confirmation.min_bars_rsi` | 20 | int |  | config/config.example.yaml:371 |
| `signals.short_the_rip.mtf_confirmation.min_bars_ema21` | 30 | int |  | config/config.example.yaml:372 |
| `signals.short_the_rip.mtf_confirmation.min_bars_ema50` | 100 | int |  | config/config.example.yaml:373 |
| `signals.short_the_rip.mtf_confirmation.min_bars_ema200` | 250 | int |  | config/config.example.yaml:374 |
| `signals.short_the_rip.volatility_stop` | {'enabled': True, 'min_sl_pct': 0.0025, 'max_sl_pct': 0.02, 'overrides': {'low': {'atr_scale': 0.75, 'min_sl_pct': 0.0015, 'max_sl_pct': 0.015}, 'normal': {'... | dict |  | config/config.example.yaml:375 |
| `signals.short_the_rip.volatility_stop.enabled` | True | bool | `STRATEGY_STR_VOL_STOP_ENABLED` | config/config.example.yaml:376 |
| `signals.short_the_rip.volatility_stop.min_sl_pct` | 0.0025 | float | `STRATEGY_STR_VOL_STOP_MIN_SL` | config/config.example.yaml:377 |
| `signals.short_the_rip.volatility_stop.max_sl_pct` | 0.02 | float | `STRATEGY_STR_VOL_STOP_MAX_SL` | config/config.example.yaml:378 |
| `signals.short_the_rip.volatility_stop.overrides` | {'low': {'atr_scale': 0.75, 'min_sl_pct': 0.0015, 'max_sl_pct': 0.015}, 'normal': {'atr_scale': 1.0}, 'high': {'atr_scale': 1.15, 'min_sl_pct': 0.003, 'max_s... | dict |  | config/config.example.yaml:379 |
| `signals.short_the_rip.volatility_stop.overrides.low` | {'atr_scale': 0.75, 'min_sl_pct': 0.0015, 'max_sl_pct': 0.015} | dict |  | config/config.example.yaml:380 |
| `signals.short_the_rip.volatility_stop.overrides.low.atr_scale` | 0.75 | float | `STRATEGY_STR_VOL_LOW_ATR_SCALE` | config/config.example.yaml:381 |
| `signals.short_the_rip.volatility_stop.overrides.low.min_sl_pct` | 0.0015 | float | `STRATEGY_STR_VOL_LOW_MIN_SL` | config/config.example.yaml:382 |
| `signals.short_the_rip.volatility_stop.overrides.low.max_sl_pct` | 0.015 | float | `STRATEGY_STR_VOL_LOW_MAX_SL` | config/config.example.yaml:383 |
| `signals.short_the_rip.volatility_stop.overrides.normal` | {'atr_scale': 1.0} | dict |  | config/config.example.yaml:384 |
| `signals.short_the_rip.volatility_stop.overrides.normal.atr_scale` | 1.0 | float | `STRATEGY_STR_VOL_NORMAL_ATR_SCALE` | config/config.example.yaml:385 |
| `signals.short_the_rip.volatility_stop.overrides.high` | {'atr_scale': 1.15, 'min_sl_pct': 0.003, 'max_sl_pct': 0.025} | dict |  | config/config.example.yaml:386 |
| `signals.short_the_rip.volatility_stop.overrides.high.atr_scale` | 1.15 | float | `STRATEGY_STR_VOL_HIGH_ATR_SCALE` | config/config.example.yaml:387 |
| `signals.short_the_rip.volatility_stop.overrides.high.min_sl_pct` | 0.003 | float | `STRATEGY_STR_VOL_HIGH_MIN_SL` | config/config.example.yaml:388 |
| `signals.short_the_rip.volatility_stop.overrides.high.max_sl_pct` | 0.025 | float | `STRATEGY_STR_VOL_HIGH_MAX_SL` | config/config.example.yaml:389 |
| `signals.short_the_rip.symbols` | {'BTC/USDT:USDT': {'rsi_threshold': 50}, 'ETH/USDT:USDT': {'rsi_threshold': 50}, 'SOL/USDT:USDT': {'rsi_threshold': 50}} | dict |  | config/config.example.yaml:390 |
| `signals.short_the_rip.symbols.BTC/USDT:USDT` | {'rsi_threshold': 50} | dict |  | config/config.example.yaml:391 |
| `signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | 50 | int | `RSI_THRESHOLD_BTC` | config/config.example.yaml:392 |
| `signals.short_the_rip.symbols.ETH/USDT:USDT` | {'rsi_threshold': 50} | dict |  | config/config.example.yaml:393 |
| `signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | 50 | int | `RSI_THRESHOLD_ETH` | config/config.example.yaml:394 |
| `signals.short_the_rip.symbols.SOL/USDT:USDT` | {'rsi_threshold': 50} | dict |  | config/config.example.yaml:395 |
| `signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | 50 | int | `RSI_THRESHOLD_SOL` | config/config.example.yaml:396 |
| `strategies` | {'regime_routing': {'bullish': {'preferred_strategies': '["trend_follower", "breakout_hunter"]', 'preferred_priority': 0.8, 'fallback_priority': 0.4, 'queue_... | dict |  | config/config.example.yaml:401 |
| `strategies.regime_routing` | {'bullish': {'preferred_strategies': '["trend_follower", "breakout_hunter"]', 'preferred_priority': 0.8, 'fallback_priority': 0.4, 'queue_priority_boost': 0.... | dict |  | config/config.example.yaml:402 |
| `strategies.regime_routing.bullish` | {'preferred_strategies': '["trend_follower", "breakout_hunter"]', 'preferred_priority': 0.8, 'fallback_priority': 0.4, 'queue_priority_boost': 0.75} | dict |  | config/config.example.yaml:403 |
| `strategies.regime_routing.bullish.preferred_strategies` | '["trend_follower", "breakout_hunter"]' | str | `REGIME_BULLISH_STRATEGIES` | config/config.example.yaml:404 |
| `strategies.regime_routing.bullish.preferred_priority` | 0.8 | float | `REGIME_BULLISH_PRIORITY` | config/config.example.yaml:405 |
| `strategies.regime_routing.bullish.fallback_priority` | 0.4 | float | `REGIME_BULLISH_FALLBACK` | config/config.example.yaml:406 |
| `strategies.regime_routing.bullish.queue_priority_boost` | 0.75 | float | `REGIME_BULLISH_QUEUE_BOOST` | config/config.example.yaml:407 |
| `strategies.regime_routing.bearish` | {'preferred_strategies': '["short_the_rip", "mean_reversion"]', 'preferred_priority': 0.85} | dict |  | config/config.example.yaml:408 |
| `strategies.regime_routing.bearish.preferred_strategies` | '["short_the_rip", "mean_reversion"]' | str | `REGIME_BEARISH_STRATEGIES` | config/config.example.yaml:409 |
| `strategies.regime_routing.bearish.preferred_priority` | 0.85 | float | `REGIME_BEARISH_PRIORITY` | config/config.example.yaml:410 |
| `strategies.regime_routing.neutral` | {'preferred_strategies': '["range_sniper", "mean_reversion"]', 'preferred_priority': 0.7} | dict |  | config/config.example.yaml:411 |
| `strategies.regime_routing.neutral.preferred_strategies` | '["range_sniper", "mean_reversion"]' | str | `REGIME_NEUTRAL_STRATEGIES` | config/config.example.yaml:412 |
| `strategies.regime_routing.neutral.preferred_priority` | 0.7 | float | `REGIME_NEUTRAL_PRIORITY` | config/config.example.yaml:413 |
| `strategies.regime_routing.volatile` | {'preferred_strategies': '["scalper", "volatility_breakout"]', 'preferred_priority': 0.9, 'queue_priority_boost': 0.9} | dict |  | config/config.example.yaml:414 |
| `strategies.regime_routing.volatile.preferred_strategies` | '["scalper", "volatility_breakout"]' | str | `REGIME_VOLATILE_STRATEGIES` | config/config.example.yaml:415 |
| `strategies.regime_routing.volatile.preferred_priority` | 0.9 | float | `REGIME_VOLATILE_PRIORITY` | config/config.example.yaml:416 |
| `strategies.regime_routing.volatile.queue_priority_boost` | 0.9 | float | `REGIME_VOLATILE_QUEUE_BOOST` | config/config.example.yaml:417 |
| `strategies.regime_routing.default` | {'preferred_priority': 0.5, 'fallback_priority': 0.4} | dict |  | config/config.example.yaml:418 |
| `strategies.regime_routing.default.preferred_priority` | 0.5 | float | `REGIME_DEFAULT_PRIORITY` | config/config.example.yaml:419 |
| `strategies.regime_routing.default.fallback_priority` | 0.4 | float | `REGIME_DEFAULT_FALLBACK` | config/config.example.yaml:420 |
| `strategies.adaptive_ob` | {'allow_low_volume': False, 'volume_filters': {'enabled': True, 'min_bucket': 'NORMAL', 'high_volume_min_bucket': 'HIGH', 'use_volume_strength_in_score': Tru... | dict |  | config/config.example.yaml:422 |
| `strategies.adaptive_ob.allow_low_volume` | False | bool | `STRATEGY_OB_ALLOW_LOW_VOLUME` | config/config.example.yaml:423 |
| `strategies.adaptive_ob.volume_filters` | {'enabled': True, 'min_bucket': 'NORMAL', 'high_volume_min_bucket': 'HIGH', 'use_volume_strength_in_score': True, 'volume_score_weight': 0.15} | dict |  | config/config.example.yaml:424 |
| `strategies.adaptive_ob.volume_filters.enabled` | True | bool | `STRATEGY_OB_VOLUME_FILTERS_ENABLED` | config/config.example.yaml:425 |
| `strategies.adaptive_ob.volume_filters.min_bucket` | 'NORMAL' | str | `STRATEGY_OB_VOLUME_FILTERS_MIN_BUCKET` | config/config.example.yaml:426 |
| `strategies.adaptive_ob.volume_filters.high_volume_min_bucket` | 'HIGH' | str | `STRATEGY_OB_VOLUME_FILTERS_HIGH_VOL_MIN_BUCKET` | config/config.example.yaml:427 |
| `strategies.adaptive_ob.volume_filters.use_volume_strength_in_score` | True | bool | `STRATEGY_OB_VOLUME_FILTERS_USE_STRENGTH` | config/config.example.yaml:428 |
| `strategies.adaptive_ob.volume_filters.volume_score_weight` | 0.15 | float | `STRATEGY_OB_VOLUME_FILTERS_WEIGHT` | config/config.example.yaml:429 |
| `strategies.adaptive_short_the_rip` | {'allow_low_volume': False, 'volume_filters': {'enabled': True, 'min_bucket': 'NORMAL', 'high_volume_min_bucket': 'HIGH', 'use_volume_strength_in_score': Tru... | dict |  | config/config.example.yaml:431 |
| `strategies.adaptive_short_the_rip.allow_low_volume` | False | bool | `STRATEGY_STR_ALLOW_LOW_VOLUME` | config/config.example.yaml:432 |
| `strategies.adaptive_short_the_rip.volume_filters` | {'enabled': True, 'min_bucket': 'NORMAL', 'high_volume_min_bucket': 'HIGH', 'use_volume_strength_in_score': True, 'volume_score_weight': 0.15} | dict |  | config/config.example.yaml:433 |
| `strategies.adaptive_short_the_rip.volume_filters.enabled` | True | bool | `STRATEGY_STR_VOLUME_FILTERS_ENABLED` | config/config.example.yaml:434 |
| `strategies.adaptive_short_the_rip.volume_filters.min_bucket` | 'NORMAL' | str | `STRATEGY_STR_VOLUME_FILTERS_MIN_BUCKET` | config/config.example.yaml:435 |
| `strategies.adaptive_short_the_rip.volume_filters.high_volume_min_bucket` | 'HIGH' | str | `STRATEGY_STR_VOLUME_FILTERS_HIGH_VOL_MIN_BUCKET` | config/config.example.yaml:436 |
| `strategies.adaptive_short_the_rip.volume_filters.use_volume_strength_in_score` | True | bool | `STRATEGY_STR_VOLUME_FILTERS_USE_STRENGTH` | config/config.example.yaml:437 |
| `strategies.adaptive_short_the_rip.volume_filters.volume_score_weight` | 0.15 | float | `STRATEGY_STR_VOLUME_FILTERS_WEIGHT` | config/config.example.yaml:438 |
| `position_management` | {'exit_monitoring': {'enabled': True, 'check_frequency': 5}, 'time_based_exit': {'max_position_duration': 3600}} | dict |  | config/config.example.yaml:443 |
| `position_management.exit_monitoring` | {'enabled': True, 'check_frequency': 5} | dict |  | config/config.example.yaml:444 |
| `position_management.exit_monitoring.enabled` | True | bool | `EXIT_MONITORING_ENABLED` | config/config.example.yaml:445 |
| `position_management.exit_monitoring.check_frequency` | 5 | int | `EXIT_CHECK_FREQUENCY` | config/config.example.yaml:446 |
| `position_management.time_based_exit` | {'max_position_duration': 3600} | dict |  | config/config.example.yaml:447 |
| `position_management.time_based_exit.max_position_duration` | 3600 | int | `MAX_POSITION_DURATION` | config/config.example.yaml:448 |
| `pyramiding` | {'enabled': False, 'max_layers_per_symbol': 3, 'min_scale_in_quality': 0.8, 'min_scale_in_unrealized_pnl_pct': 0.005, 'min_scale_in_distance_pct': 0.005} | dict |  | config/config.example.yaml:453 |
| `pyramiding.enabled` | False | bool | `PYRAMIDING_ENABLED` | config/config.example.yaml:454 |
| `pyramiding.max_layers_per_symbol` | 3 | int | `PYRAMIDING_MAX_LAYERS_PER_SYMBOL` | config/config.example.yaml:455 |
| `pyramiding.min_scale_in_quality` | 0.8 | float | `PYRAMIDING_MIN_SCALE_IN_QUALITY` | config/config.example.yaml:456 |
| `pyramiding.min_scale_in_unrealized_pnl_pct` | 0.005 | float | `PYRAMIDING_MIN_SCALE_IN_PNL` | config/config.example.yaml:457 |
| `pyramiding.min_scale_in_distance_pct` | 0.005 | float | `PYRAMIDING_MIN_SCALE_IN_DISTANCE` | config/config.example.yaml:458 |
| `quarantine` | {'enable': True, 'days': 7, 'file': 'data/quarantine.json'} | dict |  | config/config.example.yaml:463 |
| `quarantine.enable` | True | bool | `QUARANTINE_ENABLE` | config/config.example.yaml:464 |
| `quarantine.days` | 7 | int | `QUARANTINE_DAYS` | config/config.example.yaml:465 |
| `quarantine.file` | 'data/quarantine.json' | str | `QUARANTINE_FILE` | config/config.example.yaml:466 |
| `ml` | {'enabled': True, 'features': {'rsi_period': 14, 'atr_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'bb_period': 20, 'bb_std': 2, 'volatil... | dict |  | config/config.example.yaml:475 |
| `ml.enabled` | True | bool |  | config/config.example.yaml:476 |
| `ml.features` | {'rsi_period': 14, 'atr_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'bb_period': 20, 'bb_std': 2, 'volatility_windows': '[5, 10, 20, 50]... | dict |  | config/config.example.yaml:483 |
| `ml.features.rsi_period` | 14 | int | `ML_FEAT_RSI_PERIOD` | config/config.example.yaml:484 |
| `ml.features.atr_period` | 14 | int | `ML_FEAT_ATR_PERIOD` | config/config.example.yaml:485 |
| `ml.features.macd_fast` | 12 | int | `ML_FEAT_MACD_FAST` | config/config.example.yaml:486 |
| `ml.features.macd_slow` | 26 | int | `ML_FEAT_MACD_SLOW` | config/config.example.yaml:487 |
| `ml.features.macd_signal` | 9 | int | `ML_FEAT_MACD_SIGNAL` | config/config.example.yaml:488 |
| `ml.features.bb_period` | 20 | int | `ML_FEAT_BB_PERIOD` | config/config.example.yaml:489 |
| `ml.features.bb_std` | 2 | int | `ML_FEAT_BB_STD` | config/config.example.yaml:490 |
| `ml.features.volatility_windows` | '[5, 10, 20, 50]' | str | `ML_FEAT_VOL_WINDOWS` | config/config.example.yaml:491 |
| `ml.features.momentum_windows` | '[5, 10, 20, 50]' | str | `ML_FEAT_MOM_WINDOWS` | config/config.example.yaml:492 |
| `ml.price_prediction` | {'enabled': True, 'timeframes': "['5m', '15m', '30m', '1h', '4h']", 'update_interval_seconds': 60, 'cache_ttl_seconds': 300, 'models': "['lstm', 'transformer... | dict |  | config/config.example.yaml:498 |
| `ml.price_prediction.enabled` | True | bool | `ML_PRED_ENABLED` | config/config.example.yaml:500 |
| `ml.price_prediction.timeframes` | "['5m', '15m', '30m', '1h', '4h']" | str | `ML_PRED_TIMEFRAMES` | config/config.example.yaml:501 |
| `ml.price_prediction.update_interval_seconds` | 60 | int | `ML_PRED_UPDATE_INTERVAL` | config/config.example.yaml:502 |
| `ml.price_prediction.cache_ttl_seconds` | 300 | int | `ML_PRED_CACHE_TTL` | config/config.example.yaml:503 |
| `ml.price_prediction.models` | "['lstm', 'transformer']" | str | `ML_MODELS` | config/config.example.yaml:506 |
| `ml.price_prediction.feature_size` | 42 | int | `ML_FEATURE_SIZE` | config/config.example.yaml:507 |
| `ml.price_prediction.forecast_horizon` | 12 | int | `ML_FORECAST_HORIZON` | config/config.example.yaml:508 |
| `ml.price_prediction.model_params` | {'lstm': {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2}, 'transformer': {'d_model': 256, 'nhead': 8, 'num_layers': 6}} | dict |  | config/config.example.yaml:511 |
| `ml.price_prediction.model_params.lstm` | {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2} | dict |  | config/config.example.yaml:512 |
| `ml.price_prediction.model_params.lstm.hidden_size` | 128 | int | `ML_PRED_LSTM_HIDDEN` | config/config.example.yaml:513 |
| `ml.price_prediction.model_params.lstm.num_layers` | 3 | int | `ML_PRED_LSTM_LAYERS` | config/config.example.yaml:514 |
| `ml.price_prediction.model_params.lstm.dropout` | 0.2 | float | `ML_PRED_LSTM_DROPOUT` | config/config.example.yaml:515 |
| `ml.price_prediction.model_params.transformer` | {'d_model': 256, 'nhead': 8, 'num_layers': 6} | dict |  | config/config.example.yaml:516 |
| `ml.price_prediction.model_params.transformer.d_model` | 256 | int | `ML_PRED_TRANS_D_MODEL` | config/config.example.yaml:517 |
| `ml.price_prediction.model_params.transformer.nhead` | 8 | int | `ML_PRED_TRANS_NHEAD` | config/config.example.yaml:518 |
| `ml.price_prediction.model_params.transformer.num_layers` | 6 | int | `ML_PRED_TRANS_LAYERS` | config/config.example.yaml:519 |
| `ml.price_prediction.ensemble_weights` | {'lstm': 0.5, 'transformer': 0.5} | dict |  | config/config.example.yaml:522 |
| `ml.price_prediction.ensemble_weights.lstm` | 0.5 | float | `ML_PRED_WEIGHT_LSTM` | config/config.example.yaml:523 |
| `ml.price_prediction.ensemble_weights.transformer` | 0.5 | float | `ML_PRED_WEIGHT_TRANSFORMER` | config/config.example.yaml:524 |
| `ml.regime_prediction` | {'enabled': True, 'min_confidence_threshold': 0.6, 'soft_weighting_enabled': True, 'min_confidence_hard_reject': 0.3, 'min_confidence_full_weight': 0.6, 'mod... | dict |  | config/config.example.yaml:531 |
| `ml.regime_prediction.enabled` | True | bool | `ML_REGIME_ENABLED` | config/config.example.yaml:532 |
| `ml.regime_prediction.min_confidence_threshold` | 0.6 | float | `ML_REGIME_MIN_CONFIDENCE` | config/config.example.yaml:533 |
| `ml.regime_prediction.soft_weighting_enabled` | True | bool | `REGIME_SOFT_WEIGHT_ENABLED` | config/config.example.yaml:537 |
| `ml.regime_prediction.min_confidence_hard_reject` | 0.3 | float | `REGIME_MIN_CONF_REJECT` | config/config.example.yaml:538 |
| `ml.regime_prediction.min_confidence_full_weight` | 0.6 | float | `REGIME_MIN_CONF_FULL` | config/config.example.yaml:539 |
| `ml.regime_prediction.model_params` | {'random_forest': {'n_estimators': 150, 'max_depth': 15}, 'xgboost': {'n_estimators': 200, 'learning_rate': 0.05}, 'lstm_regime': {'hidden_size': 64, 'num_la... | dict |  | config/config.example.yaml:550 |
| `ml.regime_prediction.model_params.random_forest` | {'n_estimators': 150, 'max_depth': 15} | dict |  | config/config.example.yaml:551 |
| `ml.regime_prediction.model_params.random_forest.n_estimators` | 150 | int | `ML_RF_N_ESTIMATORS` | config/config.example.yaml:552 |
| `ml.regime_prediction.model_params.random_forest.max_depth` | 15 | int | `ML_RF_MAX_DEPTH` | config/config.example.yaml:553 |
| `ml.regime_prediction.model_params.xgboost` | {'n_estimators': 200, 'learning_rate': 0.05} | dict |  | config/config.example.yaml:554 |
| `ml.regime_prediction.model_params.xgboost.n_estimators` | 200 | int | `ML_XGB_N_ESTIMATORS` | config/config.example.yaml:555 |
| `ml.regime_prediction.model_params.xgboost.learning_rate` | 0.05 | float | `ML_XGB_LEARNING_RATE` | config/config.example.yaml:556 |
| `ml.regime_prediction.model_params.lstm_regime` | {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.6} | dict |  | config/config.example.yaml:557 |
| `ml.regime_prediction.model_params.lstm_regime.hidden_size` | 64 | int |  | config/config.example.yaml:559 |
| `ml.regime_prediction.model_params.lstm_regime.num_layers` | 2 | int |  | config/config.example.yaml:560 |
| `ml.regime_prediction.model_params.lstm_regime.dropout` | 0.6 | float |  | config/config.example.yaml:561 |
| `ml.regime_prediction.ensemble_weights` | {'random_forest': 0.6, 'xgboost': 0.4} | dict |  | config/config.example.yaml:564 |
| `ml.regime_prediction.ensemble_weights.random_forest` | 0.6 | float | `ML_REGIME_WEIGHT_RF` | config/config.example.yaml:565 |
| `ml.regime_prediction.ensemble_weights.xgboost` | 0.4 | float | `ML_REGIME_WEIGHT_XGB` | config/config.example.yaml:566 |
| `ml.reinforcement_learning` | {'enabled': True, 'legacy_dqn_enabled': False, 'ppo_enabled': True, 'ppo_symbols': 'BTC/USDT:USDT', 'ppo_timeframe': '1h', 'ppo_model_path': 'artifacts/ppo/p... | dict |  | config/config.example.yaml:573 |
| `ml.reinforcement_learning.enabled` | True | bool | `ML_RL_ENABLED` | config/config.example.yaml:574 |
| `ml.reinforcement_learning.legacy_dqn_enabled` | False | bool | `ML_RL_LEGACY_ENABLED` | config/config.example.yaml:575 |
| `ml.reinforcement_learning.ppo_enabled` | True | bool | `ML_RL_PPO_ENABLED` | config/config.example.yaml:576 |
| `ml.reinforcement_learning.ppo_symbols` | 'BTC/USDT:USDT' | str | `ML_RL_PPO_SYMBOLS` | config/config.example.yaml:577 |
| `ml.reinforcement_learning.ppo_timeframe` | '1h' | str | `ML_RL_PPO_TIMEFRAME` | config/config.example.yaml:578 |
| `ml.reinforcement_learning.ppo_model_path` | 'artifacts/ppo/ppo_trading_agent.zip' | str | `ML_RL_PPO_MODEL` | config/config.example.yaml:579 |
| `ml.reinforcement_learning.ppo_fallback_score` | 0.5 | float | `ML_RL_PPO_FALLBACK` | config/config.example.yaml:580 |
| `ml.reinforcement_learning.ppo_rr_down_mult` | 0.9 | float | `ML_RL_PPO_RR_DOWN` | config/config.example.yaml:581 |
| `ml.reinforcement_learning.ppo_rr_up_mult` | 1.3 | float | `ML_RL_PPO_RR_UP` | config/config.example.yaml:582 |
| `ml.reinforcement_learning.ppo_position_base` | 0.5 | float | `ML_RL_PPO_POS_BASE` | config/config.example.yaml:583 |
| `ml.reinforcement_learning.ppo_position_bonus` | 0.5 | float | `ML_RL_PPO_POS_BONUS` | config/config.example.yaml:584 |
| `ml.reinforcement_learning.ppo_lookback_bars` | 240 | int | `ML_RL_PPO_LOOKBACK_BARS` | config/config.example.yaml:585 |
| `ml.reinforcement_learning.ppo_lookback_windows` | '[12, 24, 48, 96]' | str | `ML_RL_PPO_LOOKBACK_WINDOWS` | config/config.example.yaml:586 |
| `ml.reinforcement_learning.training_mode` | True | bool | `ML_RL_TRAINING_MODE` | config/config.example.yaml:604 |
| `ml.reinforcement_learning.hold_confidence_threshold` | 0.6 | float | `ML_RL_HOLD_CONFIDENCE_THRESHOLD` | config/config.example.yaml:607 |
| `ml.reinforcement_learning.epsilon_inference` | 0.01 | float | `ML_RL_EPSILON_INFERENCE` | config/config.example.yaml:610 |
| `ml.reinforcement_learning.epsilon_start` | 1.0 | float | `ML_RL_EPSILON_START` | config/config.example.yaml:611 |
| `ml.reinforcement_learning.epsilon_decay` | 0.97 | float | `ML_RL_EPSILON_DECAY` | config/config.example.yaml:612 |
| `ml.reinforcement_learning.epsilon_min` | 0.01 | float | `ML_RL_EPSILON_MIN` | config/config.example.yaml:613 |
| `ml.reinforcement_learning.regime_bias_strength` | 0.0 | float | `ML_RL_REGIME_BIAS` | config/config.example.yaml:615 |
| `ml.reinforcement_learning.max_regime_bias` | 3.0 | float | `ML_RL_MAX_REGIME_BIAS` | config/config.example.yaml:616 |
| `ml.reinforcement_learning.min_regime_confidence_for_bias` | 0.6 | float | `ML_RL_MIN_REGIME_CONF` | config/config.example.yaml:617 |
| `ml.reinforcement_learning.q_std_bypass_threshold` | 0.0001 | float | `ML_RL_Q_STD_BYPASS` | config/config.example.yaml:618 |
| `ml.reinforcement_learning.risk_penalty_strength` | 10.0 | float | `ML_RL_RISK_PENALTY` | config/config.example.yaml:619 |
| `ml.reinforcement_learning.learning_rate` | 3e-05 | float | `ML_RL_LEARNING_RATE` | config/config.example.yaml:622 |
| `ml.reinforcement_learning.gamma` | 0.95 | float | `ML_RL_GAMMA` | config/config.example.yaml:623 |
| `ml.reinforcement_learning.gradient_clip_norm` | 1.0 | float | `ML_RL_GRADIENT_CLIP` | config/config.example.yaml:624 |
| `ml.reinforcement_learning.reward_clip_enabled` | True | bool | `ML_RL_REWARD_CLIP_ENABLED` | config/config.example.yaml:627 |
| `ml.reinforcement_learning.reward_clip_min` | -2.0 | float | `RL_REWARD_CLIP_MIN` | config/config.example.yaml:628 |
| `ml.reinforcement_learning.reward_clip_max` | 2.0 | float | `RL_REWARD_CLIP_MAX` | config/config.example.yaml:629 |
| `ml.reinforcement_learning.reward_scale` | 1.0 | float | `RL_REWARD_SCALE` | config/config.example.yaml:630 |
| `ml.reinforcement_learning.trade_penalty_alpha` | 0.0 | float | `ML_RL_TRADE_PENALTY_ALPHA` | config/config.example.yaml:635 |
| `ml.reinforcement_learning.idle_cost` | 0.0 | float | `ML_RL_IDLE_COST` | config/config.example.yaml:639 |
| `ml.reinforcement_learning.target_update_freq` | 50 | int | `ML_RL_TARGET_UPDATE_FREQ` | config/config.example.yaml:641 |
| `ml.reinforcement_learning.batch_size` | 32 | int | `ML_RL_BATCH_SIZE` | config/config.example.yaml:642 |
| `ml.reinforcement_learning.buffer_size` | 100000 | int | `ML_RL_BUFFER_SIZE` | config/config.example.yaml:643 |
| `ml.signal_scoring` | {'enabled': True, 'min_score_to_trade': 75, 'weights': {'strategy': 0.3, 'ml_price': 0.3, 'regime': 0.2, 'risk_reward': 0.2}} | dict |  | config/config.example.yaml:649 |
| `ml.signal_scoring.enabled` | True | bool | `SIGNAL_SCORING_ENABLED` | config/config.example.yaml:650 |
| `ml.signal_scoring.min_score_to_trade` | 75 | int | `SIGNAL_MIN_SCORE` | config/config.example.yaml:651 |
| `ml.signal_scoring.weights` | {'strategy': 0.3, 'ml_price': 0.3, 'regime': 0.2, 'risk_reward': 0.2} | dict |  | config/config.example.yaml:654 |
| `ml.signal_scoring.weights.strategy` | 0.3 | float | `SCORE_WEIGHT_STRATEGY` | config/config.example.yaml:655 |
| `ml.signal_scoring.weights.ml_price` | 0.3 | float | `SCORE_WEIGHT_ML` | config/config.example.yaml:656 |
| `ml.signal_scoring.weights.regime` | 0.2 | float | `SCORE_WEIGHT_REGIME` | config/config.example.yaml:657 |
| `ml.signal_scoring.weights.risk_reward` | 0.2 | float | `SCORE_WEIGHT_RR` | config/config.example.yaml:658 |
| `ml.gemma` | {'enabled': True, 'model_path': 'data/models/final/gemma_price.pt', 'features_path': 'features/gemma/selected/gemma_price_selected_82.json', 'feature_count':... | dict |  | config/config.example.yaml:663 |
| `ml.gemma.enabled` | True | bool | `GEMMA_ENABLED` | config/config.example.yaml:666 |
| `ml.gemma.model_path` | 'data/models/final/gemma_price.pt' | str | `GEMMA_MODEL_PATH` | config/config.example.yaml:670 |
| `ml.gemma.features_path` | 'features/gemma/selected/gemma_price_selected_82.json' | str | `GEMMA_FEATURES_PATH` | config/config.example.yaml:673 |
| `ml.gemma.feature_count` | 82 | int | `GEMMA_FEATURE_COUNT` | config/config.example.yaml:674 |
| `ml.gemma.scaler_path` | 'data/models/final/gemma_price_scaler.joblib' | str | `GEMMA_SCALER_PATH` | config/config.example.yaml:678 |
| `ml.gemma.feature_mask_path` | 'data/cache/gemma/feature_selection_mask.npy' | str | `GEMMA_FEATURE_MASK_PATH` | config/config.example.yaml:679 |
| `ml.gemma.shadow_mode` | False | bool | `GEMMA_SHADOW_MODE` | config/config.example.yaml:682 |
| `ml.gemma.shadow_duration_hours` | 48 | int | `GEMMA_SHADOW_DURATION` | config/config.example.yaml:683 |
| `ml.gemma.cache_ttl` | 30 | int | `GEMMA_CACHE_TTL` | config/config.example.yaml:686 |
| `ml.gemma.max_inference_time` | 0.5 | float | `GEMMA_MAX_INFERENCE_TIME` | config/config.example.yaml:687 |
| `ml.gemma.circuit_breaker` | {'enabled': True, 'failure_threshold': 5, 'recovery_timeout': 60} | dict |  | config/config.example.yaml:691 |
| `ml.gemma.circuit_breaker.enabled` | True | bool | `GEMMA_CIRCUIT_BREAKER_ENABLED` | config/config.example.yaml:692 |
| `ml.gemma.circuit_breaker.failure_threshold` | 5 | int | `GEMMA_FAILURE_THRESHOLD` | config/config.example.yaml:693 |
| `ml.gemma.circuit_breaker.recovery_timeout` | 60 | int | `GEMMA_RECOVERY_TIMEOUT` | config/config.example.yaml:694 |
| `ml.gemma.thresholds` | {'deployment_accuracy': 0.72, 'min_samples': 1000} | dict |  | config/config.example.yaml:697 |
| `ml.gemma.thresholds.deployment_accuracy` | 0.72 | float | `GEMMA_DEPLOYMENT_ACCURACY` | config/config.example.yaml:698 |
| `ml.gemma.thresholds.min_samples` | 1000 | int | `GEMMA_MIN_SAMPLES` | config/config.example.yaml:699 |
| `ml.gemma.training` | {'batch_size': 64, 'epochs': 50, 'learning_rate': 0.000133, 'early_stopping_patience': 10} | dict |  | config/config.example.yaml:701 |
| `ml.gemma.training.batch_size` | 64 | int | `GEMMA_BATCH_SIZE` | config/config.example.yaml:702 |
| `ml.gemma.training.epochs` | 50 | int | `GEMMA_EPOCHS` | config/config.example.yaml:703 |
| `ml.gemma.training.learning_rate` | 0.000133 | float | `GEMMA_LEARNING_RATE` | config/config.example.yaml:704 |
| `ml.gemma.training.early_stopping_patience` | 10 | int | `GEMMA_EARLY_STOPPING_PATIENCE` | config/config.example.yaml:705 |
| `ml.gemma.architecture` | {'input_size': 82, 'hidden_size': 55, 'num_layers': 3, 'dropout': 0.3243, 'num_classes': 3} | dict |  | config/config.example.yaml:707 |
| `ml.gemma.architecture.input_size` | 82 | int | `GEMMA_INPUT_SIZE` | config/config.example.yaml:708 |
| `ml.gemma.architecture.hidden_size` | 55 | int | `GEMMA_HIDDEN_SIZE` | config/config.example.yaml:709 |
| `ml.gemma.architecture.num_layers` | 3 | int | `GEMMA_NUM_LAYERS` | config/config.example.yaml:710 |
| `ml.gemma.architecture.dropout` | 0.3243 | float | `GEMMA_DROPOUT` | config/config.example.yaml:711 |
| `ml.gemma.architecture.num_classes` | 3 | int | `GEMMA_NUM_CLASSES` | config/config.example.yaml:712 |
| `ml.gemma.feature_set` | 'gemma_v1' | str | `GEMMA_FEATURE_SET` | config/config.example.yaml:714 |
| `ml.price` | {'min_confidence': 0.66} | dict |  | config/config.example.yaml:719 |
| `ml.price.min_confidence` | 0.66 | float | `ML_PRICE_MIN_CONFIDENCE` | config/config.example.yaml:721 |
| `ml.regime` | {'min_confidence': 0.6} | dict |  | config/config.example.yaml:723 |
| `ml.regime.min_confidence` | 0.6 | float | `ML_REGIME_AIGATE_MIN_CONFIDENCE` | config/config.example.yaml:725 |
| `models` | {'active_bundle': 'artifacts/gemma/final', 'fallback_bundle': 'artifacts/legacy', 'deployment': {'validation_mode': 'strict', 'allow_missing_features': False... | dict |  | config/config.example.yaml:730 |
| `models.active_bundle` | 'artifacts/gemma/final' | str | `ML_ACTIVE_BUNDLE` | config/config.example.yaml:732 |
| `models.fallback_bundle` | 'artifacts/legacy' | str | `ML_FALLBACK_BUNDLE` | config/config.example.yaml:733 |
| `models.deployment` | {'validation_mode': 'strict', 'allow_missing_features': False, 'canary_percentage': 0} | dict |  | config/config.example.yaml:736 |
| `models.deployment.validation_mode` | 'strict' | str | `MODEL_VALIDATION_MODE` | config/config.example.yaml:737 |
| `models.deployment.allow_missing_features` | False | bool | `MODEL_ALLOW_MISSING_FEATURES` | config/config.example.yaml:738 |
| `models.deployment.canary_percentage` | 0 | int | `MODEL_CANARY_PERCENTAGE` | config/config.example.yaml:739 |
| `models.gemma` | {'use_manifest': True, 'shadow_mode': False, 'prefer_quantized': False, 'force_quantized': False} | dict |  | config/config.example.yaml:742 |
| `models.gemma.use_manifest` | True | bool | `MODEL_GEMMA_USE_MANIFEST` | config/config.example.yaml:743 |
| `models.gemma.shadow_mode` | False | bool | `MODEL_GEMMA_SHADOW_MODE` | config/config.example.yaml:744 |
| `models.gemma.prefer_quantized` | False | bool | `MODEL_GEMMA_PREFER_QUANTIZED` | config/config.example.yaml:745 |
| `models.gemma.force_quantized` | False | bool | `MODEL_GEMMA_FORCE_QUANTIZED` | config/config.example.yaml:746 |

## 2. AppConfig Key Snapshot (Raw/Nested/Effektif)
- Source snapshot: APPCONFIG_CONTENTTYPE_JSON_AUDIT.md (value preview column; some values are truncated in source).
- AppConfig ingest: `_load_from_app_config` + `_flatten_to_nested` (no casting) (`src/config/live_trading_config.py:722,815`); env casting happens via `_get_env_overrides` + `_cast_value` (`src/config/live_trading_config.py:251,399`).
| Raw Key | Raw Value (repr) | Content-Type | Nested Path | Effective Location/Type | Source |
| --- | --- | --- | --- | --- | --- |
| `BearishAlphaBot/BINGX_REST_DEBUG` | '1' | `text/plain` | `bingx_rest_debug` | bingx_rest_debug (top-level, str) = '1' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:12 |
| `BearishAlphaBot/CCXT_TIMEOUT_MS` | '10000' | "" | `ccxt_timeout_ms` | ccxt_timeout_ms (top-level, str) = '10000' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:13 |
| `BearishAlphaBot/DEBUG_MODE` | 'false' | `text/plain` | `debug_mode` | debug_mode (top-level, str) = 'false' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:14 |
| `BearishAlphaBot/EXCHANGES` | 'bingx' | `text/plain` | `exchanges` | exchanges (top-level, str) = 'bingx' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:15 |
| `BearishAlphaBot/LOG_LEVEL` | 'INFO' | `text/plain` | `log_level` | log_level (top-level, str) = 'INFO' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:16 |
| `BearishAlphaBot/PYTHONPATH` | '/home/site/wwwroot:/home/site/wwwroot/sr...' | `text/plain` | `pythonpath` | pythonpath (top-level, str) = '/home/site/wwwroot:/home/site/wwwroot/sr...' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:17 |
| `BearishAlphaBot/PYTHONUNBUFFERED` | '1' | `text/plain` | `pythonunbuffered` | pythonunbuffered (top-level, str) = '1' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:18 |
| `BearishAlphaBot/TELEGRAM_CHAT_ID` | '1359128753' | `text/plain` | `telegram_chat_id` | telegram_chat_id (top-level, str) = '1359128753' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:19 |
| `BearishAlphaBot/TICKER_CACHE_TTL_S` | '1.0' | "" | `ticker_cache_ttl_s` | ticker_cache_ttl_s (top-level, str) = '1.0' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:20 |
| `BearishAlphaBot/TICKER_MAX_ATTEMPTS` | '2' | "" | `ticker_max_attempts` | ticker_max_attempts (top-level, str) = '2' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:21 |
| `BearishAlphaBot/TICKER_RETRY_BASE_DELAY_S` | '0.4' | "" | `ticker_retry_base_delay_s` | ticker_retry_base_delay_s (top-level, str) = '0.4' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:22 |
| `BearishAlphaBot/TRADING_DURATION` | '7200' | `text/plain` | `trading_duration` | trading_duration (top-level, str) = '7200' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:23 |
| `BearishAlphaBot/TRADING_MODE` | 'paper' | `text/plain` | `trading_mode` | trading_mode (top-level, str) = 'paper' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:24 |
| `BearishAlphaBot/adaptive_strategies.enable` | 'true' | "" | `adaptive_strategies.enable` | adaptive_strategies.enable (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:25 |
| `BearishAlphaBot/adaptive_strategies.monitoring.enabled` | 'true' | "" | `adaptive_strategies.monitoring.enabled` | adaptive_strategies.monitoring.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:26 |
| `BearishAlphaBot/adaptive_strategies.performance.max_position_multiplier` | '2.0' | "" | `adaptive_strategies.performance.max_position_multiplier` | adaptive_strategies.performance.max_position_multiplier (nested, str) = '2.0' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:27 |
| `BearishAlphaBot/adaptive_strategies.performance.min_position_multiplier` | '0.5' | "" | `adaptive_strategies.performance.min_position_multiplier` | adaptive_strategies.performance.min_position_multiplier (nested, str) = '0.5' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:28 |
| `BearishAlphaBot/adaptive_strategies.performance.min_volatility_for_adjustment` | '0.02' | "" | `adaptive_strategies.performance.min_volatility_for_adjustment` | adaptive_strategies.performance.min_volatility_for_adjustment (nested, str) = '0.02' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:29 |
| `BearishAlphaBot/ml.enabled` | 'true' | "" | `ml.enabled` | ml.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:30 |
| `BearishAlphaBot/ml.features.atr_period` | '14' | "" | `ml.features.atr_period` | ml.features.atr_period (nested, str) = '14' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:31 |
| `BearishAlphaBot/ml.features.bb_period` | '20' | "" | `ml.features.bb_period` | ml.features.bb_period (nested, str) = '20' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:32 |
| `BearishAlphaBot/ml.features.macd_fast` | '12' | "" | `ml.features.macd_fast` | ml.features.macd_fast (nested, str) = '12' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:33 |
| `BearishAlphaBot/ml.features.macd_slow` | '26' | "" | `ml.features.macd_slow` | ml.features.macd_slow (nested, str) = '26' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:34 |
| `BearishAlphaBot/ml.features.momentum_windows` | '5,10,20,50' | "" | `ml.features.momentum_windows` | ml.features.momentum_windows (nested, str) = '5,10,20,50' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:35 |
| `BearishAlphaBot/ml.features.rsi_period` | '14' | "" | `ml.features.rsi_period` | ml.features.rsi_period (nested, str) = '14' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:36 |
| `BearishAlphaBot/ml.features.volatility_windows` | '5,10,20,50' | "" | `ml.features.volatility_windows` | ml.features.volatility_windows (nested, str) = '5,10,20,50' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:37 |
| `BearishAlphaBot/ml.gemma.enabled` | 'true' | "" | `ml.gemma.enabled` | ml.gemma.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:38 |
| `BearishAlphaBot/ml.price.min_confidence` | '0.55' | "" | `ml.price.min_confidence` | ml.price.min_confidence (nested, str) = '0.55' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:39 |
| `BearishAlphaBot/ml.price_prediction.feature_size` | '42' | "" | `ml.price_prediction.feature_size` | ml.price_prediction.feature_size (nested, str) = '42' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:40 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.dropout` | '0.6' | "" | `ml.price_prediction.model_params.lstm.dropout` | ml.price_prediction.model_params.lstm.dropout (nested, str) = '0.6' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:41 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.hidden_size` | '64' | "" | `ml.price_prediction.model_params.lstm.hidden_size` | ml.price_prediction.model_params.lstm.hidden_size (nested, str) = '64' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:42 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.num_layers` | '2' | "" | `ml.price_prediction.model_params.lstm.num_layers` | ml.price_prediction.model_params.lstm.num_layers (nested, str) = '2' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:43 |
| `BearishAlphaBot/ml.price_prediction.timeframes` | '5m,15m' | "" | `ml.price_prediction.timeframes` | ml.price_prediction.timeframes (nested, str) = '5m,15m' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:44 |
| `BearishAlphaBot/ml.regime_prediction.min_confidence_threshold` | '0.6' | "" | `ml.regime_prediction.min_confidence_threshold` | ml.regime_prediction.min_confidence_threshold (nested, str) = '0.6' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:45 |
| `BearishAlphaBot/ml.reinforcement_learning.epsilon_inference` | '0.01' | "" | `ml.reinforcement_learning.epsilon_inference` | ml.reinforcement_learning.epsilon_inference (nested, str) = '0.01' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:46 |
| `BearishAlphaBot/ml.reinforcement_learning.gamma` | '0.95' | "" | `ml.reinforcement_learning.gamma` | ml.reinforcement_learning.gamma (nested, str) = '0.95' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:47 |
| `BearishAlphaBot/ml.reinforcement_learning.hold_confidence_threshold` | '0.60' | "" | `ml.reinforcement_learning.hold_confidence_threshold` | ml.reinforcement_learning.hold_confidence_threshold (nested, str) = '0.60' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:48 |
| `BearishAlphaBot/ml.reinforcement_learning.learning_rate` | '0.00003' | "" | `ml.reinforcement_learning.learning_rate` | ml.reinforcement_learning.learning_rate (nested, str) = '0.00003' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:49 |
| `BearishAlphaBot/ml.reinforcement_learning.ppo_enabled` | 'true' | "" | `ml.reinforcement_learning.ppo_enabled` | ml.reinforcement_learning.ppo_enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:50 |
| `BearishAlphaBot/ml.reinforcement_learning.training_mode` | 'false' | "" | `ml.reinforcement_learning.training_mode` | ml.reinforcement_learning.training_mode (nested, str) = 'false' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:51 |
| `BearishAlphaBot/ml.signal_scoring.min_score_to_trade` | '62' | "" | `ml.signal_scoring.min_score_to_trade` | ml.signal_scoring.min_score_to_trade (nested, str) = '62' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:52 |
| `BearishAlphaBot/ml.signal_scoring.weights.ml_price` | '0.35' | "" | `ml.signal_scoring.weights.ml_price` | ml.signal_scoring.weights.ml_price (nested, str) = '0.35' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:53 |
| `BearishAlphaBot/ml.signal_scoring.weights.risk_reward` | '0.10' | "" | `ml.signal_scoring.weights.risk_reward` | ml.signal_scoring.weights.risk_reward (nested, str) = '0.10' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:54 |
| `BearishAlphaBot/ml.signal_scoring.weights.strategy` | '0.35' | "" | `ml.signal_scoring.weights.strategy` | ml.signal_scoring.weights.strategy (nested, str) = '0.35' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:55 |
| `BearishAlphaBot/models.active_bundle` | 'artifacts/gemma/final' | "" | `models.active_bundle` | models.active_bundle (nested, str) = 'artifacts/gemma/final' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:56 |
| `BearishAlphaBot/pyramiding.enabled` | 'true' | "" | `pyramiding.enabled` | pyramiding.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:57 |
| `BearishAlphaBot/pyramiding.max_layers_per_symbol` | '2' | "" | `pyramiding.max_layers_per_symbol` | pyramiding.max_layers_per_symbol (nested, str) = '2' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:58 |
| `BearishAlphaBot/pyramiding.min_scale_in_distance_pct` | '0.003' | "" | `pyramiding.min_scale_in_distance_pct` | pyramiding.min_scale_in_distance_pct (nested, str) = '0.003' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:59 |
| `BearishAlphaBot/pyramiding.min_scale_in_quality` | '0.65' | "" | `pyramiding.min_scale_in_quality` | pyramiding.min_scale_in_quality (nested, str) = '0.65' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:60 |
| `BearishAlphaBot/pyramiding.min_scale_in_unrealized_pnl_pct` | '0.003' | "" | `pyramiding.min_scale_in_unrealized_pnl_pct` | pyramiding.min_scale_in_unrealized_pnl_pct (nested, str) = '0.003' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:61 |
| `BearishAlphaBot/risk.daily_max_trades` | '8' | "" | `risk.daily_max_trades` | risk.daily_max_trades (nested, str) = '8' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:62 |
| `BearishAlphaBot/risk.equity_usd` | '500' | "" | `risk.equity_usd` | risk.equity_usd (nested, str) = '500' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:63 |
| `BearishAlphaBot/risk.max_notional_pct_per_trade` | '0.25' | "" | `risk.max_notional_pct_per_trade` | risk.max_notional_pct_per_trade (nested, str) = '0.25' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:64 |
| `BearishAlphaBot/risk.max_position_size_pct` | '0.25' | "" | `risk.max_position_size_pct` | risk.max_position_size_pct (nested, str) = '0.25' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:65 |
| `BearishAlphaBot/risk.min_stop_pct` | '0.005' | "" | `risk.min_stop_pct` | risk.min_stop_pct (nested, str) = '0.005' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:66 |
| `BearishAlphaBot/risk.per_trade_risk_pct` | '0.003' | "" | `risk.per_trade_risk_pct` | risk.per_trade_risk_pct (nested, str) = '0.003' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:67 |
| `BearishAlphaBot/risk.position_size_policy` | 'clip' | "" | `risk.position_size_policy` | risk.position_size_policy (nested, str) = 'clip' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:68 |
| `BearishAlphaBot/risk.queue.max_pending_scale_in_per_symbol` | '1' | "" | `risk.queue.max_pending_scale_in_per_symbol` | risk.queue.max_pending_scale_in_per_symbol (nested, str) = '1' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:69 |
| `BearishAlphaBot/risk.rr_dynamic.base_target_rr` | '1.3' | "" | `risk.rr_dynamic.base_target_rr` | risk.rr_dynamic.base_target_rr (nested, str) = '1.3' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:70 |
| `BearishAlphaBot/risk.size_planner_enabled` | 'true' | `text/plain` | `risk.size_planner_enabled` | risk.size_planner_enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:71 |
| `BearishAlphaBot/signals.bypass.enabled` | 'true' | "" | `signals.bypass.enabled` | signals.bypass.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:72 |
| `BearishAlphaBot/signals.bypass.rsi_overbought_threshold` | '88' | "" | `signals.bypass.rsi_overbought_threshold` | signals.bypass.rsi_overbought_threshold (nested, str) = '88' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:73 |
| `BearishAlphaBot/signals.bypass.rsi_oversold_threshold` | '12' | "" | `signals.bypass.rsi_oversold_threshold` | signals.bypass.rsi_oversold_threshold (nested, str) = '12' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:74 |
| `BearishAlphaBot/signals.duplicate_prevention.cooldown_seconds` | '20' | "" | `signals.duplicate_prevention.cooldown_seconds` | signals.duplicate_prevention.cooldown_seconds (nested, str) = '20' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:75 |
| `BearishAlphaBot/signals.duplicate_prevention.min_price_change_pct` | '0.0005' | "" | `signals.duplicate_prevention.min_price_change_pct` | signals.duplicate_prevention.min_price_change_pct (nested, str) = '0.0005' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:76 |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_enabled` | 'true' | "" | `signals.duplicate_prevention.price_delta_bypass_enabled` | signals.duplicate_prevention.price_delta_bypass_enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:77 |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_threshold` | '0.0015' | "" | `signals.duplicate_prevention.price_delta_bypass_threshold` | signals.duplicate_prevention.price_delta_bypass_threshold (nested, str) = '0.0015' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:78 |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_base` | '28' | "" | `signals.oversold_bounce.adaptive_rsi_base` | signals.oversold_bounce.adaptive_rsi_base (nested, str) = '28' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:79 |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_range` | '8' | "" | `signals.oversold_bounce.adaptive_rsi_range` | signals.oversold_bounce.adaptive_rsi_range (nested, str) = '8' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:80 |
| `BearishAlphaBot/signals.oversold_bounce.enable` | 'true' | "" | `signals.oversold_bounce.enable` | signals.oversold_bounce.enable (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:81 |
| `BearishAlphaBot/signals.oversold_bounce.min_rr_ratio` | '1.5' | "" | `signals.oversold_bounce.min_rr_ratio` | signals.oversold_bounce.min_rr_ratio (nested, str) = '1.5' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:82 |
| `BearishAlphaBot/signals.oversold_bounce.rsi_max` | '45' | "" | `signals.oversold_bounce.rsi_max` | signals.oversold_bounce.rsi_max (nested, str) = '45' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:83 |
| `BearishAlphaBot/signals.oversold_bounce.sl_atr_mult` | '1.0' | "" | `signals.oversold_bounce.sl_atr_mult` | signals.oversold_bounce.sl_atr_mult (nested, str) = '1.0' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:84 |
| `BearishAlphaBot/signals.oversold_bounce.tp_atr_mult` | '1.8' | "" | `signals.oversold_bounce.tp_atr_mult` | signals.oversold_bounce.tp_atr_mult (nested, str) = '1.8' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:85 |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_base` | '72' | "" | `signals.short_the_rip.adaptive_rsi_base` | signals.short_the_rip.adaptive_rsi_base (nested, str) = '72' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:86 |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_range` | '8' | "" | `signals.short_the_rip.adaptive_rsi_range` | signals.short_the_rip.adaptive_rsi_range (nested, str) = '8' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:87 |
| `BearishAlphaBot/signals.short_the_rip.enable` | 'true' | "" | `signals.short_the_rip.enable` | signals.short_the_rip.enable (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:88 |
| `BearishAlphaBot/signals.short_the_rip.min_rr_ratio` | '1.5' | "" | `signals.short_the_rip.min_rr_ratio` | signals.short_the_rip.min_rr_ratio (nested, str) = '1.5' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:89 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.enabled` | 'true' | "" | `signals.short_the_rip.mtf_confirmation.enabled` | signals.short_the_rip.mtf_confirmation.enabled (nested, str) = 'true' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:90 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_15m` | 'false' | "" | `signals.short_the_rip.mtf_confirmation.require_15m` | signals.short_the_rip.mtf_confirmation.require_15m (nested, str) = 'false' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:91 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_1h` | 'false' | "" | `signals.short_the_rip.mtf_confirmation.require_1h` | signals.short_the_rip.mtf_confirmation.require_1h (nested, str) = 'false' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:92 |
| `BearishAlphaBot/signals.short_the_rip.rsi_min` | '55' | "" | `signals.short_the_rip.rsi_min` | signals.short_the_rip.rsi_min (nested, str) = '55' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:93 |
| `BearishAlphaBot/signals.short_the_rip.sl_atr_mult` | '1.0' | "" | `signals.short_the_rip.sl_atr_mult` | signals.short_the_rip.sl_atr_mult (nested, str) = '1.0' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:94 |
| `BearishAlphaBot/signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | '50' | "" | `signals.short_the_rip.symbols.btc/usdt:usdt.rsi_threshold` | signals.short_the_rip.symbols.btc/usdt:usdt.rsi_threshold (nested, str) = '50'; canonical signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold = 50 | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:95 |
| `BearishAlphaBot/signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | '50' | "" | `signals.short_the_rip.symbols.eth/usdt:usdt.rsi_threshold` | signals.short_the_rip.symbols.eth/usdt:usdt.rsi_threshold (nested, str) = '50'; canonical signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold = 50 | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:96 |
| `BearishAlphaBot/signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | '50' | "" | `signals.short_the_rip.symbols.sol/usdt:usdt.rsi_threshold` | signals.short_the_rip.symbols.sol/usdt:usdt.rsi_threshold (nested, str) = '50'; canonical signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold = 50 | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:97 |
| `BearishAlphaBot/signals.short_the_rip.tp_atr_mult` | '1.8' | "" | `signals.short_the_rip.tp_atr_mult` | signals.short_the_rip.tp_atr_mult (nested, str) = '1.8' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:98 |
| `BearishAlphaBot/universe.fixed_symbols` | 'BTC/USDT:USDT' | "" | `universe.fixed_symbols` | universe.fixed_symbols (nested, str) = 'BTC/USDT:USDT' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:99 |
| `BearishAlphaBot/volume_analyzer.buckets` | '[[0.0, "LOW"], [4.96, "NORMAL"], [6.23, ...' | `application/json` | `volume_analyzer.buckets` | volume_analyzer.buckets (nested, str) = '[[0.0, "LOW"], [4.96, "NORMAL"], [6.23, ...' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:100 |
| `BearishAlphaBot/websocket.max_streams_per_exchange.bingx` | '10' | "" | `websocket.max_streams_per_exchange.bingx` | websocket.max_streams_per_exchange.bingx (nested, str) = '10' | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:101 |

## 3. Key-Path Alignment Matrisi (MATCH / ALIAS / UNKNOWN)
**Counts**
| Classification | Count |
| --- | --- |
| MATCH | 74 |
| ALIAS | 3 |
| UNKNOWN | 13 |

**Full Alignment Table**
| Raw Key | Classification | Canonical Path | Notes | Source |
| --- | --- | --- | --- | --- |
| `BearishAlphaBot/BINGX_REST_DEBUG` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:12 |
| `BearishAlphaBot/CCXT_TIMEOUT_MS` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:13 |
| `BearishAlphaBot/DEBUG_MODE` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:14 |
| `BearishAlphaBot/EXCHANGES` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:15 |
| `BearishAlphaBot/LOG_LEVEL` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:16 |
| `BearishAlphaBot/PYTHONPATH` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:17 |
| `BearishAlphaBot/PYTHONUNBUFFERED` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:18 |
| `BearishAlphaBot/TELEGRAM_CHAT_ID` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:19 |
| `BearishAlphaBot/TICKER_CACHE_TTL_S` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:20 |
| `BearishAlphaBot/TICKER_MAX_ATTEMPTS` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:21 |
| `BearishAlphaBot/TICKER_RETRY_BASE_DELAY_S` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:22 |
| `BearishAlphaBot/TRADING_DURATION` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:23 |
| `BearishAlphaBot/TRADING_MODE` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:24 |
| `BearishAlphaBot/adaptive_strategies.enable` | MATCH | `adaptive_strategies.enable` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:25 |
| `BearishAlphaBot/adaptive_strategies.monitoring.enabled` | MATCH | `adaptive_strategies.monitoring.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:26 |
| `BearishAlphaBot/adaptive_strategies.performance.max_position_multiplier` | MATCH | `adaptive_strategies.performance.max_position_multiplier` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:27 |
| `BearishAlphaBot/adaptive_strategies.performance.min_position_multiplier` | MATCH | `adaptive_strategies.performance.min_position_multiplier` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:28 |
| `BearishAlphaBot/adaptive_strategies.performance.min_volatility_for_adjustment` | MATCH | `adaptive_strategies.performance.min_volatility_for_adjustment` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:29 |
| `BearishAlphaBot/ml.enabled` | MATCH | `ml.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:30 |
| `BearishAlphaBot/ml.features.atr_period` | MATCH | `ml.features.atr_period` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:31 |
| `BearishAlphaBot/ml.features.bb_period` | MATCH | `ml.features.bb_period` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:32 |
| `BearishAlphaBot/ml.features.macd_fast` | MATCH | `ml.features.macd_fast` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:33 |
| `BearishAlphaBot/ml.features.macd_slow` | MATCH | `ml.features.macd_slow` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:34 |
| `BearishAlphaBot/ml.features.momentum_windows` | MATCH | `ml.features.momentum_windows` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:35 |
| `BearishAlphaBot/ml.features.rsi_period` | MATCH | `ml.features.rsi_period` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:36 |
| `BearishAlphaBot/ml.features.volatility_windows` | MATCH | `ml.features.volatility_windows` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:37 |
| `BearishAlphaBot/ml.gemma.enabled` | MATCH | `ml.gemma.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:38 |
| `BearishAlphaBot/ml.price.min_confidence` | MATCH | `ml.price.min_confidence` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:39 |
| `BearishAlphaBot/ml.price_prediction.feature_size` | MATCH | `ml.price_prediction.feature_size` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:40 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.dropout` | MATCH | `ml.price_prediction.model_params.lstm.dropout` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:41 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.hidden_size` | MATCH | `ml.price_prediction.model_params.lstm.hidden_size` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:42 |
| `BearishAlphaBot/ml.price_prediction.model_params.lstm.num_layers` | MATCH | `ml.price_prediction.model_params.lstm.num_layers` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:43 |
| `BearishAlphaBot/ml.price_prediction.timeframes` | MATCH | `ml.price_prediction.timeframes` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:44 |
| `BearishAlphaBot/ml.regime_prediction.min_confidence_threshold` | MATCH | `ml.regime_prediction.min_confidence_threshold` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:45 |
| `BearishAlphaBot/ml.reinforcement_learning.epsilon_inference` | MATCH | `ml.reinforcement_learning.epsilon_inference` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:46 |
| `BearishAlphaBot/ml.reinforcement_learning.gamma` | MATCH | `ml.reinforcement_learning.gamma` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:47 |
| `BearishAlphaBot/ml.reinforcement_learning.hold_confidence_threshold` | MATCH | `ml.reinforcement_learning.hold_confidence_threshold` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:48 |
| `BearishAlphaBot/ml.reinforcement_learning.learning_rate` | MATCH | `ml.reinforcement_learning.learning_rate` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:49 |
| `BearishAlphaBot/ml.reinforcement_learning.ppo_enabled` | MATCH | `ml.reinforcement_learning.ppo_enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:50 |
| `BearishAlphaBot/ml.reinforcement_learning.training_mode` | MATCH | `ml.reinforcement_learning.training_mode` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:51 |
| `BearishAlphaBot/ml.signal_scoring.min_score_to_trade` | MATCH | `ml.signal_scoring.min_score_to_trade` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:52 |
| `BearishAlphaBot/ml.signal_scoring.weights.ml_price` | MATCH | `ml.signal_scoring.weights.ml_price` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:53 |
| `BearishAlphaBot/ml.signal_scoring.weights.risk_reward` | MATCH | `ml.signal_scoring.weights.risk_reward` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:54 |
| `BearishAlphaBot/ml.signal_scoring.weights.strategy` | MATCH | `ml.signal_scoring.weights.strategy` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:55 |
| `BearishAlphaBot/models.active_bundle` | MATCH | `models.active_bundle` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:56 |
| `BearishAlphaBot/pyramiding.enabled` | MATCH | `pyramiding.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:57 |
| `BearishAlphaBot/pyramiding.max_layers_per_symbol` | MATCH | `pyramiding.max_layers_per_symbol` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:58 |
| `BearishAlphaBot/pyramiding.min_scale_in_distance_pct` | MATCH | `pyramiding.min_scale_in_distance_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:59 |
| `BearishAlphaBot/pyramiding.min_scale_in_quality` | MATCH | `pyramiding.min_scale_in_quality` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:60 |
| `BearishAlphaBot/pyramiding.min_scale_in_unrealized_pnl_pct` | MATCH | `pyramiding.min_scale_in_unrealized_pnl_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:61 |
| `BearishAlphaBot/risk.daily_max_trades` | MATCH | `risk.daily_max_trades` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:62 |
| `BearishAlphaBot/risk.equity_usd` | MATCH | `risk.equity_usd` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:63 |
| `BearishAlphaBot/risk.max_notional_pct_per_trade` | MATCH | `risk.max_notional_pct_per_trade` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:64 |
| `BearishAlphaBot/risk.max_position_size_pct` | MATCH | `risk.max_position_size_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:65 |
| `BearishAlphaBot/risk.min_stop_pct` | MATCH | `risk.min_stop_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:66 |
| `BearishAlphaBot/risk.per_trade_risk_pct` | MATCH | `risk.per_trade_risk_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:67 |
| `BearishAlphaBot/risk.position_size_policy` | MATCH | `risk.position_size_policy` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:68 |
| `BearishAlphaBot/risk.queue.max_pending_scale_in_per_symbol` | MATCH | `risk.queue.max_pending_scale_in_per_symbol` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:69 |
| `BearishAlphaBot/risk.rr_dynamic.base_target_rr` | MATCH | `risk.rr_dynamic.base_target_rr` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:70 |
| `BearishAlphaBot/risk.size_planner_enabled` | MATCH | `risk.size_planner_enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:71 |
| `BearishAlphaBot/signals.bypass.enabled` | MATCH | `signals.bypass.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:72 |
| `BearishAlphaBot/signals.bypass.rsi_overbought_threshold` | MATCH | `signals.bypass.rsi_overbought_threshold` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:73 |
| `BearishAlphaBot/signals.bypass.rsi_oversold_threshold` | MATCH | `signals.bypass.rsi_oversold_threshold` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:74 |
| `BearishAlphaBot/signals.duplicate_prevention.cooldown_seconds` | MATCH | `signals.duplicate_prevention.cooldown_seconds` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:75 |
| `BearishAlphaBot/signals.duplicate_prevention.min_price_change_pct` | MATCH | `signals.duplicate_prevention.min_price_change_pct` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:76 |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_enabled` | MATCH | `signals.duplicate_prevention.price_delta_bypass_enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:77 |
| `BearishAlphaBot/signals.duplicate_prevention.price_delta_bypass_threshold` | MATCH | `signals.duplicate_prevention.price_delta_bypass_threshold` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:78 |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_base` | MATCH | `signals.oversold_bounce.adaptive_rsi_base` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:79 |
| `BearishAlphaBot/signals.oversold_bounce.adaptive_rsi_range` | MATCH | `signals.oversold_bounce.adaptive_rsi_range` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:80 |
| `BearishAlphaBot/signals.oversold_bounce.enable` | MATCH | `signals.oversold_bounce.enable` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:81 |
| `BearishAlphaBot/signals.oversold_bounce.min_rr_ratio` | MATCH | `signals.oversold_bounce.min_rr_ratio` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:82 |
| `BearishAlphaBot/signals.oversold_bounce.rsi_max` | MATCH | `signals.oversold_bounce.rsi_max` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:83 |
| `BearishAlphaBot/signals.oversold_bounce.sl_atr_mult` | MATCH | `signals.oversold_bounce.sl_atr_mult` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:84 |
| `BearishAlphaBot/signals.oversold_bounce.tp_atr_mult` | MATCH | `signals.oversold_bounce.tp_atr_mult` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:85 |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_base` | MATCH | `signals.short_the_rip.adaptive_rsi_base` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:86 |
| `BearishAlphaBot/signals.short_the_rip.adaptive_rsi_range` | MATCH | `signals.short_the_rip.adaptive_rsi_range` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:87 |
| `BearishAlphaBot/signals.short_the_rip.enable` | MATCH | `signals.short_the_rip.enable` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:88 |
| `BearishAlphaBot/signals.short_the_rip.min_rr_ratio` | MATCH | `signals.short_the_rip.min_rr_ratio` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:89 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.enabled` | MATCH | `signals.short_the_rip.mtf_confirmation.enabled` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:90 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_15m` | MATCH | `signals.short_the_rip.mtf_confirmation.require_15m` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:91 |
| `BearishAlphaBot/signals.short_the_rip.mtf_confirmation.require_1h` | MATCH | `signals.short_the_rip.mtf_confirmation.require_1h` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:92 |
| `BearishAlphaBot/signals.short_the_rip.rsi_min` | MATCH | `signals.short_the_rip.rsi_min` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:93 |
| `BearishAlphaBot/signals.short_the_rip.sl_atr_mult` | MATCH | `signals.short_the_rip.sl_atr_mult` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:94 |
| `BearishAlphaBot/signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:95 |
| `BearishAlphaBot/signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:96 |
| `BearishAlphaBot/signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:97 |
| `BearishAlphaBot/signals.short_the_rip.tp_atr_mult` | MATCH | `signals.short_the_rip.tp_atr_mult` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:98 |
| `BearishAlphaBot/universe.fixed_symbols` | MATCH | `universe.fixed_symbols` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:99 |
| `BearishAlphaBot/volume_analyzer.buckets` | MATCH | `volume_analyzer.buckets` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:100 |
| `BearishAlphaBot/websocket.max_streams_per_exchange.bingx` | MATCH | `websocket.max_streams_per_exchange.bingx` | canonical match | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:101 |

**En kritik 10 uyuşmazlık (risk odaklı)**
| Raw Key | Classification | Canonical Path | Notes | Source |
| --- | --- | --- | --- | --- |
| `BearishAlphaBot/signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.BTC/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:95 |
| `BearishAlphaBot/signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.ETH/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:96 |
| `BearishAlphaBot/signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | ALIAS | `signals.short_the_rip.symbols.SOL/USDT:USDT.rsi_threshold` | case mismatch (lowercased by _flatten_to_nested) | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:97 |
| `BearishAlphaBot/DEBUG_MODE` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:14 |
| `BearishAlphaBot/TRADING_MODE` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:24 |
| `BearishAlphaBot/TRADING_DURATION` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:23 |
| `BearishAlphaBot/EXCHANGES` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:15 |
| `BearishAlphaBot/CCXT_TIMEOUT_MS` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:13 |
| `BearishAlphaBot/TICKER_MAX_ATTEMPTS` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:21 |
| `BearishAlphaBot/TICKER_RETRY_BASE_DELAY_S` | UNKNOWN |  | no canonical path in YAML | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:22 |

## 4. Legacy Key Consumption Audit (Dosya/Satır Bazlı)
| Key/Area | Evidence (file:line) | Access Pattern | Canonical/Alias |
| --- | --- | --- | --- |
| `signals.short_the_rip.mtf_confirmation.enabled` | src/core/production_coordinator.py:405; src/core/production_coordinator.py:406 | mtf_cfg = ...; mtf_enabled = bool(mtf_cfg.get("enabled", False)) | canonical (nested) |
| `signals.short_the_rip.mtf_confirmation.require_15m` | src/strategies/adaptive_str.py:422 | require_15m = bool(cfg.get("require_15m", False)) | canonical (nested) |
| `signals.short_the_rip.mtf_confirmation.require_1h` | src/strategies/adaptive_str.py:540 | require_1h = bool(cfg.get("require_1h", False)) | canonical (nested) |
| `pyramiding.enabled` | src/core/risk_manager.py:373; src/core/strategy_coordinator.py:573 | pyramiding_enabled = bool(pyramiding_cfg.get("enabled", False)) | canonical (nested) |
| `strategies.regime_routing.*.preferred_strategies` | src/core/strategy_coordinator.py:405; src/core/strategy_coordinator.py:2763; src/core/strategy_coordinator.py:2786; src/core/strategy_coordinator.py:2788; src/core/strategy_coordinator.py:2794 | raw_rule normalization: str -> [str], list -> list(raw_rule) | canonical (nested) |
| `ml.reinforcement_learning.training_mode` | src/ml/reinforcement_learning.py:248 | self.config.get("training_mode", False) used directly in RL agent | canonical (nested) |
| `risk.per_trade_risk_pct` | src/config/live_trading_config.py:508; src/core/position_sizing.py:305; src/main.py:86 | risk_section.get(...) + float(...) conversions | canonical (nested) + legacy env fallback |
| `signals.*.enable` (strategy toggles) | src/core/production_coordinator.py:660; src/core/production_coordinator.py:690; src/main.py:307; src/main.py:318 | signals_config.get(...).get("enable", True) | canonical (nested) |
| `TRADING_SYMBOLS` | src/core/production_coordinator.py:912 | os.environ.get("TRADING_SYMBOLS") | legacy env-style (non-canonical) |
| `TRADING_MODE` | src/main.py:467 | os.getenv("TRADING_MODE", "paper") | legacy env-style (non-canonical) |
| `debug_mode` | src/core/system_info.py:442; src/core/logger.py:56 | passed as parameter; no cfg.get("debug_mode") usage found | system-level (non-canonical) |
| `ml_rl_training_mode` | rg -n "ml_rl_training_mode" src -> no hits | no direct consumption in codebase | alias not consumed |

## 5. Runtime Evidence (Effective Config + Log Kanıtı)
**Effective Config Snapshot (YAML + AppConfig merge, no coercion)**
| Key | AppConfig Raw (repr) | Effective Value/Type | Notes | Evidence |
| --- | --- | --- | --- | --- |
| `ml.reinforcement_learning.training_mode` | 'false' | 'false' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:51, config/config.example.yaml:604 |
| `ml_rl_training_mode` | n/a | None (null) | not present in AppConfig snapshot |  |
| `signals.short_the_rip.mtf_confirmation.enabled` | 'true' | 'true' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:90, config/config.example.yaml:355 |
| `signals.short_the_rip.mtf_confirmation.require_15m` | 'false' | 'false' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:91, config/config.example.yaml:356 |
| `signals.short_the_rip.mtf_confirmation.require_1h` | 'false' | 'false' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:92, config/config.example.yaml:357 |
| `pyramiding.enabled` | 'true' | 'true' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:57, config/config.example.yaml:454 |
| `strategies.regime_routing.bullish.preferred_strategies` | n/a | '["trend_follower", "breakout_hunter"]' (str) | YAML default only | config/config.example.yaml:404 |
| `strategies.regime_routing.bearish.preferred_strategies` | n/a | '["short_the_rip", "mean_reversion"]' (str) | YAML default only | config/config.example.yaml:409 |
| `strategies.regime_routing.neutral.preferred_strategies` | n/a | '["range_sniper", "mean_reversion"]' (str) | YAML default only | config/config.example.yaml:412 |
| `strategies.regime_routing.volatile.preferred_strategies` | n/a | '["scalper", "volatility_breakout"]' (str) | YAML default only | config/config.example.yaml:415 |
| `debug_mode` | 'false' | 'false' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:14 |
| `risk.per_trade_risk_pct` | '0.003' | '0.003' (str) |  | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:67, config/config.example.yaml:60 |

**Log Evidence: Env Overrides + Risk Normalization**
```text
5: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO - 🔧 Applying overrides from environment variables...
6: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: CAPITAL_USDT = 100 (as int)
7: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: PER_TRADE_RISK_PCT = 0.01 (as float)
8: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: DAILY_MAX_TRADES = 8 (as int)
9: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: WS_MAX_STREAMS_BINGX = 10 (as int)
10: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: TRADING_SYMBOLS = ['BTC/USDT:USDT'] (as list)
11: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: DUPLICATE_PREVENTION_THRESHOLD = 0.0005 (as float)
12: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: DUPLICATE_PREVENTION_COOLDOWN = 20 (as int)
13: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: PRICE_DELTA_BYPASS_THRESHOLD = 0.0015 (as float)
14: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: PRICE_DELTA_BYPASS_ENABLED = True (as bool)
15: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: RSI_THRESHOLD_BTC = 50 (as int)
16: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO -   ✓ Applied ENV: GEMMA_ENABLED = True (as bool)
17: 2025-12-09 15:30:31 - [config.live_trading_config] - INFO - ✅ Risk normalization: per_trade_risk_pct=0.0100 (fraction), computed_max_risk_usd=1.00 USD
```

**Log Evidence: Debug Mode**
```text
436: Operating System:  Debian GNU/Linux 13
437: Mode:              PAPER
438: Dry Run:           NO
439: Debug Mode:        DISABLED
```

**Log Evidence: Pyramiding Settings**
```text
39: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO -    Max Notional Per Trade: 125.00 USDT
40: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO - Pyramiding Settings:
41: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO -    Enabled: True
42: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO -    Max layers per symbol: 2
43: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO -    Min scale-in quality: 0.65 | Min scale-in PnL pct: 0.003 | Min scale-in distance pct: 0.003
44: 2025-12-18 11:08:14 - [config.live_trading_config] - INFO -    Queue max pending scale_in per symbol: 1
```

**Log Evidence: RL training_mode Enforcement**
```text
362: 2025-11-17 02:28:17 - [core.production_coordinator] - WARNING - 🧠 [ML-INIT] training_mode=True detected for RL agent while running paper. Forcing inference mode (set ALLOW_RL_TRAINING_MODE=true to override).
363: 2025-11-17 02:28:17 - [core.production_coordinator] - INFO - 🧠 [ML-INIT] Initializing RL agent with state_size=42
364: 2025-11-17 02:28:17 - [ml.reinforcement_learning] - INFO - 🎯 Epsilon Initialization:
365: 2025-11-17 02:28:17 - [ml.reinforcement_learning] - INFO -    training_mode:      False
```

- MTF behavior logs: `rg --line-number "mtf|MTF" -S --glob "*.log"` returned no matches in current log set.

## 6. Central Type Coercion Tasarımı (Allowlist + Kapsam)
**Zorunlu Kontroller**
- Key-path doğruluğu: canonical dot-path setiyle doğrula; env-style/alias key tespitinde uyarı logla ve override etmeye çalışma.
- Kapsam: bool + numeric + JSON list/dict + comma-separated list birlikte ele alınmalı; ilk iterasyon allowlist ile sınırlandırılmalı.

**Allowlist Taslağı (yüksek riskli canonical path’ler)**
| Canonical Path | Expected Type | Example AppConfig Value | Parse Rule | Source |
| --- | --- | --- | --- | --- |
| `ml.reinforcement_learning.training_mode` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:604 |
| `ml.reinforcement_learning.ppo_enabled` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:576 |
| `signals.short_the_rip.mtf_confirmation.enabled` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:355 |
| `signals.short_the_rip.mtf_confirmation.require_15m` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:356 |
| `signals.short_the_rip.mtf_confirmation.require_1h` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:357 |
| `pyramiding.enabled` | bool | true/false | bool parse (true/false/1/0/yes/no) | config/config.example.yaml:454 |
| `risk.per_trade_risk_pct` | float | 0.003 | float(value) (allow percent -> /100 if >1) | config/config.example.yaml:60 |
| `risk.max_position_size_pct` | float | 0.003 | float(value) (allow percent -> /100 if >1) | config/config.example.yaml:65 |
| `risk.max_notional_pct_per_trade` | float | 0.003 | float(value) (allow percent -> /100 if >1) | config/config.example.yaml:66 |
| `strategies.regime_routing.bullish.preferred_strategies` | str | "value" | string (no coercion) | config/config.example.yaml:404 |
| `strategies.regime_routing.bearish.preferred_strategies` | str | "value" | string (no coercion) | config/config.example.yaml:409 |
| `strategies.regime_routing.neutral.preferred_strategies` | str | "value" | string (no coercion) | config/config.example.yaml:412 |
| `strategies.regime_routing.volatile.preferred_strategies` | str | "value" | string (no coercion) | config/config.example.yaml:415 |
| `volume_analyzer.buckets` | list | [[0.0, "LOW"], [0.3, "NORMAL"]] | json.loads if content-type json; else reject | config/config.example.yaml:279 |
| `universe.fixed_symbols` | str | "value" | string (no coercion) | config/config.example.yaml:241 |
| `ml.features.volatility_windows` | str | "value" | string (no coercion) | config/config.example.yaml:491 |
| `ml.features.momentum_windows` | str | "value" | string (no coercion) | config/config.example.yaml:492 |
| `ml.price_prediction.timeframes` | str | "value" | string (no coercion) | config/config.example.yaml:501 |

**Bilinmeyen Key Politikası**
- Unknown/alias key’lere dokunma; `WARN` logla (key adı, raw value, content-type, label).

**Dry-run Önerisi**
- Coercion öncesi: “hangi alanlar değişecek” raporu üret (raw → parsed), diff çıktısını logla ve opsiyonel olarak dosyaya yaz.

## 7. Risk Matrisi + Uygulama Sırası Önerisi
**Risk Matrisi**
| Area/Key | Wrong Type Scenario | Impact | Severity | Evidence |
| --- | --- | --- | --- | --- |
| `signals.short_the_rip.mtf_confirmation.require_15m/require_1h` | string "false" => bool("false") == True | MTF requirement enforced unintentionally; missed signals | High | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:86-87; src/strategies/adaptive_str.py:422,540 |
| `pyramiding.enabled` | string "false" => bool("false") == True | Scale-in logic and queue limits activate unexpectedly | High | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:61; src/core/risk_manager.py:373 |
| `ml.reinforcement_learning.training_mode` | string "false" => truthy in config.get | RL agent runs in training mode or needs forced override | High | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:44; src/ml/reinforcement_learning.py:248; logs/live_trading_20251117_022805_680296.log:362-365 |
| `signals.short_the_rip.symbols.*.rsi_threshold` | case mismatch => AppConfig override ignored | Per-symbol RSI thresholds stay at YAML defaults | Medium-High | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:95-97; src/config/live_trading_config.py:815; config/config.example.yaml:390-397 |
| `strategies.regime_routing.*.preferred_strategies` | JSON string or comma list not parsed | Preferred strategies treated as single string; routing bias lost | Medium | config/config.example.yaml:402-415; src/core/strategy_coordinator.py:2786 |
| `volume_analyzer.buckets` | JSON string not parsed to list | Volume analyzer disables context (invalid buckets) | Medium | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:99; src/core/volume_analyzer.py:181-185 |
| `risk.per_trade_risk_pct` | string non-numeric => float() fails | Risk sizing defaults or fails; inconsistent risk limits | Medium | APPCONFIG_CONTENTTYPE_JSON_AUDIT.md:58; src/core/position_sizing.py:305 |

**En kritik 5 düzeltme hedefi (uygulama sırası)**
- 1) `ml.reinforcement_learning.training_mode` string → bool coercion (merging sonrası; `src/config/live_trading_config.py` içinde `_load_and_merge_configs` sonrası merkezi adım).
- 2) `signals.short_the_rip.mtf_confirmation.require_15m/require_1h` string → bool coercion + MTF log visibility (kullanım: `src/strategies/adaptive_str.py`).
- 3) `pyramiding.enabled` string → bool coercion (kullanım: `src/core/risk_manager.py`, `src/core/strategy_coordinator.py`).
- 4) `strategies.regime_routing.*.preferred_strategies` JSON/list parsing (kullanım: `src/core/strategy_coordinator.py:2786`).
- 5) `signals.short_the_rip.symbols.*.rsi_threshold` case-mismatch fix (AppConfig key normalization; `_flatten_to_nested` lowercasing → symbol key mapping).
