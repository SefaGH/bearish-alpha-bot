"""
Adaptive OversoldBounce strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
import math
import time
from typing import Optional, Dict, Any
from .oversold_bounce import OversoldBounce
from core.strategy_shadow_eval import (
    shadow_enabled,
    extract_last_closed_ts_ms,
    extract_df_meta,
    emit_shadow_log,
)
from core.data_validator import TIMEFRAME_SECONDS
from core.indicators import rsi

# Default market regime for fallback
DEFAULT_MARKET_REGIME = {
    'trend': 'neutral',
    'momentum': 'sideways',
    'volatility': 'normal',
    'micro_trend_strength': 0.5,
    'entry_score': 0.5,
    'risk_multiplier': 1.0
}

logger = logging.getLogger(__name__)


class AdaptiveOversoldBounce(OversoldBounce):
    """
    Market regime-aware OversoldBounce strategy.
    
    Adapts RSI thresholds, position sizing, and EMA requirements
    based on real-time market regime analysis.
    """
    
    # Maximum adjustment to base threshold (in RSI points)
    MAX_THRESHOLD_ADJUSTMENT = 5
    
    def __init__(self, cfg: Dict, regime_analyzer=None, market_data_pipeline=None):
        """
        Initialize adaptive OversoldBounce strategy.
        
        Args:
            cfg: Strategy configuration dictionary
            regime_analyzer: MarketRegimeAnalyzer instance for regime detection
        
        🔥 GÜNCELLEME: Kalıtım zinciri `super().__init__(cfg)` çağrısıyla onarıldı.
        Minimum R/R oranı artık başlangıçta config'den okunup `self.min_rr_ratio` olarak saklanıyor.
        """
        # `OversoldBounce` sınıfının __init__ metodunu çağırarak doğru kurulumu sağla.
        super().__init__(cfg)
        
        # Üst sınıf "oversold_bounce" olarak ayarlayabileceği için, bu alt sınıfın adını yeniden tanımla.
        self.strategy_name = "adaptive_ob"
        
        self.regime_analyzer = regime_analyzer
        self.market_data_pipeline = market_data_pipeline
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)
        self._persistency_state: Dict[str, Dict[str, float]] = {}
        self._persistency_skip_log_ts: Dict[str, float] = {}
        self._persistency_cfg = {
            "mode": "time",
            "seconds": 5.0,
            "min_samples": 2,
            "wick_closeness_k": 0.25,
        }

        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")
        # Interface guard
        if not hasattr(self, "signal"):
            self.signal = self._default_signal_wrapper  # type: ignore
        assert callable(getattr(self, "signal", None)), f"{self.strategy_name}: signal method not callable"

    def _validate_input_data(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, regime_data: Dict, symbol: str) -> tuple[bool, str]:
        """Gerekli tüm verilerin varlığını ve geçerliliğini kontrol eder."""
        # 1. Ana DataFrame Kontrolü
        if df_30m is None or df_30m.empty:
            return False, "Input data 'df_30m' is missing or empty."
        if self.debug_logging:
            logger.debug(f"[{self.strategy_name.upper()}] Validation data lengths: total_rows={len(df_30m)}")
            
        # 2. Zorunlu Sütunların Kontrolü
        required_cols = ['close', 'rsi', 'atr', 'ema_fast']
        missing_cols = [col for col in required_cols if col not in df_30m.columns]
        if missing_cols:
            return False, f"df_30m is missing required indicator columns: {missing_cols}."
            
        # 3. Yüksek Önemli "Adaptive" Verilerin Kontrolü (Uyarı Niteliğinde)
        if self.debug_logging:
            if df_1h is None or df_1h.empty:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'df_1h' is missing. Market regime analysis may be less accurate.")
            if regime_data is None:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'regime_data' is missing. Strategy will fallback to non-adaptive mode.")

        return True, "All required data is present."
        
    def get_adaptive_rsi_threshold(self, market_regime: Dict) -> float:
        """Sniper-mode RSI floors force patience for deep oversold prints."""
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base', 32.0))
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 8.0))

        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        threshold = base_rsi
        if trend == 'bullish':
            if momentum == 'strong':
                threshold = base_rsi + 2.0
            else:
                threshold = base_rsi
        elif trend == 'bearish':
            if momentum == 'strong':
                threshold = base_rsi - 5.0  # Target ~27
            else:
                threshold = base_rsi - 2.0  # Target ~30

        min_threshold = max(20, base_rsi - adapt_range)
        max_threshold = min(40, base_rsi + adapt_range)
        return max(min_threshold, min(max_threshold, threshold))
    
    def calculate_dynamic_position_size(self, volatility_regime: str, 
                                       base_multiplier: float = 1.0) -> float:
        """
        Volatility-adjusted position sizing multiplier.
        
        Args:
            volatility_regime: 'high', 'normal', or 'low'
            base_multiplier: Base position size multiplier
            
        Returns:
            Adjusted position size multiplier
        """
        # High volatility: Reduce position size for risk management
        if volatility_regime == 'high':
            return base_multiplier * 0.50
        
        # Low volatility: Can increase position size slightly
        elif volatility_regime == 'low':
            return base_multiplier * 1.25
        
        # Normal volatility: Use base multiplier
        else:
            return base_multiplier
    
    def adapt_ema_distances(self, trend_strength: float) -> Dict[str, float]:
        """
        EMA distance requirements based on trend strength.
        
        Args:
            trend_strength: Trend strength metric (0.0 to 1.0)
            
        Returns:
            Dictionary with EMA distance multipliers
        """
        # Strong trends: Require larger EMA distances (more confirmation)
        if trend_strength > 0.7:
            return {
                'ema_distance_mult': 1.5,
                'require_ema_separation': True
            }
        
        # Weak trends: Smaller EMA distances acceptable
        elif trend_strength < 0.3:
            return {
                'ema_distance_mult': 0.7,
                'require_ema_separation': False
            }
        
        # Moderate trends: Standard requirements
        else:
            return {
                'ema_distance_mult': 1.0,
                'require_ema_separation': False
            }
    
    def get_symbol_specific_threshold(self, symbol: str) -> Optional[float]:
        """
        Get symbol-specific RSI threshold override if configured.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            
        Returns:
            Symbol-specific threshold or None
        """
        if not symbol:
            return None
        
        # Check if symbol-specific config exists
        symbols_cfg = self.base_cfg.get('symbols', {})
        if symbol in symbols_cfg:
            symbol_cfg = symbols_cfg[symbol]
            if 'rsi_threshold' in symbol_cfg:
                return float(symbol_cfg['rsi_threshold'])
        
        return None
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None,
               symbol: str = None,
               market_data: Optional[Dict] = None,
               ml_context=None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime and ML insights.
        Logs the specific reason if no signal is generated.
        """
        symbol_display = symbol or "UNKNOWN"
        log_prefix = f"[{self.strategy_name.upper()}/{symbol_display}]"

        # --- Data Validation Step ---
        validation_passed, reason = self._validate_input_data(df_30m, df_1h, regime_data, symbol_display)
        if not validation_passed:
            logger.info(f"🚫 {log_prefix} No Signal: {reason}")
            return None
        
        try:
            includes_forming = bool(df_30m.attrs.get("includes_forming", False))
            forming_open_ms_attr = df_30m.attrs.get("forming_ts")

            fallback_reason = df_30m.attrs.get("fallback_reason", None)
            # Backwards-compat: older pipeline versions stored the string "none".
            if isinstance(fallback_reason, str) and fallback_reason.strip().lower() in ("none", ""):
                fallback_reason = None

            forming_last_update_ts = df_30m.attrs.get("forming_last_update_ts")
            forming_update_age_ms = df_30m.attrs.get("forming_update_age_ms")

            df_closed = df_30m
            df_used = df_30m
            forming_row = None
            forming_open_ms = None
            used_forming = False
            rsi_source = "closed"
            trigger_price_source = "closed_close"

            if includes_forming and len(df_30m) >= 2:
                df_closed = df_30m.iloc[:-1]
                forming_row = df_30m.iloc[-1]
                try:
                    forming_open_ms = int(forming_open_ms_attr or int(forming_row.name.timestamp() * 1000))
                except Exception:
                    forming_open_ms = None
            else:
                includes_forming = False
                forming_open_ms = None
                df_closed = df_30m

            # Strategy should only *use* forming data when it exists AND pipeline indicates no fallback.
            used_forming = bool(includes_forming and fallback_reason is None)
            if used_forming:
                df_used = df_30m
                rsi_source = "live"
                trigger_price_source = "forming_close"
            else:
                df_used = df_closed
                rsi_source = "closed"
                trigger_price_source = "closed_close"

            if self.debug_logging:
                logger.debug(f"{log_prefix} Data sufficiency check: total_rows={len(df_closed)}")

            try:
                # Always use the last CLOSED candle for indicator context (EMA/ATR/etc).
                trend_row = df_closed.iloc[-1]
            except IndexError:
                logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal (IndexError).")
                return None

            required_cols = ['close', 'rsi', 'atr', 'ema_fast']
            missing = [c for c in required_cols if c not in trend_row.index]
            if missing:
                logger.warning(f"{log_prefix} Missing required columns in latest closed row: {missing}")
                return None
            if any(pd.isna(trend_row[c]) for c in required_cols):
                logger.info(f"{log_prefix} Latest closed row has NaN in required columns; skipping this tick.")
                return None

            if regime_data is None:
                regime_data = DEFAULT_MARKET_REGIME.copy()

            # --- Initial Data Extraction ---
            if 'close' not in trend_row.index:
                logger.warning(f"🚫 {log_prefix} No Signal: 'close' column missing in the latest data.")
                return None

            if used_forming:
                rsi_series = rsi(df_used['close'])
            elif 'rsi' in df_closed.columns:
                rsi_series = df_closed['rsi']
            else:
                rsi_series = rsi(df_closed['close'])

            rsi_val = float(rsi_series.iloc[-1])
            close_price = float(trend_row['close'])
            forming_price = float(forming_row['close']) if forming_row is not None else close_price
            forming_low = float(forming_row['low']) if forming_row is not None and 'low' in forming_row else None
            forming_high = float(forming_row['high']) if forming_row is not None and 'high' in forming_row else None
            ema_fast = float(trend_row.get('ema_fast', 0))
            ema_mid = float(trend_row.get('ema_mid', 0))
            atr_value = float(trend_row.get('atr', close_price * 0.02))
            volume_val = None
            if forming_row is not None and 'volume' in forming_row:
                volume_val = float(forming_row['volume'])
            elif 'volume' in trend_row:
                volume_val = float(trend_row['volume'])

            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }

            # Hardening (Issue #454):
            # - Never log "Hybrid fallback: none"
            # - Only emit "reverting to closed-only" when we explicitly chose NOT to use forming
            #   AND the pipeline provided a real fallback reason.
            if (not used_forming) and (fallback_reason is not None):
                if fallback_reason == "pivot_grace_prev_bucket":
                    logger.info(
                        f"{log_prefix} Hybrid downgrade: pivot_grace_prev_bucket. "
                        "Previous bucket still updating within grace window; using closed-only for safety."
                    )
                else:
                    logger.warning(
                        f"{log_prefix} Hybrid fallback: {fallback_reason}. Reverting to closed-only data."
                    )

            trigger_cfg_source = str(self.base_cfg.get("adaptive_ob_trigger_price_source", "mid")).lower()
            resolved_trigger_source = trigger_cfg_source
            trigger_fallback_chain = "none"
            trigger_price = forming_price
            if used_forming:
                if self.market_data_pipeline:
                    trigger_price, resolved_trigger_source, trigger_fallback_chain = self.market_data_pipeline.get_live_trigger_price(
                        symbol=symbol_display,
                        timeframe="30m",
                        source=trigger_cfg_source,
                        forming_close=forming_price,
                    )
                if trigger_price is None:
                    trigger_price = forming_price
                    resolved_trigger_source = "forming_close"
                    trigger_fallback_chain = f"{trigger_fallback_chain}->forming_close_none" if trigger_fallback_chain != "none" else "forming_close_none"
            else:
                resolved_trigger_source = "closed_close"
                trigger_price = close_price
                trigger_fallback_chain = "closed_only"

            trigger_price_source = resolved_trigger_source

            trigger_price_log = f"{trigger_price:.2f}" if trigger_price is not None else "None"

            if includes_forming or self.debug_logging:
                logger.info(
                    f"{log_prefix} Hybrid meta | includes_forming={includes_forming} "
                    f"used_forming={used_forming} "
                    f"forming_open_time={forming_open_ms} rsi_source={rsi_source} "
                    f"trigger_price_source={resolved_trigger_source} fallback_reason={(fallback_reason if fallback_reason is not None else 'none')} "
                    f"forming_last_update_ts={forming_last_update_ts} forming_update_age_ms={forming_update_age_ms} "
                    f"fallback_chain={trigger_fallback_chain} trigger_price={trigger_price_log}"
                )

            persist_mode = str(self.base_cfg.get("adaptive_ob_persistency_mode", "time")).lower()
            persist_seconds = float(self.base_cfg.get("adaptive_ob_persistency_seconds", 5))
            persist_min_samples = int(self.base_cfg.get("adaptive_ob_persistency_min_samples", 2))
            persist_wick_k = float(self.base_cfg.get("adaptive_ob_wick_closeness_k", 0.25))
            self._persistency_cfg = {
                "mode": persist_mode,
                "seconds": max(persist_seconds, 0.0),
                "min_samples": max(persist_min_samples, 1),
                "wick_closeness_k": max(min(persist_wick_k, 1.0), 0.0),
            }
            
            # --- Adaptive Threshold Calculation ---
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                if self.debug_logging: logger.info(f"ℹ️ {log_prefix} Using symbol-specific RSI threshold: {adaptive_rsi_threshold:.2f}")
            else:
                adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)

            # Trend confirmation: if market is in a strong downswing (ema_fast < ema_mid), demand deeper RSI
            ema_trend_penalty = float(self.strategy_config.get('trend_confirmation_rsi_penalty', 5.0))
            min_adaptive_rsi = float(self.strategy_config.get('trend_confirmation_min_rsi', 8.0))
            trend_bias_active = False
            if ema_fast > 0 and ema_mid > 0 and ema_fast < ema_mid:
                new_threshold = max(min_adaptive_rsi, adaptive_rsi_threshold - ema_trend_penalty)
                if new_threshold != adaptive_rsi_threshold:
                    logger.info(
                        f"⚠️ {log_prefix} Trend confirmation active: EMA fast ${ema_fast:,.2f} below EMA mid ${ema_mid:,.2f}."
                        f" Adjusting RSI threshold {adaptive_rsi_threshold:.2f} → {new_threshold:.2f}."
                    )
                    adaptive_rsi_threshold = new_threshold
                    trend_bias_active = True

            # Volume logic centralized in StrategyCoordinator (Issue #450)
            # Legacy volume confirmation removed.

            timeframe = "30m"
            last_closed_ts_ms = extract_last_closed_ts_ms(df_closed)
            df_meta = extract_df_meta(df_closed)

            def _shadow_ob(decision: str, fail_reason: str = "", extra: Optional[Dict] = None) -> None:
                if not shadow_enabled():
                    return
                try:
                    payload = {
                        "event": "strategy_shadow_eval",
                        "strategy": "adaptive_ob",
                        "symbol": symbol_display,
                        "timeframe": timeframe,
                        "last_closed_ts": last_closed_ts_ms,
                        **df_meta,
                        "close": trigger_price,
                        "rsi": rsi_val,
                        "rsi_threshold": adaptive_rsi_threshold,
                        "rsi_delta": rsi_val - adaptive_rsi_threshold if adaptive_rsi_threshold is not None else None,
                        "ema_fast": ema_fast,
                        "ema_mid": ema_mid,
                        "atr": atr_value,
                        "includes_forming": includes_forming,
                        "used_forming": used_forming,
                        "forming_open_time": forming_open_ms,
                        "forming_last_update_ts": forming_last_update_ts,
                        "forming_update_age_ms": forming_update_age_ms,
                        "rsi_source": rsi_source,
                        "trigger_price_source": trigger_price_source,
                        "fallback_reason": fallback_reason,
                        "trend_penalty_applied": trend_bias_active,
                        "trend_penalty": ema_trend_penalty,
                        "decision": decision,
                        "fail_reason": fail_reason,
                    }
                    if extra:
                        payload.update(extra)
                    emit_shadow_log(logger, payload, "adaptive_ob", symbol_display, timeframe, last_closed_ts_ms)
                except Exception:
                    return

            # --- Core Signal Condition Checks with Tracing ---
            if self.debug_logging:
                logger.info(f"🔍 {log_prefix} Checking conditions...")
                logger.info(f"  - Regime: {market_regime['trend']}, Volatility: {market_regime['volatility']}")
                logger.info(f"  - Price: ${trigger_price:,.2f}, RSI: {rsi_val:.2f}, EMA Fast: ${ema_fast:,.2f}, EMA Mid: ${ema_mid:,.2f}")
                logger.info(f"  - RSI Threshold: {adaptive_rsi_threshold:.2f}")

            # 1. RSI Condition Check
            if rsi_val > adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is above the threshold ({adaptive_rsi_threshold:.2f}).")
                _shadow_ob("no_signal_rsi", "rsi_above_threshold")
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "rsi_above_threshold",
                    includes_forming,
                    forming_open_ms,
                    {
                        "rsi": f"{rsi_val:.2f}",
                        "threshold": f"{adaptive_rsi_threshold:.2f}",
                    },
                )
                return None

            # 2. Price vs. EMA Condition Check
            if ema_fast > 0 and trigger_price >= ema_fast:
                logger.info(f"🚫 {log_prefix} No Signal: Price (${trigger_price:,.2f}) is not below the fast EMA (${ema_fast:,.2f}).")
                _shadow_ob(
                    "no_signal_price_vs_ema",
                    "price_not_below_ema_fast",
                    {"price_vs_ema_fast": trigger_price - ema_fast},
                )
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "price_not_below_ema_fast",
                    includes_forming,
                    forming_open_ms,
                    {
                        "price": f"{trigger_price:.2f}",
                        "ema_fast": f"{ema_fast:.2f}",
                    },
                )
                return None

            # 3. Volume Check (basic data sanity)
            if volume_val is not None and volume_val <= 0:
                logger.info(f"🚫 {log_prefix} No Signal: Volume is zero or negative.")
                _shadow_ob(
                    "no_signal_volume",
                    "non_positive_volume",
                    {"volume": volume_val},
                )
                self._log_persistency_skipped(
                    symbol_display,
                    log_prefix,
                    "non_positive_volume",
                    includes_forming,
                    forming_open_ms,
                    {"volume": volume_val},
                )
                return None
            
            # --- Persistency Guard ---
            guard_passed, persist_meta = self._apply_persistency_guard(
                symbol_display,
                includes_forming,
                forming_open_ms,
                rsi_val,
                adaptive_rsi_threshold,
                trigger_price,
                ema_fast,
                forming_price,
                forming_low,
                forming_high,
            )

            logger.info(
                f"{log_prefix} Persistency | mode={persist_meta['mode']} condition_true={persist_meta['condition_true']} "
                f"elapsed_s={persist_meta['elapsed_s']:.2f} samples={persist_meta['samples']} passed={guard_passed} "
                f"wick_close_to_low={persist_meta.get('wick_closeness')} k={persist_meta.get('wick_closeness_k')} "
                f"wick_passed={persist_meta.get('wick_closeness_passed')} "
                f"wick_skipped={persist_meta.get('wick_closeness_skipped')}"
            )
            if not guard_passed:
                _shadow_ob(
                    "no_signal_persistency",
                    "persistency_not_met",
                    persist_meta,
                )
                return None

            logger.info(f"✅ {log_prefix} Base conditions met. Proceeding to ML & Risk checks.")

            # --- ML-Aware Decision Making ---
            position_size_modifier = 1.0
            ml_enhanced = False
            MIN_ML_CONFIDENCE_THRESHOLD = 0.60

            if ml_context and ml_context.get('is_healthy', False) and ml_context.get('regime_confidence', 0) >= MIN_ML_CONFIDENCE_THRESHOLD:
                ml_enhanced = True
                
                if ml_context.get('regime_prediction') == 'bearish' and ml_context.get('regime_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong bearish regime detected (confidence: {ml_context.get('regime_confidence', 0):.2%}).")
                    _shadow_ob(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_regime": ml_context.get('regime_prediction'),
                            "ml_confidence": ml_context.get('regime_confidence'),
                        },
                    )
                    return None
                
                if ml_context.get('price_direction') == 'down' and ml_context.get('price_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price down prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
                    _shadow_ob(
                        "no_signal_ml",
                        "ml_veto",
                        {
                            "ml_price_direction": ml_context.get('price_direction'),
                            "ml_price_confidence": ml_context.get('price_confidence'),
                        },
                    )
                    return None
                
                if (ml_context.get('regime_prediction') == 'bullish') or (ml_context.get('price_direction') == 'up'):
                    position_size_modifier = 1.0 + (0.25 * ml_context.get('consensus_score', 0))
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Confirmation: Increasing position size modifier to {position_size_modifier:.2f}x.")
                elif ml_context.get('consensus_score', 1.0) < 0.5:
                    position_size_modifier *= 0.75
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Caution: Reducing position size modifier to {position_size_modifier:.2f}x.")

            # --- Final Risk/Reward and Position Sizing ---
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility) * position_size_modifier
            
            entry_price = float(trigger_price)
            
            # 🔥 GÜNCELLEME: Config okumaları artık tutarlı bir şekilde `self.strategy_config` üzerinden yapılıyor.
            tp_atr_mult = float(self.strategy_config.get("tp_atr_mult", 2.5))
            sl_atr_mult = float(self.strategy_config.get("sl_atr_mult", 1.2))
            
            # Calculate theoretical ATR-based levels
            theoretical_sl_distance = atr_value * sl_atr_mult
            theoretical_tp_distance = atr_value * tp_atr_mult
            
            theoretical_sl_pct = theoretical_sl_distance / entry_price
            theoretical_tp_pct = theoretical_tp_distance / entry_price
            
            min_tp_pct = float(self.strategy_config.get("min_tp_pct", 0.008))
            max_sl_pct = float(self.strategy_config.get("max_sl_pct", 0.015))
            
            # Apply stop-loss cap CORRECTLY and realign TP if needed
            if theoretical_sl_pct > max_sl_pct:
                # Risk needs capping
                logger.info(f"📊 {log_prefix} [SL Cap Applied] {theoretical_sl_pct:.1%} → {max_sl_pct:.1%}")
                actual_sl_pct = max_sl_pct
                actual_sl_distance = entry_price * actual_sl_pct
                
                # CRITICAL - Realign TP to maintain intended R/R ratio
                intended_rr = tp_atr_mult / sl_atr_mult  # e.g., 2.5/1.2 = 2.08
                adjusted_tp_distance = actual_sl_distance * intended_rr
                actual_tp_pct = adjusted_tp_distance / entry_price
                
                logger.info(f"📊 {log_prefix} [TP Realigned] Maintaining R/R={intended_rr:.2f}")
            else:
                # No capping needed
                actual_sl_pct = theoretical_sl_pct
                actual_tp_pct = theoretical_tp_pct
            
            # Calculate final prices
            stop_price = entry_price * (1 - actual_sl_pct)
            
            # ✅ CRITICAL FIX: Use max() for LONG positions to prevent stop going too low
            # This safety net ensures we never exceed max_sl_pct even with rounding errors
            # or edge cases in the logic above (defensive programming for financial systems)
            stop_price = max(stop_price, entry_price * (1 - max_sl_pct))
            
            target_price = entry_price * (1 + actual_tp_pct)
            
            # Ensure minimum TP
            target_price = max(target_price, entry_price * (1 + min_tp_pct))
            
            # Calculate final R/R
            rr_numerator = target_price - entry_price
            rr_denominator = entry_price - stop_price
            rr_ratio = (rr_numerator / rr_denominator) if rr_denominator > 0 else 0

            # Enhanced logging with R/R details
            logger.info(
                f"🔎 {log_prefix} [OB R/R] Entry=${entry_price:.2f}, "
                f"Stop=${stop_price:.2f} (-{actual_sl_pct:.1%}), "
                f"Target=${target_price:.2f} (+{actual_tp_pct:.1%}), "
                f"R/R={rr_ratio:.2f}"
            )

            # 4. R/R Ratio Check
            # 🔥 GÜNCELLEME: R/R kontrolü artık __init__ içinde ayarlanan `self.min_rr_ratio` özelliğini kullanıyor.
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"🚫 {log_prefix} No Signal: Calculated R/R Ratio ({rr_ratio:.2f}) is below the minimum required ({self.min_rr_ratio}).")
                _shadow_ob(
                    "no_signal_rr",
                    "rr_below_min",
                    {
                        "rr_ratio": rr_ratio,
                        "min_rr_required": self.min_rr_ratio,
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                    },
                )
                return None

            # --- Signal Generation ---
            logger.info(f"✅ {log_prefix} All checks passed. Generating BUY signal.")
            
            signal = {
                "strategy_name": self.strategy_name, "side": "buy", "symbol": symbol,
                "entry": entry_price, "stop": stop_price, "target": target_price,
                "reason": f"Adaptive RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}",
                "rr_ratio": rr_ratio, "is_adaptive": True, "position_multiplier": position_mult,
                "ml_enhanced": ml_enhanced, "strategy_type": 'adaptive',
                "strategy_min_rr": self.min_rr_ratio,  # NEW: Strategy's own minimum R/R
            }

            # Expose RSI telemetry so downstream duplicate logic can react dynamically
            signal["rsi"] = float(rsi_val)
            signal.setdefault("features", {})["rsi"] = float(rsi_val)

            meta = signal.setdefault('meta', {})
            meta.update(
                {
                    'includes_forming': includes_forming,
                    'forming_open_time': forming_open_ms,
                    'rsi_source': rsi_source,
                    'trigger_price_source': trigger_price_source,
                    'fallback_reason': fallback_reason,
                }
            )
            if trend_bias_active:
                meta['trend_confirmation'] = {
                    'ema_fast': ema_fast,
                    'ema_mid': ema_mid,
                    'rsi_threshold': adaptive_rsi_threshold,
                }
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier

            _shadow_ob(
                "pass",
                "",
                {
                    "rr_ratio": rr_ratio,
                    "min_rr_required": self.min_rr_ratio,
                    "position_multiplier": position_mult,
                },
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"💥 {log_prefix} Critical error during signal generation: {e}", exc_info=True)
            # Fallback logic remains unchanged
            try:
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m)
                    if base_signal:
                        base_signal.update({'strategy_type': 'base_fallback', 'fallback_reason': str(e), 'symbol': symbol})
                        logger.warning(f"⚠️ {log_prefix} Fallback to base strategy successful.")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"💥 {log_prefix} Fallback to base strategy also failed: {fallback_error}")
                
        return None

    def _persistency_key(self, symbol: str) -> str:
        return f"30m::{symbol}"

    def _reset_persistency(self, symbol: str) -> None:
        try:
            self._persistency_state.pop(self._persistency_key(symbol), None)
        except Exception:
            return

    def _log_persistency_skipped(
        self,
        symbol: str,
        log_prefix: str,
        reason: str,
        includes_forming: bool,
        forming_open_ms: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now_ts = time.time()
        throttle_key = f"{symbol}:{reason}"
        last_ts = self._persistency_skip_log_ts.get(throttle_key, 0.0)
        if (now_ts - last_ts) < 60.0:
            return
        self._persistency_skip_log_ts[throttle_key] = now_ts

        cfg = getattr(self, "_persistency_cfg", {}) or {}
        mode = str(cfg.get("mode", "time"))
        seconds = float(cfg.get("seconds", 5.0))
        min_samples = int(cfg.get("min_samples", 2))
        wick_k = float(cfg.get("wick_closeness_k", 0.25))

        state = self._persistency_state.get(self._persistency_key(symbol)) or {}
        first_ts = state.get("first_true_ts")
        samples = state.get("samples")
        bucket = state.get("bucket")

        state_bits = []
        if first_ts is not None:
            state_bits.append(f"state_first_ts={first_ts:.0f}")
        if samples is not None:
            state_bits.append(f"state_samples={samples}")
        if bucket is not None:
            state_bits.append(f"state_bucket={bucket}")
        state_suffix = " " + " ".join(state_bits) if state_bits else ""

        extra_bits = ""
        if extra:
            extra_bits = " " + " ".join(f"{k}={v}" for k, v in extra.items())

        logger.debug(
            f"{log_prefix} PersistencySkipped | reason={reason} mode={mode} seconds={seconds:.2f} "
            f"min_samples={min_samples} wick_k={wick_k:.2f} includes_forming={includes_forming} "
            f"bucket={forming_open_ms}{state_suffix}{extra_bits}"
        )

    def _apply_persistency_guard(
        self,
        symbol: str,
        includes_forming: bool,
        forming_open_ms: Optional[int],
        rsi_val: float,
        adaptive_rsi_threshold: float,
        trigger_price: float,
        entry_threshold: float,
        forming_close: Optional[float],
        forming_low: Optional[float],
        forming_high: Optional[float],
    ) -> tuple[bool, Dict[str, Any]]:
        cfg = getattr(self, "_persistency_cfg", {"mode": "time", "seconds": 5.0, "min_samples": 1})
        mode = str(cfg.get("mode", "time")).lower()
        seconds = max(float(cfg.get("seconds", 5.0)), 0.0)
        min_samples = max(int(cfg.get("min_samples", 1)), 1)
        wick_k = max(min(float(cfg.get("wick_closeness_k", 0.25)), 1.0), 0.0)
        eps = 1e-9

        meta = {
            "mode": mode,
            "includes_forming": includes_forming,
            "forming_open_time": forming_open_ms,
            "condition_true": False,
            "elapsed_s": 0.0,
            "samples": 0,
            "threshold_seconds": seconds,
            "threshold_samples": min_samples,
            "trigger_price": trigger_price,
            "entry_threshold": entry_threshold,
            "forming_close": forming_close,
            "forming_low": forming_low,
            "forming_high": forming_high,
            "wick_closeness": None,
            "wick_closeness_k": wick_k,
            "wick_closeness_passed": None,
            "wick_closeness_skipped": False,
            "wick_closeness_reason": None,
            "bucket_changed": False,
        }

        if mode == "off":
            self._reset_persistency(symbol)
            meta["condition_true"] = True
            meta["elapsed_s"] = 0.0
            meta["samples"] = 0
            return True, meta

        if not includes_forming:
            self._reset_persistency(symbol)
            meta["condition_true"] = True
            return True, meta

        key = self._persistency_key(symbol)

        had_state = key in self._persistency_state
        state = self._persistency_state.get(
            key, {"first_true_ts": None, "samples": 0, "bucket": forming_open_ms}
        )

        prev_bucket = state.get("bucket")
        bucket_changed = had_state and (prev_bucket != forming_open_ms)

        if bucket_changed:
            state = {"first_true_ts": None, "samples": 0, "bucket": forming_open_ms}

        condition_true = False
        if mode == "bar_low_and_close":
            if includes_forming and forming_low is not None and forming_close is not None and forming_close > 0:
                # Canonical wick filter
                dip_occurred = forming_low <= entry_threshold
                no_snap_back = trigger_price <= entry_threshold
                if forming_high is None or pd.isna(forming_high) or pd.isna(forming_low):
                    meta["wick_closeness_skipped"] = True
                    meta["wick_closeness_reason"] = "missing_high_or_low"
                    closeness = None
                    closeness_pass = True  # fall back to dip + no_snap_back only
                    meta["wick_closeness_passed"] = closeness_pass
                meta = {
                    "mode": mode,
                    "includes_forming": includes_forming,
                    "forming_open_time": forming_open_ms,
                    "condition_true": False,
                    "elapsed_s": 0.0,
                    "samples": 0,
                    "threshold_seconds": seconds,
                    "threshold_samples": min_samples,
                    "trigger_price": trigger_price,
                    "entry_threshold": entry_threshold,
                    "forming_close": forming_close,
                    "forming_low": forming_low,
                    "forming_high": forming_high,
                    "wick_closeness": None,
                    "wick_closeness_k": wick_k,
                    "wick_closeness_passed": None,
                    "wick_closeness_skipped": False,
                    "wick_closeness_reason": None,
                    "bucket_changed": False,
                    "prev_bucket": prev_bucket,
                }

                meta["prev_bucket"] = prev_bucket
                meta["bucket_changed"] = bucket_changed
            self._reset_persistency(symbol)
            meta["bucket_changed"] = bucket_changed
            return False, meta

        now_ts = time.time()
        if state.get("first_true_ts") is None:
            state["first_true_ts"] = now_ts
            state["samples"] = 1
        else:
            state["samples"] = int(state.get("samples", 0)) + 1
        elapsed = max(0.0, now_ts - float(state.get("first_true_ts", now_ts)))
        passed = (elapsed >= seconds) and (state["samples"] >= min_samples)

        state["bucket"] = forming_open_ms
        self._persistency_state[key] = state

        meta.update(
            {
                "condition_true": True,
                "elapsed_s": elapsed,
                "samples": state["samples"],
                "bucket_changed": bucket_changed,
            }
        )
        return passed, meta
    
    def get_strategy_state(self) -> Dict:
        """
        Get current strategy state and parameters.
        
        Returns:
            Dictionary with current adaptive parameters
        """
        return {
            'strategy': 'adaptive_oversold_bounce',
            'base_config': self.base_cfg,
            'has_regime_analyzer': self.regime_analyzer is not None
        }
