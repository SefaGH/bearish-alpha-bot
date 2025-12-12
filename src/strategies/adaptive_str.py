"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Tuple, ClassVar, Any
from .short_the_rip import ShortTheRip

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


class AdaptiveShortTheRip(ShortTheRip):
    """
    Market regime-aware ShortTheRip strategy.
    
    Adapts RSI thresholds, position sizing, and EMA requirements
    based on real-time market regime analysis.
    """
    
    # Maximum adjustment to base threshold (in RSI points)
    MAX_THRESHOLD_ADJUSTMENT = 5

    DEFAULT_VOLATILITY_STOP_CFG: ClassVar[Dict[str, Any]] = {
        'enabled': True,
        'min_sl_pct': 0.0025,
        'max_sl_pct': 0.02,
        'overrides': {
            'low': {
                'atr_scale': 0.75,
                'min_sl_pct': 0.0015,
                'max_sl_pct': 0.015,
            },
            'normal': {
                'atr_scale': 1.0,
            },
            'high': {
                'atr_scale': 1.15,
                'min_sl_pct': 0.003,
                'max_sl_pct': 0.025,
            },
        },
    }
    
    def __init__(self, cfg: Dict, regime_analyzer=None):
        """
        Initialize adaptive ShortTheRip strategy.
        
        Args:
            cfg: Strategy configuration dictionary
            regime_analyzer: MarketRegimeAnalyzer instance for regime detection
            
        🔥 GÜNCELLEME: Kalıtım zinciri `super().__init__(cfg)` çağrısıyla onarıldı.
        Minimum R/R oranı artık başlangıçta config'den okunup `self.min_rr_ratio` olarak saklanıyor.
        """
        # `ShortTheRip` sınıfının __init__ metodunu çağırarak doğru kurulumu sağla.
        super().__init__(cfg)
        
        # Üst sınıf "short_the_rip" olarak ayarlayabileceği için, bu alt sınıfın adını yeniden tanımla.
        self.strategy_name = "adaptive_str"
        
        self.regime_analyzer = regime_analyzer
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)
        
        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")

        self.volatility_stop_cfg = self._build_volatility_stop_cfg(self.strategy_config)

    def _validate_input_data(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, regime_data: Dict, symbol: str) -> tuple[bool, str]:
        """Gerekli tüm verilerin varlığını ve geçerliliğini kontrol eder."""
        if df_30m is None or df_30m.empty:
            return False, "Input data 'df_30m' is missing or empty."
        
        required_cols = ['close', 'rsi', 'atr', 'ema_fast', 'ema21', 'ema50', 'ema200']
        missing_cols = [col for col in required_cols if col not in df_30m.columns]
        if missing_cols:
            return False, f"df_30m is missing required indicator columns: {missing_cols}."
            
        if self.debug_logging:
            if df_1h is None or df_1h.empty:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'df_1h' is missing. Market regime analysis may be less accurate.")
            if regime_data is None:
                logger.info(f"[{self.strategy_name.upper()}-INFO] {symbol} - 'regime_data' is missing. Strategy will fallback to non-adaptive mode.")

        return True, "All required data is present."

    def _build_volatility_stop_cfg(self, strategy_config: Dict) -> Dict:
        overrides = dict(self.DEFAULT_VOLATILITY_STOP_CFG)
        user_cfg = strategy_config.get('volatility_stop') or {}
        merged = {
            **overrides,
            **{k: v for k, v in user_cfg.items() if k not in ('overrides',)},
        }
        merged_overrides = dict(self.DEFAULT_VOLATILITY_STOP_CFG.get('overrides', {}))
        merged_overrides.update(user_cfg.get('overrides', {}) or {})
        merged['overrides'] = merged_overrides

        # Respect global max_sl_pct if user didn't override explicitly.
        if 'max_sl_pct' not in merged or merged['max_sl_pct'] is None:
            merged['max_sl_pct'] = strategy_config.get('max_sl_pct', self.DEFAULT_VOLATILITY_STOP_CFG['max_sl_pct'])

        if 'min_sl_pct' not in merged or merged['min_sl_pct'] is None:
            merged['min_sl_pct'] = min(
                merged['max_sl_pct'],
                self.DEFAULT_VOLATILITY_STOP_CFG['min_sl_pct'],
            )

        return merged
    
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
    
    def get_adaptive_rsi_threshold(self, market_regime: Dict, *, return_reason: bool = False):
        """Sniper-mode SELL signals only trigger on stretched pumps."""
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base', 68.0))
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 8.0))

        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        threshold = base_rsi
        adjustment_reason = None

        if trend == 'bearish':
            if momentum == 'strong':
                threshold = base_rsi - 3.0
                adjustment_reason = "Bearish trend with strong momentum: lowering RSI threshold"
            else:
                threshold = base_rsi
        elif trend == 'bullish':
            if momentum == 'strong':
                threshold = base_rsi + 7.0  # Target ~75
                adjustment_reason = "Bullish trend with strong momentum: demanding higher RSI"
            else:
                threshold = base_rsi + 3.0  # Target ~71
                adjustment_reason = "Bullish trend: raising RSI threshold"

        min_threshold = max(60, base_rsi - adapt_range)
        max_threshold = min(90, base_rsi + adapt_range)
        clamped_threshold = max(min_threshold, min(max_threshold, threshold))

        if adjustment_reason is None and clamped_threshold != base_rsi:
            adjustment_reason = "Adaptive RSI clamp applied"

        if not return_reason:
            return clamped_threshold

        return clamped_threshold, {
            'base_threshold': base_rsi,
            'reason': adjustment_reason,
            'trend': trend,
            'momentum': momentum
        }
    
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
            return base_multiplier * 0.75
        
        # Low volatility: Can increase position size slightly
        elif volatility_regime == 'low':
            return base_multiplier * 1.25
        
        # Normal volatility: Use base multiplier
        else:
            return base_multiplier
    
    def adapt_ema_requirements(self, trend_strength: float) -> Dict[str, Any]:
        """
        EMA alignment requirements based on trend strength.
        
        Args:
            trend_strength: Trend strength metric (0.0 to 1.0)
            
        Returns:
            Dictionary with EMA requirement parameters
        """
        # Strong trends: Require strict EMA alignment
        if trend_strength > 0.7:
            return {
                'require_strict_ema_align': True,
                'ema_tolerance': 0.001  # 0.1% tolerance
            }
        
        # Weak trends: Relax EMA requirements
        elif trend_strength < 0.3:
            return {
                'require_strict_ema_align': False,
                'ema_tolerance': 0.01  # 1% tolerance
            }
        
        # Moderate trends: Standard requirements
        else:
            return {
                'require_strict_ema_align': True,
                'ema_tolerance': 0.005  # 0.5% tolerance
            }

    def _apply_volatility_stop_tuning(self, volatility: str, base_sl_pct: float, atr_pct: float) -> Tuple[float, Dict[str, float]]:
        cfg = self.volatility_stop_cfg or {}
        if not cfg.get('enabled', False):
            return base_sl_pct, {}

        overrides = cfg.get('overrides', {})
        vol_cfg = overrides.get(volatility, {})

        atr_scale = float(vol_cfg.get('atr_scale', 1.0))
        tuned_sl_pct = base_sl_pct * atr_scale

        min_sl_pct = float(vol_cfg.get('min_sl_pct', cfg.get('min_sl_pct', base_sl_pct)))
        max_sl_pct = float(vol_cfg.get('max_sl_pct', cfg.get('max_sl_pct', base_sl_pct)))

        strategy_cap = float(self.strategy_config.get('max_sl_pct', max_sl_pct))
        max_sl_pct = min(max_sl_pct, strategy_cap)

        tuned_sl_pct = max(min_sl_pct, min(max_sl_pct, tuned_sl_pct))

        metadata = {
            'volatility': volatility,
            'atr_pct': atr_pct,
            'base_sl_pct': base_sl_pct,
            'min_sl_pct': min_sl_pct,
            'max_sl_pct': max_sl_pct,
            'applied_scale': atr_scale,
            'final_sl_pct': tuned_sl_pct,
        }
        return tuned_sl_pct, metadata
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None,
               symbol: str = None,
               market_data: Optional[Dict] = None,
               ml_context=None) -> Optional[Dict]:
        """
        Generate adaptive trading signal for a short position based on market regime and ML insights.
        Logs the specific reason if no signal is generated for transparency.
        """
        symbol_display = symbol or "UNKNOWN"
        log_prefix = f"[{self.strategy_name.upper()}/{symbol_display}]"
        # Volume logic centralized in StrategyCoordinator (Issue #450)

        # --- Data Validation Step ---
        validation_passed, reason = self._validate_input_data(df_30m, df_1h, regime_data, symbol_display)
        if not validation_passed:
            logger.info(f"🚫 {log_prefix} No Signal: {reason}")
            return None

        try:
            last30 = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal.")
            return None

        if regime_data is None:
            regime_data = DEFAULT_MARKET_REGIME.copy()

        try:
            signal = {
                "strategy_name": self.strategy_name,
                "side": "sell",
                "symbol": symbol,
                "features": {}
            }
            # --- Initial Data Extraction ---
            if 'rsi' not in last30.index or 'close' not in last30.index:
                logger.warning(f"🚫 {log_prefix} No Signal: 'rsi' or 'close' column missing in the latest data.")
                return None
            
            close_price = float(last30['close'])
            rsi_val = float(last30['rsi'])
            
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            # --- Adaptive Threshold Calculation ---
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                base_threshold = float(self.base_cfg.get('adaptive_rsi_base', 68.0))
                logger.info(
                    f"⚠️ {log_prefix} Symbol-specific RSI threshold active: base {base_threshold:.2f} → {adaptive_rsi_threshold:.2f}."
                )
            else:
                adaptive_rsi_threshold, threshold_meta = self.get_adaptive_rsi_threshold(
                    market_regime,
                    return_reason=True
                )
                meta_base = threshold_meta.get('base_threshold', adaptive_rsi_threshold)
                reason = threshold_meta.get('reason')
                if reason and abs(adaptive_rsi_threshold - meta_base) >= 0.1:
                    logger.info(
                        f"⚠️ {log_prefix} {reason} (base {meta_base:.2f} → {adaptive_rsi_threshold:.2f})."
                    )
                elif abs(adaptive_rsi_threshold - meta_base) >= 0.1:
                    logger.info(
                        f"⚠️ {log_prefix} Adaptive RSI adjustment applied (base {meta_base:.2f} → {adaptive_rsi_threshold:.2f})."
                    )
                else:
                    logger.info(
                        f"ℹ️ {log_prefix} RSI threshold steady at {adaptive_rsi_threshold:.2f} (trend={market_regime['trend']}, momentum={market_regime['momentum']})."
                    )

            # --- Core Signal Condition Checks with Tracing ---
            if self.debug_logging:
                # Calculate EMA alignment for debug log
                ema21 = float(last30.get('ema21', 0))
                ema50 = float(last30.get('ema50', 0))
                ema200 = float(last30.get('ema200', 0))
                ema_aligned = (ema21 < ema50 <= ema200)
                
                atr_val = float(last30.get('atr', 0))
                vol = float(last30.get('volume', 0))
                
                logger.info(f"🔍 [STR-DEBUG] {symbol_display}")
                logger.info(f"   RSI: {rsi_val:.2f} (threshold: {adaptive_rsi_threshold:.2f})")
                logger.info(f"   EMA Align: {ema_aligned} (21={ema21:.1f}, 50={ema50:.1f}, 200={ema200:.1f})")
                logger.info(f"   Volume: {vol:.2f}")
                logger.info(f"   ATR: {atr_val:.4f}")
                logger.info(f"   Regime: {market_regime['trend']}, Volatility: {market_regime['volatility']}")

            # 1. RSI Condition Check
            if rsi_val < adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is below the threshold ({adaptive_rsi_threshold:.2f}).")
                return None

            # 2. Dynamic Rip Check (Replaces strict EMA alignment)
            # "Rip" is defined as price extending significantly above a Long-Term EMA
            
            # Get Long-Term EMA (using EMA 50 as proxy for trend baseline)
            long_ema_value = float(last30.get('ema50', 0))
            atr_value = float(last30.get('atr', 0))
            
            if long_ema_value > 0 and atr_value > 0:
                # Determine ATR Multiplier based on volatility
                volatility_regime = market_regime.get('volatility', 'normal')
                
                if volatility_regime == 'high':
                    atr_multiplier = 1.5  # High vol: require more aggressive extension
                elif volatility_regime == 'low':
                    atr_multiplier = 0.75 # Low vol: smaller extension is enough
                else:
                    atr_multiplier = 1.0  # Normal
                
                # Calculate Rip Threshold
                rip_value = atr_value * atr_multiplier
                required_price_threshold = long_ema_value + rip_value
                
                if close_price < required_price_threshold:
                    logger.info(
                        f"🚫 {log_prefix} No Signal: Rip Check Failed. Price ${close_price:,.2f} "
                        f"is not above EMA50+Rip (${required_price_threshold:,.2f}). "
                        f"(EMA50=${long_ema_value:,.2f}, Rip=${rip_value:,.2f}, Vol={volatility_regime})"
                    )
                    return None
                else:
                    if self.debug_logging:
                        logger.info(
                            f"✅ {log_prefix} Rip Check Passed: Price ${close_price:,.2f} > "
                            f"Threshold ${required_price_threshold:,.2f} (Rip=${rip_value:,.2f})"
                        )
            else:
                # Fallback if indicators missing (should be caught by validation, but safe guard)
                logger.warning(f"⚠️ {log_prefix} Missing EMA50 or ATR for Rip Check. Skipping.")
            
            logger.info(f"✅ {log_prefix} Base conditions met. Proceeding to ML & Risk checks.")

            # --- ML-Aware Decision Making ---
            position_size_modifier = 1.0
            signal["rsi"] = float(rsi_val)
            signal.setdefault("features", {})["rsi"] = float(rsi_val)
            ml_enhanced = False
            MIN_ML_CONFIDENCE_THRESHOLD = 0.60

            if ml_context and ml_context.get('is_healthy', False) and ml_context.get('regime_confidence', 0) >= MIN_ML_CONFIDENCE_THRESHOLD:
                ml_enhanced = True
                
                if ml_context.get('regime_prediction') == 'bullish' and ml_context.get('regime_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong bullish regime detected (confidence: {ml_context.get('regime_confidence', 0):.2%}).")
                    return None
                
                if ml_context.get('price_direction') == 'up' and ml_context.get('price_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price up prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
                    return None
                
                if (ml_context.get('regime_prediction') == 'bearish') or (ml_context.get('price_direction') == 'down'):
                    position_size_modifier = 1.0 + (0.25 * ml_context.get('consensus_score', 0))
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Confirmation: Increasing position size modifier to {position_size_modifier:.2f}x.")
                elif ml_context.get('consensus_score', 1.0) < 0.5:
                    position_size_modifier *= 0.75
                    if self.debug_logging: logger.info(f"🧠 {log_prefix} ML Caution: Reducing position size modifier to {position_size_modifier:.2f}x.")

            # --- Final Risk/Reward and Position Sizing ---
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility) * position_size_modifier
            
            entry_price = float(last30['close'])
            atr_value = float(last30.get('atr', entry_price * 0.02))
            atr_pct = (atr_value / entry_price) if entry_price else 0.0
            
            # 🔥 GÜNCELLEME: Config okumaları artık tutarlı bir şekilde `self.strategy_config` üzerinden yapılıyor.
            tp_atr_mult = float(self.strategy_config.get("tp_atr_mult", 3.0))
            sl_atr_mult = float(self.strategy_config.get("sl_atr_mult", 1.5))
            
            # Calculate theoretical ATR-based levels
            theoretical_sl_distance = atr_value * sl_atr_mult
            theoretical_tp_distance = atr_value * tp_atr_mult
            
            theoretical_sl_pct = theoretical_sl_distance / entry_price
            theoretical_tp_pct = theoretical_tp_distance / entry_price
            
            min_tp_pct = float(self.strategy_config.get("min_tp_pct", 0.010))
            max_sl_pct = float(self.strategy_config.get("max_sl_pct", 0.020))
            
            intended_rr = tp_atr_mult / sl_atr_mult if sl_atr_mult else self.min_rr_ratio

            # Apply stop-loss cap CORRECTLY and realign TP if needed
            if theoretical_sl_pct > max_sl_pct:
                logger.info(f"📊 {log_prefix} [SL Cap Applied] {theoretical_sl_pct:.1%} → {max_sl_pct:.1%}")
                actual_sl_pct = max_sl_pct
            else:
                actual_sl_pct = theoretical_sl_pct

            volatility_stop_meta = None
            if self.volatility_stop_cfg.get('enabled', False):
                tuned_sl_pct, volatility_stop_meta = self._apply_volatility_stop_tuning(
                    volatility,
                    actual_sl_pct,
                    atr_pct,
                )
                if abs(tuned_sl_pct - actual_sl_pct) >= 1e-6:
                    logger.info(
                        f"⚙️ {log_prefix} [VolStop] {volatility.capitalize()} volatility adjusted SL {actual_sl_pct:.1%} → {tuned_sl_pct:.1%}."
                    )
                actual_sl_pct = tuned_sl_pct

            scaled_tp_pct = actual_sl_pct * intended_rr
            actual_tp_pct = max(min_tp_pct, min(theoretical_tp_pct, scaled_tp_pct))
            
            # SHORT: Stop is ABOVE entry
            stop_price = entry_price * (1 + actual_sl_pct)
            
            # ✅ CRITICAL FIX: Use min() for SHORT to prevent stop going too high
            # This safety net ensures we never exceed max_sl_pct even with rounding errors
            # or edge cases in the logic above (defensive programming for financial systems)
            stop_price = min(stop_price, entry_price * (1 + max_sl_pct))
            
            # SHORT: Target is BELOW entry
            target_price = entry_price * (1 - actual_tp_pct)
            target_price = min(target_price, entry_price * (1 - min_tp_pct))
            
            # Calculate final R/R
            risk_amount = stop_price - entry_price
            reward_amount = entry_price - target_price
            rr_ratio = (reward_amount / risk_amount) if risk_amount > 0 else float('inf')

            # Enhanced logging with R/R details
            logger.info(
                f"🔎 {log_prefix} [STR R/R] SHORT Entry=${entry_price:.2f}, "
                f"Stop=${stop_price:.2f} (+{actual_sl_pct:.1%}), "
                f"Target=${target_price:.2f} (-{actual_tp_pct:.1%}), "
                f"R/R={rr_ratio:.2f}"
            )    

            # 3. R/R Ratio Check
            # 🔥 GÜNCELLEME: R/R kontrolü artık __init__ içinde ayarlanan `self.min_rr_ratio` özelliğini kullanıyor.
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"🚫 {log_prefix} No Signal: Calculated R/R Ratio ({rr_ratio:.2f}) is below the minimum required ({self.min_rr_ratio}).")
                return None

            # --- Signal Generation ---
            logger.info(f"✅ {log_prefix} All checks passed. Generating SELL signal.")
            
            signal.update({
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "reason": f"Adaptive RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}",
                "rr_ratio": rr_ratio,
                "is_adaptive": True,
                "position_multiplier": position_mult,
                "ml_enhanced": ml_enhanced,
                "strategy_type": 'adaptive',
                "strategy_min_rr": self.min_rr_ratio,
            })
            if volatility_stop_meta:
                signal['volatility_stop_meta'] = volatility_stop_meta
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier
            
            return signal
            
        except Exception as e:
            logger.error(f"💥 {log_prefix} Critical error during signal generation: {e}", exc_info=True)
            # Fallback logic remains unchanged
            try:
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m, df_1h)
                    if base_signal:
                        base_signal.update({'strategy_type': 'base_fallback', 'fallback_reason': str(e), 'symbol': symbol})
                        logger.warning(f"⚠️ {log_prefix} Fallback to base strategy successful.")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"💥 {log_prefix} Fallback to base strategy also failed: {fallback_error}")
                
        return None
    
    def get_strategy_state(self) -> Dict:
        """
        Get current strategy state and parameters.
        
        Returns:
            Dictionary with current adaptive parameters
        """
        return {
            'strategy': 'adaptive_short_the_rip',
            'base_config': self.base_cfg,
            'has_regime_analyzer': self.regime_analyzer is not None
        }
