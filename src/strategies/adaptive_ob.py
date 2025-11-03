"""
Adaptive OversoldBounce strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
import math
from typing import Optional, Dict
from .oversold_bounce import OversoldBounce

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
    
    def __init__(self, cfg: Dict, regime_analyzer=None):
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
        self.base_cfg = cfg.copy()
        self.debug_logging = self.base_cfg.get('debug', {}).get('strategy_logging', False)

        # Minimum R/R oranını başlangıçta, bir kez olmak üzere config'den oku.
        # `super()` çağrısı `self.strategy_config`'i oluşturduğu için artık bunu güvenle kullanabiliriz.
        self.min_rr_ratio = self.strategy_config.get('min_rr_ratio', 1.2)
        logger.info(f"[{self.strategy_name.upper()}] Minimum R/R Ratio initialized to: {self.min_rr_ratio}")

    def _validate_input_data(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, regime_data: Dict, symbol: str) -> tuple[bool, str]:
        """Gerekli tüm verilerin varlığını ve geçerliliğini kontrol eder."""
        # 1. Ana DataFrame Kontrolü
        if df_30m is None or df_30m.empty:
            return False, "Input data 'df_30m' is missing or empty."
            
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
        """
        Dynamic RSI thresholds based on market conditions.
        Now respects config values and uses gentler adjustments.
        
        Args:
            market_regime: Dictionary with 'trend', 'momentum', 'volatility'
            
        Returns:
            Adaptive RSI threshold for oversold detection
        """
        # Get config values with proper fallbacks
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base', 
                         self.base_cfg.get('rsi_max', 45)))
        
        # Get adjustment range from config (default ±10)
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 10))
        
        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        
        # Start with base value
        threshold = base_rsi
        
        # Gentler adjustments based on regime
        if trend == 'bullish':
            # In uptrends, be slightly more selective
            if momentum == 'strong':
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        elif trend == 'bearish':
            # In downtrends, be slightly more aggressive
            if momentum == 'strong':
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        # Clamp to reasonable range (never below 30 or above 50)
        min_threshold = max(30, base_rsi - adapt_range)
        max_threshold = min(50, base_rsi + adapt_range)
        
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
            return base_multiplier * 0.75
        
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
            last = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.info(f"🚫 {log_prefix} No Signal: Insufficient 30m data to generate a signal.")
            return None
        
        if regime_data is None:
            regime_data = DEFAULT_MARKET_REGIME.copy()
        
        try:
            # --- Initial Data Extraction ---
            if 'rsi' not in last.index or 'close' not in last.index:
                logger.warning(f"🚫 {log_prefix} No Signal: 'rsi' or 'close' column missing in the latest data.")
                return None
            
            close_price = float(last['close'])
            rsi_val = float(last['rsi'])
            ema_fast = float(last.get('ema_fast', 0))
            
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            # --- Adaptive Threshold Calculation ---
            adaptive_rsi_threshold = self.get_symbol_specific_threshold(symbol_display)
            if adaptive_rsi_threshold is not None:
                if self.debug_logging: logger.info(f"ℹ️ {log_prefix} Using symbol-specific RSI threshold: {adaptive_rsi_threshold:.2f}")
            else:
                adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)

            # --- Core Signal Condition Checks with Tracing ---
            if self.debug_logging:
                logger.info(f"🔍 {log_prefix} Checking conditions...")
                logger.info(f"  - Regime: {market_regime['trend']}, Volatility: {market_regime['volatility']}")
                logger.info(f"  - Price: ${close_price:,.2f}, RSI: {rsi_val:.2f}, EMA Fast: ${ema_fast:,.2f}")
                logger.info(f"  - RSI Threshold: {adaptive_rsi_threshold:.2f}")

            # 1. RSI Condition Check
            if rsi_val > adaptive_rsi_threshold:
                logger.info(f"🚫 {log_prefix} No Signal: RSI ({rsi_val:.2f}) is above the threshold ({adaptive_rsi_threshold:.2f}).")
                return None

            # 2. Price vs. EMA Condition Check
            if ema_fast > 0 and close_price >= ema_fast:
                logger.info(f"🚫 {log_prefix} No Signal: Price (${close_price:,.2f}) is not below the fast EMA (${ema_fast:,.2f}).")
                return None

            # 3. Volume Check
            if 'volume' in last.index and float(last['volume']) <= 0:
                logger.info(f"🚫 {log_prefix} No Signal: Volume is zero or negative.")
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
                    return None
                
                if ml_context.get('price_direction') == 'down' and ml_context.get('price_confidence', 0) > 0.7:
                    logger.info(f"🚫 {log_prefix} No Signal: ML VETO - Strong price down prediction (confidence: {ml_context.get('price_confidence', 0):.2%}).")
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
            
            entry_price = float(last['close'])
            atr_value = float(last.get('atr', entry_price * 0.02))
            
            # 🔥 GÜNCELLEME: Config okumaları artık tutarlı bir şekilde `self.strategy_config` üzerinden yapılıyor.
            tp_atr_mult = float(self.strategy_config.get("tp_atr_mult", 2.5))
            sl_atr_mult = float(self.strategy_config.get("sl_atr_mult", 1.2))
            
            target_price = entry_price + (atr_value * tp_atr_mult)
            stop_price = entry_price - (atr_value * sl_atr_mult)
            
            min_tp_pct = float(self.strategy_config.get("min_tp_pct", 0.008))
            max_sl_pct = float(self.strategy_config.get("max_sl_pct", 0.015))
            
            target_price = max(target_price, entry_price * (1 + min_tp_pct))
            stop_price = min(stop_price, entry_price * (1 - max_sl_pct))
            
            rr_numerator = target_price - entry_price
            rr_denominator = entry_price - stop_price
            rr_ratio = (rr_numerator / rr_denominator) if rr_denominator > 0 else 0

            if self.debug_logging:
                logger.info(
                    f"🔍 {log_prefix} R/R Calculation: "
                    f"Entry={entry_price:.2f}, TP={target_price:.2f}, SL={stop_price:.2f} | "
                    f"Reward=${rr_numerator:.2f}, Risk=${rr_denominator:.2f} -> R/R={rr_ratio:.2f}"
                )

            # 4. R/R Ratio Check
            # 🔥 GÜNCELLEME: R/R kontrolü artık __init__ içinde ayarlanan `self.min_rr_ratio` özelliğini kullanıyor.
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"🚫 {log_prefix} No Signal: Calculated R/R Ratio ({rr_ratio:.2f}) is below the minimum required ({self.min_rr_ratio}).")
                return None

            # --- Signal Generation ---
            logger.info(f"✅ {log_prefix} All checks passed. Generating BUY signal.")
            
            signal = {
                "strategy_name": self.strategy_name, "side": "buy", "symbol": symbol,
                "entry": entry_price, "stop": stop_price, "target": target_price,
                "reason": f"Adaptive RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}",
                "rr_ratio": rr_ratio, "is_adaptive": True, "position_multiplier": position_mult,
                "ml_enhanced": ml_enhanced, "strategy_type": 'adaptive'
            }
            
            if ml_enhanced:
                signal['ml_consensus'] = ml_context.get('consensus_score')
                signal['ml_position_modifier'] = position_size_modifier
            
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
