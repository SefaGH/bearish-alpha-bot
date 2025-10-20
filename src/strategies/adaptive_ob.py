"""
Adaptive OversoldBounce strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
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
        """
        super().__init__(cfg)
        self.regime_analyzer = regime_analyzer
        self.base_cfg = cfg.copy()
        
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
            return base_multiplier * 0.5
        
        # Low volatility: Can increase position size slightly
        elif volatility_regime == 'low':
            return base_multiplier * 1.5
        
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
               symbol: str = None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime.
        
        Args:
            df_30m: 30-minute OHLCV dataframe with indicators
            df_1h: Optional 1-hour OHLCV dataframe (for compatibility)
            regime_data: Optional market regime data for adaptation
                        If None, creates default regime data with neutral settings
            symbol: Symbol name for debug logging
        
        Returns:
            Signal dictionary or None
        """
        # Log symbol for debugging multi-symbol trading
        symbol_display = symbol or "UNKNOWN"
        logger.info(f"[OB-DEBUG] {symbol_display}")
        
        # Data validation
        if df_30m is None or df_30m.empty:
            logger.info(f"  ❌ No data available")
            return None
        
        # Safely get last row
        try:
            last = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.info(f"  ❌ Insufficient 30m data")
            return None
        
        # If no regime data provided, use default neutral regime
        if regime_data is None:
            regime_data = DEFAULT_MARKET_REGIME.copy()
        
        try:
            # Ensure we have valid data with critical columns
            if 'rsi' not in last.index or 'close' not in last.index:
                logger.info(f"  ❌ Missing required columns (RSI or close)")
                return None
            
            # Get price and RSI data
            close_price = float(last['close'])
            rsi_val = float(last['rsi'])
            
            # Get adaptive RSI threshold
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)
            
            # Get symbol-specific threshold override if available
            symbol_specific_threshold = self.get_symbol_specific_threshold(symbol_display)
            if symbol_specific_threshold is not None:
                adaptive_rsi_threshold = symbol_specific_threshold
                logger.info(f"  📌 Using symbol-specific RSI threshold: {adaptive_rsi_threshold:.2f}")
            
            # Log current state
            logger.info(f"  RSI: {rsi_val:.2f} (threshold: {adaptive_rsi_threshold:.2f})")
            
            # Check RSI condition (for oversold: RSI should be BELOW threshold)
            if rsi_val > adaptive_rsi_threshold:
                logger.info(f"  ❌ Signal: NONE - RSI {rsi_val:.2f} > threshold {adaptive_rsi_threshold:.2f}")
                return None
            else:
                logger.info(f"  ✅ RSI check passed: {rsi_val:.2f} <= {adaptive_rsi_threshold:.2f}")
            
            # Check volume if available
            volume_ok = True
            if 'volume' in last.index:
                volume_val = float(last['volume'])
                logger.info(f"  Volume: {volume_val:.2f}")
                # Volume check can be added here if needed
                volume_ok = volume_val > 0
            else:
                logger.info(f"  Volume: N/A")
            
            if not volume_ok:
                logger.info(f"  ❌ Signal: NONE - Volume check failed")
                return None
            
            # Calculate position size adjustment
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility)
            
            # Get trend strength for EMA adaptation
            trend_strength = regime_data.get('micro_trend_strength', 0.5)
            ema_params = self.adapt_ema_distances(trend_strength)
            
            # ===== ATR-BASED TP/SL CALCULATION =====
            entry_price = float(last['close'])
            atr_value = float(last['atr']) if 'atr' in last.index else entry_price * 0.02
            logger.info(f"  ATR: {atr_value:.4f}")
            
            # Get ATR multipliers from config
            tp_atr_mult = float(self.cfg.get("tp_atr_mult", 2.5))
            sl_atr_mult = float(self.cfg.get("sl_atr_mult", 1.2))
            
            # Calculate TP and SL from ATR
            target_price = entry_price + (atr_value * tp_atr_mult)
            stop_price = entry_price - (atr_value * sl_atr_mult)
            
            # Safety boundaries
            min_tp_pct = float(self.cfg.get("min_tp_pct", 0.008))
            max_sl_pct = float(self.cfg.get("max_sl_pct", 0.015))
            
            # Enforce minimum TP
            if (target_price - entry_price) / entry_price < min_tp_pct:
                target_price = entry_price * (1 + min_tp_pct)
            
            # Enforce maximum SL
            if (entry_price - stop_price) / entry_price > max_sl_pct:
                stop_price = entry_price * (1 - max_sl_pct)
            
            # Calculate and validate R/R ratio
            rr_numerator = target_price - entry_price
            rr_denominator = entry_price - stop_price
            if rr_numerator <= 0 or rr_denominator <= 0:
                logging.error(f"Invalid R/R calculation: numerator={rr_numerator}, denominator={rr_denominator}, entry={entry_price}, target={target_price}, stop={stop_price}")
                rr_ratio = float('nan')
            else:
                rr_ratio = rr_numerator / rr_denominator
            
            # Calculate percentages for signal
            tp_pct = (target_price - entry_price) / entry_price
            sl_pct = (entry_price - stop_price) / entry_price
            
            # Build adaptive signal with ATR-based TP/SL
            signal = {
                "side": "buy",
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "reason": f"Adaptive RSI oversold {rsi_val:.1f} (threshold: {adaptive_rsi_threshold:.1f}, regime: {market_regime['trend']}, R/R: {rr_ratio:.2f})",
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_atr_mult": tp_atr_mult,
                "sl_atr_mult": sl_atr_mult,
                "atr": atr_value,
                "rr_ratio": rr_ratio,
                "is_adaptive": True,
                "adaptive_threshold": adaptive_rsi_threshold,
                "position_multiplier": position_mult,
                "market_regime": market_regime,
                "ema_params": ema_params
            }
            
            logger.info(f"  ✅ Signal: BUY (RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}, regime={market_regime['trend']})")
            logger.info(f"  Entry: ${entry_price:.2f}, Target: ${target_price:.2f}, Stop: ${stop_price:.2f}, R/R: {rr_ratio:.2f}")
            
            # Strategy type ekle ve signal'i döndür
            signal['strategy_type'] = 'adaptive'
            return signal
            
        except Exception as e:
            logger.warning(f"Adaptive strategy failed: {e}, falling back to base")
            
            # FALLBACK TO BASE STRATEGY
            try:
                # Base OversoldBounce için
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m)
                    if base_signal:
                        base_signal['strategy_type'] = 'base_fallback'
                        base_signal['fallback_reason'] = str(e)
                        logger.info("✅ Fallback to base strategy successful")
                        return base_signal
            except Exception as fallback_error:
                logger.error(f"Base strategy also failed: {fallback_error}")
                
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
