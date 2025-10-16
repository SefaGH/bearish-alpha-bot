"""
Adaptive OversoldBounce strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from typing import Optional, Dict
from .oversold_bounce import OversoldBounce

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
    
    def signal(self, df_30m: pd.DataFrame, 
               regime_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime.
        
        Args:
            df_30m: 30-minute OHLCV dataframe with indicators
            regime_data: Optional market regime data for adaptation
                        If None, falls back to base strategy
        
        Returns:
            Signal dictionary or None
        """
        # If no regime data provided, use base strategy
        if regime_data is None:
            logger.debug("No regime data provided, using base OversoldBounce strategy")
            return super().signal(df_30m)
        
        try:
            # Debug: Market analysis started
            logger.debug("🎯 [STRATEGY-AdaptiveOB] Market analysis started")
            
            # Ensure we have valid data
            if df_30m.empty:
                logger.debug("🎯 [STRATEGY-AdaptiveOB] Empty dataframe, no signal")
                return None
            
            # Get last row, checking critical columns only (not ema200 which needs 200 bars)
            df_clean = df_30m.dropna(subset=['rsi', 'close'])
            if df_clean.empty:
                logger.debug("🎯 [STRATEGY-AdaptiveOB] No valid data after cleaning, no signal")
                return None
                
            last = df_clean.iloc[-1]
            
            # Debug: Price data
            logger.debug(f"📊 [STRATEGY-AdaptiveOB] Price data: close=${last['close']:.2f}, RSI={last['rsi']:.2f}")
            
            # Get adaptive RSI threshold
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            logger.debug(f"📊 [STRATEGY-AdaptiveOB] Market regime: {market_regime}")
            
            adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)
            logger.debug(f"📊 [STRATEGY-AdaptiveOB] Adaptive RSI threshold: {adaptive_rsi_threshold:.2f}")
            
            # Check RSI condition
            rsi_val = float(last['rsi'])
            if rsi_val > adaptive_rsi_threshold:
                logger.debug(f"❌ [STRATEGY-AdaptiveOB] Signal result: No signal - RSI {rsi_val:.2f} > {adaptive_rsi_threshold:.2f}")
                return None
            
            # Calculate position size adjustment
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility)
            logger.debug(f"📊 [STRATEGY-AdaptiveOB] Position multiplier: {position_mult:.2f} (volatility: {volatility})")
            
            # Get trend strength for EMA adaptation
            trend_strength = regime_data.get('micro_trend_strength', 0.5)
            ema_params = self.adapt_ema_distances(trend_strength)
            
            # Build adaptive signal
            signal = {
                "side": "buy",
                "reason": f"Adaptive RSI oversold {rsi_val:.1f} (threshold: {adaptive_rsi_threshold:.1f}, regime: {market_regime['trend']})",
                "tp_pct": float(self.cfg.get("tp_pct", 0.015)),
                "sl_pct": (float(self.cfg["sl_pct"]) if "sl_pct" in self.cfg else None),
                "sl_atr_mult": float(self.cfg.get("sl_atr_mult", 1.0)),
                
                # Adaptive parameters
                "position_multiplier": position_mult,
                "market_regime": market_regime,
                "adaptive_rsi_threshold": adaptive_rsi_threshold,
                "ema_params": ema_params
            }
            
            logger.debug(f"✅ [STRATEGY-AdaptiveOB] Signal result: BUY signal generated")
            logger.debug(f"📈 [STRATEGY-AdaptiveOB] Signal strength: RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}")
            logger.info(f"Adaptive OB signal: RSI {rsi_val:.1f} <= {adaptive_rsi_threshold:.1f}, "
                       f"regime={market_regime['trend']}, pos_mult={position_mult:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in adaptive signal generation: {e}")
            # Fallback to base strategy on error
            return super().signal(df_30m)
    
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
