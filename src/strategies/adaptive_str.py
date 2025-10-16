"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from typing import Optional, Dict
from .short_the_rip import ShortTheRip

logger = logging.getLogger(__name__)


class AdaptiveShortTheRip(ShortTheRip):
    """
    Market regime-aware ShortTheRip strategy.
    
    Adapts RSI thresholds, position sizing, and EMA requirements
    based on real-time market regime analysis.
    """
    
    def __init__(self, cfg: Dict, regime_analyzer=None):
        """
        Initialize adaptive ShortTheRip strategy.
        
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
            Adaptive RSI threshold for overbought detection
        """
        # Get config values with proper fallbacks
        base_rsi = float(self.base_cfg.get('adaptive_rsi_base',
                         self.base_cfg.get('rsi_min', 50)))
        
        # Get adjustment range from config (default ±10)
        adapt_range = float(self.base_cfg.get('adaptive_rsi_range', 10))
        
        trend = market_regime.get('trend', 'neutral')
        momentum = market_regime.get('momentum', 'sideways')
        
        # Start with base value
        threshold = base_rsi
        
        # Gentler adjustments based on regime
        # For short strategy: bearish = more aggressive (lower threshold), bullish = more selective (higher threshold)
        if trend == 'bearish':
            # In downtrends, be slightly more aggressive with shorts
            if momentum == 'strong':
                threshold = base_rsi - min(5, adapt_range/2)  # Max -5 adjustment
            else:
                threshold = base_rsi - min(3, adapt_range/3)  # Max -3 adjustment
        
        elif trend == 'bullish':
            # In uptrends, be more selective (need higher RSI)
            if momentum == 'strong':
                threshold = base_rsi + min(5, adapt_range/2)  # Max +5 adjustment
            else:
                threshold = base_rsi + min(3, adapt_range/3)  # Max +3 adjustment
        
        # Clamp to reasonable range for shorts (50-70 range)
        min_threshold = max(50, base_rsi - adapt_range)
        max_threshold = min(70, base_rsi + adapt_range)
        
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
    
    def adapt_ema_requirements(self, trend_strength: float) -> Dict[str, any]:
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
    
    def signal(self, df_30m: pd.DataFrame, 
               df_1h: pd.DataFrame,
               regime_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime.
        
        Args:
            df_30m: 30-minute OHLCV dataframe with indicators
            df_1h: 1-hour OHLCV dataframe with indicators
            regime_data: Optional market regime data for adaptation
                        If None, falls back to base strategy
        
        Returns:
            Signal dictionary or None
        """
        # If no regime data provided, use base strategy
        if regime_data is None:
            logger.debug("No regime data provided, using base ShortTheRip strategy")
            return super().signal(df_30m, df_1h)
        
        try:
            # Debug: Market analysis started
            logger.debug("🎯 [STRATEGY-AdaptiveSTR] Market analysis started")
            
            # Ensure we have valid data
            if df_30m.empty:
                logger.debug("🎯 [STRATEGY-AdaptiveSTR] Empty dataframe, no signal")
                return None
            
            # Get last row, checking critical columns only  
            df_clean = df_30m.dropna(subset=['rsi', 'close'])
            if df_clean.empty:
                logger.debug("🎯 [STRATEGY-AdaptiveSTR] No valid data after cleaning, no signal")
                return None
                
            last30 = df_clean.iloc[-1]
            
            # Debug: Price data
            logger.debug(f"📊 [STRATEGY-AdaptiveSTR] Price data: close=${last30['close']:.2f}, RSI={last30['rsi']:.2f}")
            
            # Get adaptive RSI threshold
            market_regime = {
                'trend': regime_data.get('trend', 'neutral'),
                'momentum': regime_data.get('momentum', 'sideways'),
                'volatility': regime_data.get('volatility', 'normal')
            }
            
            logger.debug(f"📊 [STRATEGY-AdaptiveSTR] Market regime: {market_regime}")
            
            adaptive_rsi_threshold = self.get_adaptive_rsi_threshold(market_regime)
            logger.debug(f"📊 [STRATEGY-AdaptiveSTR] Adaptive RSI threshold: {adaptive_rsi_threshold:.2f}")
            
            # Check RSI condition
            rsi_val = float(last30['rsi'])
            if rsi_val < adaptive_rsi_threshold:
                logger.debug(f"❌ [STRATEGY-AdaptiveSTR] Signal result: No signal - RSI {rsi_val:.2f} < {adaptive_rsi_threshold:.2f}")
                return None
            
            # Get trend strength for EMA adaptation
            trend_strength = regime_data.get('micro_trend_strength', 0.5)
            ema_params = self.adapt_ema_requirements(trend_strength)
            
            # Check EMA alignment if required
            ema_ok = True
            if ema_params['require_strict_ema_align']:
                if all(col in last30.index for col in ('ema21','ema50','ema200')):
                    ema21 = float(last30['ema21'])
                    ema50 = float(last30['ema50'])
                    ema200 = float(last30['ema200'])
                    # Strict alignment: 21 < 50 <= 200 (bearish alignment)
                    ema_ok = ema21 < ema50 <= ema200
                    logger.debug(f"📊 [STRATEGY-AdaptiveSTR] EMA alignment check: {ema_ok} (21={ema21:.2f}, 50={ema50:.2f}, 200={ema200:.2f})")
                else:
                    logger.warning("Missing EMA columns for strict alignment check")
            
            if not ema_ok:
                logger.debug(f"❌ [STRATEGY-AdaptiveSTR] Signal result: No signal - EMA alignment check failed")
                return None
            
            # Calculate position size adjustment
            volatility = regime_data.get('volatility', 'normal')
            position_mult = self.calculate_dynamic_position_size(volatility)
            logger.debug(f"📊 [STRATEGY-AdaptiveSTR] Position multiplier: {position_mult:.2f} (volatility: {volatility})")
            
            # Build adaptive signal
            signal = {
                "side": "sell",
                "reason": f"Adaptive RSI overbought {rsi_val:.1f} (threshold: {adaptive_rsi_threshold:.1f}, regime: {market_regime['trend']})",
                "tp_pct": float(self.cfg.get("tp_pct", 0.012)),
                "sl_atr_mult": float(self.cfg.get("sl_atr_mult", 1.2)),
                
                # Adaptive parameters
                "position_multiplier": position_mult,
                "market_regime": market_regime,
                "adaptive_rsi_threshold": adaptive_rsi_threshold,
                "ema_params": ema_params
            }
            
            logger.debug(f"✅ [STRATEGY-AdaptiveSTR] Signal result: SELL signal generated")
            logger.debug(f"📈 [STRATEGY-AdaptiveSTR] Signal strength: RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}")
            logger.info(f"Adaptive STR signal: RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}, "
                       f"regime={market_regime['trend']}, pos_mult={position_mult:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in adaptive signal generation: {e}")
            # Fallback to base strategy on error
            return super().signal(df_30m, df_1h)
    
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
