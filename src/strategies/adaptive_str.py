"""
Adaptive ShortTheRip strategy with market regime awareness.
Dynamically adjusts parameters based on market conditions.
"""

import pandas as pd
import logging
from typing import Optional, Dict
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
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi - min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        elif trend == 'bullish':
            # In uptrends, be more selective (need higher RSI)
            if momentum == 'strong':
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT, adapt_range/2)
            else:
                threshold = base_rsi + min(self.MAX_THRESHOLD_ADJUSTMENT * 0.6, adapt_range/3)
        
        # Clamp to reasonable range for shorts (55-85 range)
        min_threshold = max(55, base_rsi - adapt_range)  # STR için minimum 55
        max_threshold = min(85, base_rsi + adapt_range)  # STR için maximum 85
        
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
               df_1h: pd.DataFrame = None,
               regime_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate adaptive trading signal based on market regime.
        
        Args:
            df_30m: 30-minute OHLCV dataframe with indicators
            df_1h: Optional 1-hour OHLCV dataframe with indicators
            regime_data: Optional market regime data for adaptation
                        If None, falls back to base strategy
        
        Returns:
            Signal dictionary or None
        """
        # Data validation
        if df_30m is None or df_30m.empty:
            return None
        
        # Safely get last row
        try:
            last30 = df_30m.dropna().iloc[-1]
        except IndexError:
            logger.warning(f"[STRATEGY-AdaptiveSTR] Insufficient 30m data")
            return None
        
        # 1h data is optional for adaptive strategy
        last1h = None
        if df_1h is not None and not df_1h.empty:
            try:
                last1h = df_1h.dropna().iloc[-1]
            except IndexError:
                # Continue without 1h data
                pass
        
        # Analyze market regime with available data
        if regime_data is None:
            if last1h is not None:
                # Try to analyze regime if we have regime analyzer
                if self.regime_analyzer:
                    try:
                        regime_data = self.regime_analyzer.analyze_regime(last30, last1h)
                    except Exception as e:
                        logger.debug(f"Failed to analyze regime: {e}")
                        regime_data = None
            
            if regime_data is None:
                # Use default neutral regime
                regime_data = DEFAULT_MARKET_REGIME.copy()
        
        try:
            # Ensure we have valid data with critical columns
            if 'rsi' not in last30.index or 'close' not in last30.index:
                logger.debug("🎯 [STRATEGY-AdaptiveSTR] Missing required columns, no signal")
                return None
            
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
            
            # ===== KRİTİK DÜZELTME: ENTRY FİYATI EKLE =====
            entry_price = float(last30['close'])
            atr_value = float(last30['atr']) if 'atr' in last30.index else entry_price * 0.02
            
            # Calculate stop-loss from ATR (SHORT position - opposite direction)
            sl_atr_mult = float(self.cfg.get("sl_atr_mult", 1.2))
            stop_price = entry_price + (atr_value * sl_atr_mult)
            
            # Calculate target price from tp_pct
            tp_pct = float(self.cfg.get("tp_pct", 0.012))
            target_price = entry_price * (1 - tp_pct)
            
            # Build adaptive signal - TEK VE DÜZGÜN DICTIONARY
            signal = {
                "side": "sell",
                "entry": entry_price,
                "stop": stop_price,
                "target": target_price,
                "reason": f"Adaptive RSI overbought {rsi_val:.1f} (threshold: {adaptive_rsi_threshold:.1f}, regime: {market_regime['trend']})",
                "tp_pct": tp_pct,
                "sl_atr_mult": sl_atr_mult,
                "atr": atr_value,
                "is_adaptive": True,
                "adaptive_threshold": adaptive_rsi_threshold,
                "position_multiplier": position_mult,
                "market_regime": market_regime,
                "ema_params": ema_params
            }
            
            logger.debug(f"✅ [STRATEGY-AdaptiveSTR] Signal result: SELL signal generated")
            logger.debug(f"📈 [STRATEGY-AdaptiveSTR] Signal strength: RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}")
            logger.info(f"Adaptive STR signal: RSI {rsi_val:.1f} >= {adaptive_rsi_threshold:.1f}, "
                       f"regime={market_regime['trend']}, pos_mult={position_mult:.2f}")
            
            # Strategy type ekle ve signal'i döndür
            signal['strategy_type'] = 'adaptive'
            return signal
            
        except Exception as e:
            logger.warning(f"Adaptive strategy failed: {e}, falling back to base")
            
            # FALLBACK TO BASE STRATEGY
            try:
                # Base ShortTheRip için
                if hasattr(super(), 'signal'):
                    base_signal = super().signal(df_30m, df_1h)
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
            'strategy': 'adaptive_short_the_rip',
            'base_config': self.base_cfg,
            'has_regime_analyzer': self.regime_analyzer is not None
        }
