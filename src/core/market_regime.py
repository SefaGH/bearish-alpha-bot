"""
Market Regime Detection Engine for real-time market classification.
Provides multi-timeframe analysis for intelligent strategy adaptation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Real-time market regime detection and classification.
    
    Analyzes multiple timeframes to classify market conditions:
    - Primary trend (4H): bullish, bearish, neutral
    - Volatility regime: high, normal, low
    - Momentum state: strong, weak, sideways
    """
    
    def __init__(self):
        """Initialize the market regime analyzer."""
        self.regime_states = {
            'trend': ['bullish', 'bearish', 'neutral'],
            'volatility': ['high', 'normal', 'low'], 
            'momentum': ['strong', 'weak', 'sideways']
        }
        self.current_regime = {
            'trend': 'neutral',
            'volatility': 'normal',
            'momentum': 'sideways'
        }
        
    def detect_primary_trend_4h(self, ohlcv_4h: pd.DataFrame) -> str:
        """
        Primary trend detection using 4H timeframe.
        
        Uses EMA crossover analysis and price action to classify trend.
        
        Args:
            ohlcv_4h: 4H OHLCV dataframe with indicators
            
        Returns:
            Trend classification: 'bullish', 'bearish', or 'neutral'
        """
        try:
            if ohlcv_4h.empty or len(ohlcv_4h) < 20:
                return 'neutral'
                
            df = ohlcv_4h.dropna()
            if df.empty:
                return 'neutral'
                
            last = df.iloc[-1]
            
            # Check required columns
            required = ['close', 'ema21', 'ema50', 'ema200']
            if not all(col in last.index for col in required):
                logger.warning("Missing EMA columns for trend detection")
                return 'neutral'
            
            close = float(last['close'])
            ema21 = float(last['ema21'])
            ema50 = float(last['ema50'])
            ema200 = float(last['ema200'])
            
            # Bullish: EMAs aligned upward and price above all EMAs
            if ema21 > ema50 > ema200 and close > ema21:
                return 'bullish'
            
            # Bearish: EMAs aligned downward and price below all EMAs
            if ema21 < ema50 < ema200 and close < ema21:
                return 'bearish'
            
            # Mixed signals or transitioning
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error in primary trend detection: {e}")
            return 'neutral'
    
    def confirm_momentum_1h(self, ohlcv_1h: pd.DataFrame) -> str:
        """
        Momentum confirmation using 1H timeframe.
        
        Analyzes RSI and short-term price action for momentum validation.
        
        Args:
            ohlcv_1h: 1H OHLCV dataframe with indicators
            
        Returns:
            Momentum state: 'strong', 'weak', or 'sideways'
        """
        try:
            if ohlcv_1h.empty or len(ohlcv_1h) < 14:
                return 'sideways'
                
            df = ohlcv_1h.dropna()
            if df.empty:
                return 'sideways'
                
            last = df.iloc[-1]
            
            # Check RSI
            if 'rsi' not in last.index:
                return 'sideways'
            
            rsi = float(last['rsi'])
            
            # Calculate recent price momentum (last 10 bars)
            if len(df) >= 10:
                recent = df.tail(10)
                close_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                # Strong momentum: RSI extreme + significant price movement
                if (rsi > 65 and close_change > 0.02) or (rsi < 35 and close_change < -0.02):
                    return 'strong'
                
                # Weak/sideways: RSI neutral or small price movement
                if 40 <= rsi <= 60 or abs(close_change) < 0.01:
                    return 'sideways'
            
            # Moderate momentum
            return 'weak'
            
        except Exception as e:
            logger.error(f"Error in momentum confirmation: {e}")
            return 'sideways'
    
    def analyze_micro_trends_30m(self, ohlcv_30m: pd.DataFrame) -> Dict[str, float]:
        """
        Entry timing analysis using 30m timeframe.
        
        Identifies micro-trends and optimal entry/exit conditions.
        
        Args:
            ohlcv_30m: 30m OHLCV dataframe with indicators
            
        Returns:
            Dictionary with micro-trend metrics
        """
        try:
            if ohlcv_30m.empty or len(ohlcv_30m) < 20:
                return {'trend_strength': 0.0, 'entry_score': 0.5}
                
            df = ohlcv_30m.dropna()
            if df.empty:
                return {'trend_strength': 0.0, 'entry_score': 0.5}
            
            last = df.iloc[-1]
            
            # Calculate short-term trend strength
            if len(df) >= 20 and 'ema21' in last.index:
                recent = df.tail(20)
                ema_slope = (recent['ema21'].iloc[-1] - recent['ema21'].iloc[0]) / recent['ema21'].iloc[0]
                trend_strength = min(abs(ema_slope) * 100, 1.0)  # Normalize to 0-1
            else:
                trend_strength = 0.0
            
            # Calculate entry score based on RSI and price position
            entry_score = 0.5  # Neutral default
            if 'rsi' in last.index and 'ema21' in last.index:
                rsi = float(last['rsi'])
                close = float(last['close'])
                ema21 = float(last['ema21'])
                
                # Better entry score for oversold or near support
                if rsi < 30:
                    entry_score = 0.8
                elif rsi > 70:
                    entry_score = 0.2
                elif close < ema21:
                    entry_score = 0.6
                else:
                    entry_score = 0.4
            
            return {
                'trend_strength': trend_strength,
                'entry_score': entry_score
            }
            
        except Exception as e:
            logger.error(f"Error in micro-trend analysis: {e}")
            return {'trend_strength': 0.0, 'entry_score': 0.5}
    
    def classify_volatility_regime(self, price_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Volatility regime classification for risk adjustment.
        
        Uses ATR percentile ranking to determine volatility state.
        
        Args:
            price_data: OHLCV dataframe with ATR indicator
            
        Returns:
            Tuple of (volatility_class, risk_multiplier)
        """
        try:
            if price_data.empty or len(price_data) < 20:
                return ('normal', 1.0)
                
            df = price_data.dropna()
            if df.empty or 'atr' not in df.columns:
                return ('normal', 1.0)
            
            # Calculate ATR percentile (last 100 bars if available)
            lookback = min(100, len(df))
            recent_atr = df['atr'].tail(lookback)
            current_atr = float(df['atr'].iloc[-1])
            
            # Calculate percentile rank
            percentile = (recent_atr < current_atr).sum() / len(recent_atr)
            
            # Classify based on percentile
            if percentile > 0.75:
                # High volatility: reduce position size
                return ('high', 0.5)
            elif percentile < 0.25:
                # Low volatility: can increase position size
                return ('low', 1.5)
            else:
                # Normal volatility
                return ('normal', 1.0)
                
        except Exception as e:
            logger.error(f"Error in volatility classification: {e}")
            return ('normal', 1.0)
    
    def analyze_market_regime(self, df_30m: pd.DataFrame, 
                             df_1h: pd.DataFrame, 
                             df_4h: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive market regime analysis across multiple timeframes.
        
        Args:
            df_30m: 30-minute OHLCV data with indicators
            df_1h: 1-hour OHLCV data with indicators
            df_4h: 4-hour OHLCV data with indicators
            
        Returns:
            Dictionary containing regime classification and metrics
        """
        # Detect primary trend
        trend = self.detect_primary_trend_4h(df_4h)
        
        # Confirm momentum
        momentum = self.confirm_momentum_1h(df_1h)
        
        # Analyze micro-trends
        micro_metrics = self.analyze_micro_trends_30m(df_30m)
        
        # Classify volatility (use 4h for longer-term view)
        volatility_class, risk_mult = self.classify_volatility_regime(df_4h)
        
        # Update internal state
        self.current_regime = {
            'trend': trend,
            'volatility': volatility_class,
            'momentum': momentum
        }
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility_class,
            'risk_multiplier': risk_mult,
            'micro_trend_strength': micro_metrics['trend_strength'],
            'entry_score': micro_metrics['entry_score'],
            'timestamp': pd.Timestamp.now()
        }
    
    def get_current_regime(self) -> Dict[str, str]:
        """
        Get the current market regime classification.
        
        Returns:
            Dictionary with current regime states
        """
        return self.current_regime.copy()
