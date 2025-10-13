"""
VST Market Intelligence System for BingX-specific optimization.
Provides specialized analysis and parameter tuning for VST test trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class VSTMarketAnalyzer:
    """
    BingX VST-specific market intelligence and optimization.
    
    Analyzes VST price patterns, optimizes test trading parameters,
    and monitors performance for the VST asset on BingX.
    """
    
    def __init__(self, bingx_client):
        """
        Initialize VST market analyzer.
        
        Args:
            bingx_client: BingX CcxtClient instance for VST data access
        """
        self.bingx_client = bingx_client
        self.vst_patterns = {}
        self.performance_history = []
        self.vst_symbol = 'VST/USDT:USDT'
        
    def analyze_vst_price_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        VST-specific price action pattern recognition.
        
        Analyzes VST volatility characteristics and price behavior.
        
        Args:
            df: VST OHLCV dataframe with indicators
            
        Returns:
            Dictionary with VST pattern analysis
        """
        try:
            if df.empty or len(df) < 50:
                return {
                    'volatility_profile': 'unknown',
                    'average_move': 0.0,
                    'support_resistance': []
                }
            
            # Calculate VST volatility characteristics
            returns = df['close'].pct_change().dropna()
            avg_volatility = returns.std()
            avg_move = returns.abs().mean()
            
            # Classify volatility profile
            if avg_volatility > 0.03:
                volatility_profile = 'high'
            elif avg_volatility < 0.015:
                volatility_profile = 'low'
            else:
                volatility_profile = 'moderate'
            
            # Identify support/resistance levels using recent highs/lows
            recent_data = df.tail(100)
            resistance_levels = self._find_resistance_levels(recent_data)
            support_levels = self._find_support_levels(recent_data)
            
            # Volume-price relationship
            if 'volume' in df.columns:
                volume_trend = self._analyze_volume_trend(df.tail(50))
            else:
                volume_trend = 'unknown'
            
            self.vst_patterns = {
                'volatility_profile': volatility_profile,
                'average_move': avg_move,
                'average_volatility': avg_volatility,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'volume_trend': volume_trend,
                'timestamp': datetime.now(timezone.utc)
            }
            
            logger.info(f"VST patterns updated: volatility={volatility_profile}, "
                       f"avg_move={avg_move:.4f}")
            
            return self.vst_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing VST price patterns: {e}")
            return {
                'volatility_profile': 'unknown',
                'average_move': 0.0,
                'support_resistance': []
            }
    
    def _find_resistance_levels(self, df: pd.DataFrame, n_levels: int = 3) -> List[float]:
        """Find resistance levels from recent highs."""
        if len(df) < 20:
            return []
        
        try:
            # Use rolling max to find local highs
            window = 10
            local_highs = df['high'].rolling(window=window, center=True).max()
            resistance_candidates = df[df['high'] == local_highs]['high'].values
            
            # Cluster similar levels
            if len(resistance_candidates) > 0:
                sorted_levels = sorted(resistance_candidates, reverse=True)
                # Take top N unique levels (within 1% tolerance)
                unique_levels = []
                for level in sorted_levels:
                    if not unique_levels or all(abs(level - ul) / ul > 0.01 for ul in unique_levels):
                        unique_levels.append(float(level))
                        if len(unique_levels) >= n_levels:
                            break
                return unique_levels
        except Exception as e:
            logger.warning(f"Error finding resistance levels: {e}")
        
        return []
    
    def _find_support_levels(self, df: pd.DataFrame, n_levels: int = 3) -> List[float]:
        """Find support levels from recent lows."""
        if len(df) < 20:
            return []
        
        try:
            # Use rolling min to find local lows
            window = 10
            local_lows = df['low'].rolling(window=window, center=True).min()
            support_candidates = df[df['low'] == local_lows]['low'].values
            
            # Cluster similar levels
            if len(support_candidates) > 0:
                sorted_levels = sorted(support_candidates)
                # Take bottom N unique levels (within 1% tolerance)
                unique_levels = []
                for level in sorted_levels:
                    if not unique_levels or all(abs(level - ul) / ul > 0.01 for ul in unique_levels):
                        unique_levels.append(float(level))
                        if len(unique_levels) >= n_levels:
                            break
                return unique_levels
        except Exception as e:
            logger.warning(f"Error finding support levels: {e}")
        
        return []
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend (increasing, decreasing, stable)."""
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return 'unknown'
            
            recent_volume = df['volume'].tail(10).mean()
            older_volume = df['volume'].iloc[:-10].mean() if len(df) > 10 else recent_volume
            
            if recent_volume > older_volume * 1.2:
                return 'increasing'
            elif recent_volume < older_volume * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'unknown'
    
    def optimize_test_trading_parameters(self, market_regime: Optional[Dict] = None) -> Dict[str, any]:
        """
        Optimize parameters specifically for VST test trading.
        
        Conservative parameters for test trading with risk adjustment.
        
        Args:
            market_regime: Optional market regime data for context
            
        Returns:
            Dictionary with optimized VST trading parameters
        """
        # Base conservative parameters for test trading
        base_params = {
            'position_size_mult': 0.1,  # 10% allocation for testing
            'max_positions': 1,          # Single position at a time
            'risk_per_trade': 0.01,      # 1% risk per trade
        }
        
        # Adjust based on VST patterns if available
        if self.vst_patterns:
            volatility = self.vst_patterns.get('volatility_profile', 'moderate')
            
            # VST-specific RSI thresholds
            if volatility == 'high':
                # High volatility: More conservative entries
                rsi_params = {
                    'ob_rsi_max': 20,   # Oversold bounce: RSI <= 20
                    'str_rsi_min': 75,  # Short the rip: RSI >= 75
                }
            elif volatility == 'low':
                # Low volatility: More opportunities
                rsi_params = {
                    'ob_rsi_max': 30,   # Oversold bounce: RSI <= 30
                    'str_rsi_min': 65,  # Short the rip: RSI >= 65
                }
            else:
                # Moderate volatility: Balanced
                rsi_params = {
                    'ob_rsi_max': 25,   # Oversold bounce: RSI <= 25
                    'str_rsi_min': 70,  # Short the rip: RSI >= 70
                }
            
            base_params.update(rsi_params)
        
        # Further adjust based on market regime if provided
        if market_regime:
            regime_trend = market_regime.get('trend', 'neutral')
            
            if regime_trend == 'bearish':
                # More aggressive for oversold bounces in bearish regime
                base_params['ob_rsi_max'] = base_params.get('ob_rsi_max', 25) + 5
            elif regime_trend == 'bullish':
                # More selective for shorts in bullish regime
                base_params['str_rsi_min'] = base_params.get('str_rsi_min', 70) + 5
        
        # BingX execution optimization
        base_params.update({
            'exchange': 'bingx',
            'symbol': self.vst_symbol,
            'timeframe': '30m',
            'min_volume': 10000,  # Minimum volume requirement
            'slippage_tolerance': 0.002  # 0.2% slippage tolerance
        })
        
        logger.info(f"VST test trading parameters optimized: {base_params}")
        
        return base_params
    
    def monitor_vst_performance(self, trade_result: Dict) -> Dict[str, any]:
        """
        Real-time VST trading performance monitoring.
        
        Tracks test trading results and provides feedback.
        
        Args:
            trade_result: Dictionary with trade execution results
            
        Returns:
            Performance summary and recommendations
        """
        try:
            # Add to performance history
            self.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'result': trade_result
            })
            
            # Keep only last 100 trades
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Calculate performance metrics
            if len(self.performance_history) >= 5:
                metrics = self._calculate_performance_metrics()
            else:
                metrics = {
                    'trades': len(self.performance_history),
                    'status': 'collecting_data'
                }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            return {
                'metrics': metrics,
                'recommendations': recommendations,
                'trade_count': len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring VST performance: {e}")
            return {
                'metrics': {},
                'recommendations': [],
                'trade_count': len(self.performance_history)
            }
    
    def _calculate_performance_metrics(self) -> Dict[str, any]:
        """Calculate performance metrics from trade history."""
        try:
            trades = [t['result'] for t in self.performance_history if 'result' in t]
            if not trades:
                return {}
            
            # Extract PnL values
            pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
            if not pnls:
                return {'trades': len(trades)}
            
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            total_pnl = sum(pnls)
            
            return {
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'risk_reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {}
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        
        if not metrics or 'win_rate' not in metrics:
            return recommendations
        
        win_rate = metrics.get('win_rate', 0)
        risk_reward = metrics.get('risk_reward', 0)
        
        # Win rate recommendations
        if win_rate < 0.4:
            recommendations.append("Win rate below 40% - Consider tightening entry criteria")
        elif win_rate > 0.6:
            recommendations.append("Win rate above 60% - Good performance, consider scaling")
        
        # Risk/reward recommendations
        if risk_reward < 1.5:
            recommendations.append("Risk/reward below 1.5 - Consider wider targets or tighter stops")
        
        return recommendations
    
    def get_vst_status(self) -> Dict[str, any]:
        """
        Get current VST analysis status and summary.
        
        Returns:
            Dictionary with VST status information
        """
        return {
            'symbol': self.vst_symbol,
            'patterns': self.vst_patterns,
            'performance_trades': len(self.performance_history),
            'last_update': self.vst_patterns.get('timestamp') if self.vst_patterns else None
        }
