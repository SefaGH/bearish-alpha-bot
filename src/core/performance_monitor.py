"""
Real-Time Performance Monitoring and Optimization.
Tracks strategy performance and provides optimization feedback.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

# ⬇️ ARTIK BOŞ DEĞİL, TÜM KODLAR BURADA!
class PerformanceMonitor: # ← PerformanceMonitor artık ana sınıf
    """
    Real-time strategy performance monitoring and optimization.
    
    Tracks performance metrics, detects parameter drift, and provides
    optimization recommendations for continuous improvement.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.performance_history = {}
        self.optimization_feedback = {}
        self.strategy_states = {}
        self.trades = []  # Production coordinator için
        self.metrics = {}  # Production coordinator için
        
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade for tracking (production compatibility)."""
        self.trades.append({
            'timestamp': datetime.now(timezone.utc),
            'data': trade_data
        })
        
        # Extract strategy name if available
        strategy_name = trade_data.get('strategy', 'unknown')
        
        # Track using existing method
        result = {
            'pnl': trade_data.get('pnl', 0),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'price': trade_data.get('price'),
            'quantity': trade_data.get('quantity')
        }
        
        self.track_strategy_performance(strategy_name, result)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get overall metrics (production compatibility)."""
        all_metrics = {
            'total_trades': len(self.trades),
            'strategies': {}
        }
        
        # Aggregate metrics from all strategies
        for strategy_name, history in self.performance_history.items():
            metrics = history.get('metrics', {})
            all_metrics['strategies'][strategy_name] = metrics
            
            # Update overall metrics
            if metrics:
                if 'total_pnl' not in all_metrics:
                    all_metrics['total_pnl'] = 0
                all_metrics['total_pnl'] += metrics.get('total_pnl', 0)
                
                if 'overall_win_rate' not in all_metrics:
                    all_metrics['overall_win_rate'] = []
                all_metrics['overall_win_rate'].append(metrics.get('win_rate', 0))
        
        # Calculate averages
        if 'overall_win_rate' in all_metrics and all_metrics['overall_win_rate']:
            all_metrics['avg_win_rate'] = sum(all_metrics['overall_win_rate']) / len(all_metrics['overall_win_rate'])
        
        return all_metrics
        
    def track_strategy_performance(self, strategy_name: str, 
                                   results: Dict) -> Dict[str, Any]:
        """
        Track real-time strategy performance metrics.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'oversold_bounce')
            results: Dictionary with trade results and metrics
            
        Returns:
            Updated performance summary for the strategy
        """
        try:
            # Initialize strategy history if needed
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = {
                    'trades': [],
                    'parameters': [],
                    'metrics': {}
                }
            
            # Add trade to history
            trade_entry = {
                'timestamp': datetime.now(timezone.utc),
                'result': results
            }
            self.performance_history[strategy_name]['trades'].append(trade_entry)
            
            # Keep last 200 trades per strategy
            if len(self.performance_history[strategy_name]['trades']) > 200:
                self.performance_history[strategy_name]['trades'] = \
                    self.performance_history[strategy_name]['trades'][-200:]
            
            # Calculate updated metrics
            metrics = self._calculate_strategy_metrics(strategy_name)
            self.performance_history[strategy_name]['metrics'] = metrics
            
            logger.info(f"Performance tracked for {strategy_name}: "
                       f"{len(self.performance_history[strategy_name]['trades'])} trades, "
                       f"win_rate={metrics.get('win_rate', 0):.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking performance for {strategy_name}: {e}")
            return {}
    
    def _calculate_strategy_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            trades = self.performance_history[strategy_name]['trades']
            if not trades:
                return {}
            
            # Extract results
            results = [t['result'] for t in trades if 'result' in t]
            if not results:
                return {'trade_count': len(trades)}
            
            # Extract PnL data
            pnls = []
            for r in results:
                if isinstance(r, dict) and 'pnl' in r:
                    pnls.append(float(r['pnl']))
                elif isinstance(r, (int, float)):
                    pnls.append(float(r))
            
            if not pnls:
                return {'trade_count': len(trades)}
            
            # Basic metrics
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            total_pnl = sum(pnls)
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(pnls)
            max_drawdown = self._calculate_max_drawdown(pnls)
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Recent performance (last 20 trades)
            recent_pnls = pnls[-20:] if len(pnls) >= 20 else pnls
            recent_wins = [p for p in recent_pnls if p > 0]
            recent_win_rate = len(recent_wins) / len(recent_pnls) if recent_pnls else 0
            
            metrics = {
                'trade_count': len(pnls),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'risk_reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'recent_win_rate': recent_win_rate,
                'last_updated': datetime.now(timezone.utc)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, pnls: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from PnL series."""
        try:
            if not pnls or len(pnls) < 2:
                return 0.0
            
            returns = pd.Series(pnls)
            excess_returns = returns - risk_free_rate
            
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std()
            # Annualize (assuming daily data)
            sharpe_annualized = sharpe * np.sqrt(252)
            
            return float(sharpe_annualized)
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        try:
            if not pnls:
                return 0.0
            
            cumulative = pd.Series(pnls).cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            max_dd = drawdown.min()
            
            return float(max_dd)
        except Exception:
            return 0.0
    
    def detect_parameter_drift(self, strategy_name: str, 
                              current_params: Dict,
                              performance_threshold: float = 0.4) -> Tuple[bool, List[str]]:
        """
        Detect when parameters need adjustment.
        
        Args:
            strategy_name: Name of the strategy
            current_params: Current parameter configuration
            performance_threshold: Minimum acceptable win rate
            
        Returns:
            Tuple of (needs_adjustment, reasons)
        """
        try:
            if strategy_name not in self.performance_history:
                return (False, ["Insufficient data for drift detection"])
            
            metrics = self.performance_history[strategy_name].get('metrics', {})
            if not metrics:
                return (False, ["No metrics available"])
            
            needs_adjustment = False
            reasons = []
            
            # Check win rate degradation
            win_rate = metrics.get('win_rate', 0.5)
            recent_win_rate = metrics.get('recent_win_rate', 0.5)
            
            if win_rate < performance_threshold:
                needs_adjustment = True
                reasons.append(f"Win rate ({win_rate:.1%}) below threshold ({performance_threshold:.1%})")
            
            if recent_win_rate < win_rate * 0.7:
                needs_adjustment = True
                reasons.append(f"Recent performance degradation: {recent_win_rate:.1%} vs {win_rate:.1%}")
            
            # Check risk/reward ratio
            risk_reward = metrics.get('risk_reward', 0)
            if risk_reward < 1.0:
                needs_adjustment = True
                reasons.append(f"Risk/reward ratio ({risk_reward:.2f}) below 1.0")
            
            # Check drawdown
            max_dd = metrics.get('max_drawdown', 0)
            total_pnl = metrics.get('total_pnl', 0)
            if total_pnl > 0 and abs(max_dd) > total_pnl * 0.5:
                needs_adjustment = True
                reasons.append(f"Excessive drawdown: {max_dd:.2f} vs PnL {total_pnl:.2f}")
            
            # Track parameter history
            if 'parameters' not in self.performance_history[strategy_name]:
                self.performance_history[strategy_name]['parameters'] = []
            
            self.performance_history[strategy_name]['parameters'].append({
                'timestamp': datetime.now(timezone.utc),
                'params': current_params.copy(),
                'metrics': metrics.copy()
            })
            
            return (needs_adjustment, reasons)
            
        except Exception as e:
            logger.error(f"Error detecting parameter drift: {e}")
            return (False, [f"Error: {str(e)}"])
    
    def provide_optimization_feedback(self, strategy_name: str) -> Dict[str, Any]:
        """
        Provide real-time optimization recommendations.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            if strategy_name not in self.performance_history:
                return {
                    'status': 'no_data',
                    'recommendations': []
                }
            
            metrics = self.performance_history[strategy_name].get('metrics', {})
            if not metrics or metrics.get('trade_count', 0) < 10:
                return {
                    'status': 'insufficient_data',
                    'recommendations': ['Collect at least 10 trades for meaningful analysis']
                }
            
            recommendations = []
            adjustments = {}
            
            # Win rate analysis
            win_rate = metrics.get('win_rate', 0)
            if win_rate < 0.4:
                recommendations.append("Win rate low - Consider tightening entry criteria")
                adjustments['entry_criteria'] = 'tighter'
            elif win_rate > 0.7:
                recommendations.append("High win rate - Consider relaxing entry criteria for more opportunities")
                adjustments['entry_criteria'] = 'relaxed'
            
            # Risk/reward analysis
            risk_reward = metrics.get('risk_reward', 0)
            if risk_reward < 1.5:
                recommendations.append("Risk/reward suboptimal - Consider wider targets or tighter stops")
                adjustments['targets'] = 'wider'
            elif risk_reward > 3.0:
                recommendations.append("Excellent risk/reward - Current parameters working well")
            
            # Profit factor analysis
            profit_factor = metrics.get('profit_factor', 0)
            if profit_factor < 1.0:
                recommendations.append("Profit factor below 1.0 - Strategy losing money")
                adjustments['urgency'] = 'high'
            elif profit_factor > 2.0:
                recommendations.append("Strong profit factor - Strategy performing well")
            
            # Drawdown analysis
            max_dd = metrics.get('max_drawdown', 0)
            if abs(max_dd) > 100:  # Assuming USDT
                recommendations.append("High drawdown detected - Consider reducing position sizes")
                adjustments['position_size'] = 'reduce'
            
            # Performance trend
            recent_win_rate = metrics.get('recent_win_rate', 0)
            if recent_win_rate < win_rate * 0.8:
                recommendations.append("Recent performance declining - Parameters may need refresh")
                adjustments['parameter_review'] = 'needed'
            
            feedback = {
                'status': 'analyzed',
                'metrics': metrics,
                'recommendations': recommendations,
                'suggested_adjustments': adjustments,
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Cache feedback
            self.optimization_feedback[strategy_name] = feedback
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error providing optimization feedback: {e}")
            return {
                'status': 'error',
                'recommendations': [f"Error: {str(e)}"]
            }
    
    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy summary
        """
        if strategy_name not in self.performance_history:
            return {
                'strategy': strategy_name,
                'status': 'no_data'
            }
        
        return {
            'strategy': strategy_name,
            'metrics': self.performance_history[strategy_name].get('metrics', {}),
            'trade_count': len(self.performance_history[strategy_name].get('trades', [])),
            'last_feedback': self.optimization_feedback.get(strategy_name, {}),
            'status': 'active'
        }
    
    def get_all_strategies_summary(self) -> Dict[str, Dict]:
        """
        Get summary for all monitored strategies.
        
        Returns:
            Dictionary mapping strategy names to their summaries
        """
        return {
            strategy: self.get_strategy_summary(strategy)
            for strategy in self.performance_history.keys()
        }
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot for production coordinator."""
        return {
            'timestamp': datetime.now(timezone.utc),
            'total_trades': len(self.trades),
            'metrics': self.get_metrics(),
            'strategies': self.get_all_strategies_summary()
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (for daily limit tracking)."""
        logger.info("Resetting daily performance statistics")
        
        # Keep history but reset daily counters
        for strategy_name in self.performance_history:
            # Mark daily reset in parameters
            if 'parameters' not in self.performance_history[strategy_name]:
                self.performance_history[strategy_name]['parameters'] = []
            
            self.performance_history[strategy_name]['parameters'].append({
                'timestamp': datetime.now(timezone.utc),
                'event': 'daily_reset',
                'metrics_before_reset': self.performance_history[strategy_name].get('metrics', {}).copy()
            })
        
        # Clear today's trades for daily counting
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.trades = [t for t in self.trades if t.get('timestamp', datetime.now(timezone.utc)) >= today_start]
        
        logger.info(f"Daily stats reset complete. Kept {len(self.trades)} trades from today")

# ⬇️ EN SONDA, backward compatibility için alias ekle
RealTimePerformanceMonitor = PerformanceMonitor
