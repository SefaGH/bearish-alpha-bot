#!/usr/bin/env python3
"""
Performance analytics and reporting for trading bot.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Calculate and track trading performance metrics.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize performance analytics.
        
        Args:
            data_dir: Directory for storing analytics data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Convert to daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/365) - 1
        
        excess_returns = returns - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        daily_rf = (1 + risk_free_rate) ** (1/365) - 1
        excess_returns = returns - daily_rf
        
        # Only consider downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Tuple of (max_drawdown_pct, start_date, end_date)
        """
        if len(equity_curve) < 2:
            return 0.0, None, None
        
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        
        # Find start of maximum drawdown
        max_dd_start = equity_curve[:max_dd_end].idxmax()
        
        return abs(max_dd), max_dd_start, max_dd_end
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate win rate from trade history.
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
            
        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
            
        Returns:
            Profit factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    def calculate_risk_reward_ratio(self, trades: List[Dict]) -> float:
        """
        Calculate average risk/reward ratio.
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
            
        Returns:
            Average risk/reward ratio
        """
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if not winning_trades or not losing_trades:
            return 0.0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
        
        if avg_loss == 0:
            return 0.0
            
        return avg_win / avg_loss
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Series of returns
            max_drawdown: Maximum drawdown as decimal (e.g., 0.20 for 20%)
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2 or max_drawdown == 0:
            return 0.0
        
        # Annualized return
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Trading days
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return / abs(max_drawdown)
    
    def generate_performance_report(self, trades: List[Dict], 
                                   equity_curve: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Generate comprehensive performance report.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Optional equity curve series
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0
            }
        
        # Basic metrics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        risk_reward = self.calculate_risk_reward_ratio(trades)
        
        # Calculate returns if equity curve provided
        sharpe = 0.0
        sortino = 0.0
        max_dd = 0.0
        calmar = 0.0
        
        if equity_curve is not None and len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
            calmar = self.calculate_calmar_ratio(returns, max_dd)
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'avg_trade': total_pnl / len(trades) if trades else 0.0,
            'best_trade': max((t.get('pnl', 0) for t in trades), default=0.0),
            'worst_trade': min((t.get('pnl', 0) for t in trades), default=0.0)
        }
    
    def save_report(self, report: Dict[str, any], filename: str = 'performance_report.json'):
        """
        Save performance report to file.
        
        Args:
            report: Performance report dictionary
            filename: Output filename
        """
        report_path = self.data_dir / filename
        
        # Add timestamp
        report['generated_at'] = datetime.now().isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
    
    def load_report(self, filename: str = 'performance_report.json') -> Optional[Dict[str, any]]:
        """
        Load performance report from file.
        
        Args:
            filename: Report filename
            
        Returns:
            Performance report dictionary or None if not found
        """
        report_path = self.data_dir / filename
        
        if not report_path.exists():
            return None
        
        with open(report_path, 'r') as f:
            return json.load(f)
    
    def calculate_rolling_metrics(self, trades: List[Dict], 
                                  window_days: int = 7) -> Dict[str, List]:
        """
        Calculate rolling performance metrics.
        
        Args:
            trades: List of trade dictionaries with timestamps
            window_days: Window size in days
            
        Returns:
            Dictionary with rolling metrics
        """
        if not trades:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        if 'timestamp' not in df.columns or 'pnl' not in df.columns:
            logger.warning("Trades missing required fields for rolling metrics")
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Calculate rolling metrics
        rolling_pnl = df['pnl'].rolling(window=f'{window_days}D').sum()
        rolling_count = df['pnl'].rolling(window=f'{window_days}D').count()
        rolling_wins = (df['pnl'] > 0).rolling(window=f'{window_days}D').sum()
        rolling_win_rate = (rolling_wins / rolling_count * 100).fillna(0)
        
        return {
            'timestamps': rolling_pnl.index.tolist(),
            'rolling_pnl': rolling_pnl.tolist(),
            'rolling_trade_count': rolling_count.tolist(),
            'rolling_win_rate': rolling_win_rate.tolist()
        }
