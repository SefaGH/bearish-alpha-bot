"""
Risk-Aware Backtesting Module.

This module integrates the Phase 3 Risk Rules Engine with backtesting,
allowing different risk configurations to be tested and compared on historical data.

Features:
- Test different risk parameter sets
- Compare performance across risk configurations
- Generate comprehensive reports with risk metrics
- Simulate real-world risk management in backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskAwareBacktest:
    """
    Enhanced backtesting engine with integrated risk management.
    
    This class simulates trading with realistic risk management rules,
    allowing optimization of both strategy and risk parameters.
    """
    
    def __init__(self, initial_capital: float = 10000, risk_config: Dict = None):
        """
        Initialize risk-aware backtest engine.
        
        Args:
            initial_capital: Starting capital in USD
            risk_config: Dictionary of risk configuration parameters
        """
        self.initial_capital = initial_capital
        self.risk_config = risk_config or self._default_risk_config()
        
        # Initialize state
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        
        logger.info(f"Initialized RiskAwareBacktest with ${initial_capital:.2f} capital")
        logger.info(f"Risk config: {self.risk_config}")
    
    def _default_risk_config(self) -> Dict:
        """
        Get default risk configuration.
        
        Returns:
            Dictionary of default risk parameters
        """
        return {
            'max_position_size': 0.10,  # 10% max per position
            'max_portfolio_risk': 0.02,  # 2% max risk per trade
            'max_drawdown': 0.15,  # 15% max drawdown
            'max_portfolio_heat': 0.10,  # 10% max total risk
            'min_risk_reward': 1.5,  # 1.5:1 minimum R:R
        }
    
    def run_backtest(self, signals: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest with risk management.
        
        Args:
            signals: List of trading signals with entry, stop, target, etc.
            price_data: DataFrame with OHLCV data
            
        Returns:
            Backtest results dictionary with performance and risk metrics
        """
        logger.info(f"Starting backtest with {len(signals)} signals")
        
        # Reset state
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        
        # Track metrics
        signals_processed = 0
        signals_approved = 0
        signals_rejected = 0
        rejection_reasons = {}
        
        for signal in signals:
            signals_processed += 1
            
            # Validate signal with risk rules
            is_valid, reason = self._validate_signal_with_risk_rules(signal)
            
            if is_valid:
                signals_approved += 1
                # Execute trade
                trade_result = self._execute_trade(signal, price_data)
                if trade_result:
                    self.closed_trades.append(trade_result)
                    # Update capital
                    self.capital += trade_result['pnl']
                    self.peak_capital = max(self.peak_capital, self.capital)
            else:
                signals_rejected += 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        # Calculate final metrics
        results = self._calculate_results()
        
        # Add risk analysis
        results['risk_analysis'] = {
            'signals_processed': signals_processed,
            'signals_approved': signals_approved,
            'signals_rejected': signals_rejected,
            'approval_rate': signals_approved / max(signals_processed, 1),
            'rejection_reasons': rejection_reasons,
            'risk_config': self.risk_config
        }
        
        logger.info(f"Backtest complete: {signals_approved}/{signals_processed} signals approved")
        logger.info(f"Final capital: ${self.capital:.2f} (Return: {(self.capital/self.initial_capital - 1)*100:.2f}%)")
        
        return results
    
    def _validate_signal_with_risk_rules(self, signal: Dict) -> Tuple[bool, str]:
        """
        Validate signal against risk management rules.
        
        Args:
            signal: Trading signal
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # 1. Check position size
        position_value = signal.get('position_size', 0) * signal.get('entry', 0)
        max_position_value = self.capital * self.risk_config['max_position_size']
        
        if position_value > max_position_value:
            return (False, "Position size exceeds limit")
        
        # 2. Check risk amount
        entry = signal.get('entry', 0)
        stop = signal.get('stop', 0)
        position_size = signal.get('position_size', 0)
        
        if not stop:
            # Calculate stop from sl_pct if available
            if 'sl_pct' in signal:
                sl_pct = signal['sl_pct']
                side = signal.get('side', 'long')
                if side in ['long', 'buy']:
                    stop = entry * (1 - sl_pct)
                else:
                    stop = entry * (1 + sl_pct)
        
        if stop:
            risk_amount = abs(entry - stop) * position_size
            max_risk = self.capital * self.risk_config['max_portfolio_risk']
            
            if risk_amount > max_risk:
                return (False, "Risk amount exceeds limit")
        
        # 3. Check drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0
        if current_drawdown > self.risk_config['max_drawdown']:
            return (False, "Drawdown limit exceeded")
        
        # 4. Check risk/reward ratio
        target = signal.get('target', 0)
        if stop and target:
            risk_distance = abs(entry - stop)
            reward_distance = abs(target - entry)
            if risk_distance > 0:
                rr_ratio = reward_distance / risk_distance
                if rr_ratio < self.risk_config['min_risk_reward']:
                    return (False, "Risk/reward ratio too low")
        
        # 5. Check portfolio heat
        total_risk = sum(pos.get('risk_amount', 0) for pos in self.positions)
        if stop:
            total_risk += abs(entry - stop) * position_size
            portfolio_heat = total_risk / self.capital if self.capital > 0 else 0
            
            if portfolio_heat > self.risk_config['max_portfolio_heat']:
                return (False, "Portfolio heat exceeds limit")
        
        return (True, "All risk checks passed")
    
    def _execute_trade(self, signal: Dict, price_data: pd.DataFrame) -> Optional[Dict]:
        """
        Execute trade and calculate result.
        
        Args:
            signal: Trading signal
            price_data: Price data DataFrame
            
        Returns:
            Trade result dictionary or None if cannot execute
        """
        entry_price = signal.get('entry', 0)
        stop_loss = signal.get('stop', 0)
        target = signal.get('target', 0)
        position_size = signal.get('position_size', 0)
        side = signal.get('side', 'long')
        
        if not all([entry_price, stop_loss, target, position_size]):
            return None
        
        # Simulate trade execution
        # In a real backtest, would need to find actual exit based on price action
        # For simplicity, assume we hit either stop or target based on probability
        
        # Calculate potential outcomes
        risk_amount = abs(entry_price - stop_loss) * position_size
        reward_amount = abs(target - entry_price) * position_size
        
        # Simplified: assume 50% chance of hitting target, 50% of hitting stop
        # In real backtest, would use actual price data
        # Note: Using unseeded random for demo; in production would use actual price data
        hit_target = np.random.random() > 0.5
        
        if hit_target:
            pnl = reward_amount if side in ['long', 'buy'] else -reward_amount
            exit_price = target
            result = 'win'
        else:
            pnl = -risk_amount if side in ['long', 'buy'] else risk_amount
            exit_price = stop_loss
            result = 'loss'
        
        return {
            'symbol': signal.get('symbol', 'UNKNOWN'),
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * position_size)) * 100,
            'result': result,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount if hit_target else 0,
            'timestamp': datetime.now()
        }
    
    def _calculate_results(self) -> Dict[str, Any]:
        """
        Calculate comprehensive backtest results.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'max_drawdown': 0,
            }
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t['result'] == 'win']
        losing_trades = [t for t in self.closed_trades if t['result'] == 'loss']
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Win/loss analysis
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        max_drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average risk/reward
        avg_risk = np.mean([t['risk_amount'] for t in self.closed_trades])
        avg_reward = np.mean([t.get('reward_amount', 0) for t in winning_trades]) if winning_trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_risk_amount': avg_risk,
            'avg_reward_amount': avg_reward,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio from trade returns.
        
        Returns:
            Sharpe ratio
        """
        if not self.closed_trades:
            return 0.0
        
        returns = [t['pnl'] / self.initial_capital for t in self.closed_trades]
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily trades)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe


def compare_risk_configurations(signals: List[Dict], price_data: pd.DataFrame, 
                                risk_configs: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple risk configurations on the same signal set.
    
    Args:
        signals: List of trading signals
        price_data: Price data DataFrame
        risk_configs: List of risk configuration dictionaries
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for i, config in enumerate(risk_configs):
        logger.info(f"Testing risk configuration {i+1}/{len(risk_configs)}")
        
        backtest = RiskAwareBacktest(risk_config=config)
        result = backtest.run_backtest(signals, price_data)
        
        # Flatten results for DataFrame
        flat_result = {
            'config_id': i,
            **config,
            **result,
            'approval_rate': result.get('risk_analysis', {}).get('approval_rate', 0)
        }
        
        results.append(flat_result)
    
    return pd.DataFrame(results)


def generate_backtest_report(results: Dict[str, Any], output_file: str = None) -> str:
    """
    Generate markdown report from backtest results.
    
    Args:
        results: Backtest results dictionary
        output_file: Optional file path to save report
        
    Returns:
        Markdown report string
    """
    risk_analysis = results.get('risk_analysis', {})
    
    report = f"""# Risk-Aware Backtest Report

## Risk Configuration

- Max Position Size: {risk_analysis.get('risk_config', {}).get('max_position_size', 0):.1%}
- Max Portfolio Risk: {risk_analysis.get('risk_config', {}).get('max_portfolio_risk', 0):.1%}
- Max Drawdown: {risk_analysis.get('risk_config', {}).get('max_drawdown', 0):.1%}
- Max Portfolio Heat: {risk_analysis.get('risk_config', {}).get('max_portfolio_heat', 0):.1%}
- Min Risk/Reward: {risk_analysis.get('risk_config', {}).get('min_risk_reward', 0):.2f}

## Signal Analysis

- Signals Processed: {risk_analysis.get('signals_processed', 0)}
- Signals Approved: {risk_analysis.get('signals_approved', 0)}
- Signals Rejected: {risk_analysis.get('signals_rejected', 0)}
- Approval Rate: {risk_analysis.get('approval_rate', 0):.1%}

### Rejection Reasons

"""
    
    for reason, count in risk_analysis.get('rejection_reasons', {}).items():
        report += f"- {reason}: {count}\n"
    
    report += f"""
## Performance Metrics

- Initial Capital: ${results.get('initial_capital', 0):,.2f}
- Final Capital: ${results.get('final_capital', 0):,.2f}
- Total Return: {results.get('total_return_pct', 0):.2f}%
- Total P&L: ${results.get('total_pnl', 0):,.2f}

## Trade Statistics

- Total Trades: {results.get('total_trades', 0)}
- Winning Trades: {results.get('winning_trades', 0)}
- Losing Trades: {results.get('losing_trades', 0)}
- Win Rate: {results.get('win_rate', 0):.1%}

## Risk/Reward Analysis

- Average Win: ${results.get('avg_win', 0):,.2f}
- Average Loss: ${results.get('avg_loss', 0):,.2f}
- Profit Factor: {results.get('profit_factor', 0):.2f}
- Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}

## Risk Metrics

- Max Drawdown: {results.get('max_drawdown', 0):.1%}
- Avg Risk Amount: ${results.get('avg_risk_amount', 0):,.2f}
- Avg Reward Amount: ${results.get('avg_reward_amount', 0):,.2f}
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample signals
    signals = [
        {
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.02,
            'side': 'long',
            'sl_pct': 0.02
        }
    ] * 10  # 10 similar signals
    
    # Create sample price data
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'open': 50000,
        'high': 51000,
        'low': 49000,
        'close': 50500,
        'volume': 1000
    })
    
    # Run backtest
    backtest = RiskAwareBacktest(initial_capital=10000)
    results = backtest.run_backtest(signals, price_data)
    
    # Generate report
    report = generate_backtest_report(results)
    print(report)
