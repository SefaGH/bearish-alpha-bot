"""
Example: Risk-Aware Backtesting

This example demonstrates how to use the risk-aware backtesting module
to test different risk configurations and optimize risk parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.risk_aware_backtest import (
    RiskAwareBacktest,
    compare_risk_configurations,
    generate_backtest_report
)


def generate_sample_signals(n_signals=20):
    """Generate sample trading signals for demonstration."""
    np.random.seed(42)  # For reproducibility
    
    signals = []
    base_price = 50000
    
    for i in range(n_signals):
        # Generate realistic-looking signals
        entry = base_price + np.random.normal(0, 1000)
        
        # Stop loss (1-3% below entry for long)
        sl_pct = np.random.uniform(0.01, 0.03)
        stop = entry * (1 - sl_pct)
        
        # Target (2-5% above entry for long)
        tp_pct = np.random.uniform(0.02, 0.05)
        target = entry * (1 + tp_pct)
        
        # Position size (0.5-2% of capital in BTC)
        position_size = np.random.uniform(0.01, 0.03)
        
        signals.append({
            'symbol': 'BTC/USDT',
            'entry': entry,
            'stop': stop,
            'target': target,
            'position_size': position_size,
            'side': 'long',
            'sl_pct': sl_pct,
            'timestamp': datetime.now() + timedelta(hours=i)
        })
    
    return signals


def generate_sample_price_data():
    """Generate sample price data for demonstration."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 50000
    
    data = {
        'timestamp': dates,
        'open': base_price + np.random.normal(0, 1000, 100),
        'high': base_price + np.random.normal(500, 1000, 100),
        'low': base_price + np.random.normal(-500, 1000, 100),
        'close': base_price + np.random.normal(0, 1000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }
    
    return pd.DataFrame(data)


def example_1_basic_backtest():
    """Example 1: Basic risk-aware backtesting."""
    print("="*60)
    print("Example 1: Basic Risk-Aware Backtest")
    print("="*60)
    
    # Generate sample data
    signals = generate_sample_signals(n_signals=10)
    price_data = generate_sample_price_data()
    
    # Create backtest with default risk configuration
    backtest = RiskAwareBacktest(initial_capital=10000)
    
    print(f"\nRunning backtest with {len(signals)} signals...")
    print(f"Initial Capital: ${backtest.initial_capital:,.2f}")
    print(f"Risk Configuration:")
    for key, value in backtest.risk_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}" if value < 1 else f"  {key}: {value:.2f}")
    
    # Run backtest
    results = backtest.run_backtest(signals, price_data)
    
    # Display results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print('='*60)
    
    print(f"\nPerformance:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Total P&L: ${results['total_pnl']:,.2f}")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']}")
    print(f"  Losing Trades: {results['losing_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    risk_analysis = results['risk_analysis']
    print(f"\nRisk Analysis:")
    print(f"  Signals Processed: {risk_analysis['signals_processed']}")
    print(f"  Signals Approved: {risk_analysis['signals_approved']}")
    print(f"  Signals Rejected: {risk_analysis['signals_rejected']}")
    print(f"  Approval Rate: {risk_analysis['approval_rate']:.1%}")
    
    if risk_analysis['rejection_reasons']:
        print(f"\n  Rejection Reasons:")
        for reason, count in risk_analysis['rejection_reasons'].items():
            print(f"    - {reason}: {count}")


def example_2_compare_risk_configurations():
    """Example 2: Comparing different risk configurations."""
    print("\n" + "="*60)
    print("Example 2: Comparing Risk Configurations")
    print("="*60)
    
    # Generate sample data
    signals = generate_sample_signals(n_signals=20)
    price_data = generate_sample_price_data()
    
    # Define different risk configurations to compare
    risk_configs = [
        {
            'name': 'Conservative',
            'max_position_size': 0.05,  # 5%
            'max_portfolio_risk': 0.01,  # 1%
            'max_drawdown': 0.10,  # 10%
            'max_portfolio_heat': 0.08,  # 8%
            'min_risk_reward': 2.0,  # 2:1
        },
        {
            'name': 'Moderate',
            'max_position_size': 0.10,  # 10%
            'max_portfolio_risk': 0.02,  # 2%
            'max_drawdown': 0.15,  # 15%
            'max_portfolio_heat': 0.10,  # 10%
            'min_risk_reward': 1.5,  # 1.5:1
        },
        {
            'name': 'Aggressive',
            'max_position_size': 0.15,  # 15%
            'max_portfolio_risk': 0.03,  # 3%
            'max_drawdown': 0.20,  # 20%
            'max_portfolio_heat': 0.15,  # 15%
            'min_risk_reward': 1.0,  # 1:1
        }
    ]
    
    print(f"\nComparing {len(risk_configs)} risk configurations...")
    print(f"Signals: {len(signals)}")
    
    # Extract names and configs
    config_names = [c.pop('name') for c in risk_configs]
    
    # Run comparison
    comparison = compare_risk_configurations(signals, price_data, risk_configs)
    
    # Add names back
    comparison['config_name'] = config_names
    
    # Display comparison
    print(f"\n{'='*80}")
    print("Risk Configuration Comparison")
    print('='*80)
    
    print(f"\n{'Config':<15} {'Approval':<10} {'Trades':<8} {'Win Rate':<10} {'Return':<10} {'Max DD':<10}")
    print('-'*80)
    
    for idx, row in comparison.iterrows():
        config_name = row['config_name']
        approval_rate = row['approval_rate'] * 100
        trades = row['total_trades']
        win_rate = row['win_rate'] * 100
        return_pct = row['total_return_pct']
        max_dd = row['max_drawdown'] * 100
        
        print(f"{config_name:<15} {approval_rate:>8.1f}% {trades:>8} {win_rate:>9.1f}% {return_pct:>9.1f}% {max_dd:>9.1f}%")
    
    # Analyze results
    print(f"\n{'='*80}")
    print("Analysis")
    print('='*80)
    
    best_return_idx = comparison['total_return_pct'].idxmax()
    best_sharpe_idx = comparison['sharpe_ratio'].idxmax()
    lowest_dd_idx = comparison['max_drawdown'].idxmin()
    
    print(f"\nBest Return: {config_names[best_return_idx]} ({comparison.iloc[best_return_idx]['total_return_pct']:.2f}%)")
    print(f"Best Sharpe: {config_names[best_sharpe_idx]} ({comparison.iloc[best_sharpe_idx]['sharpe_ratio']:.2f})")
    print(f"Lowest Drawdown: {config_names[lowest_dd_idx]} ({comparison.iloc[lowest_dd_idx]['max_drawdown']:.1%})")
    
    print(f"\nKey Insights:")
    print(f"  - Conservative config approved {comparison.iloc[0]['approval_rate']:.1%} of signals")
    print(f"  - Aggressive config approved {comparison.iloc[2]['approval_rate']:.1%} of signals")
    print(f"  - Higher approval rates may lead to more trades but also higher risk")


def example_3_generate_report():
    """Example 3: Generating comprehensive backtest report."""
    print("\n" + "="*60)
    print("Example 3: Generating Backtest Report")
    print("="*60)
    
    # Generate sample data
    signals = generate_sample_signals(n_signals=15)
    price_data = generate_sample_price_data()
    
    # Run backtest
    backtest = RiskAwareBacktest(initial_capital=10000)
    results = backtest.run_backtest(signals, price_data)
    
    # Generate report
    print("\nGenerating markdown report...")
    report = generate_backtest_report(results)
    
    # Save report to file
    output_file = '/tmp/backtest_report.md'
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_file}")
    
    # Display report
    print("\n" + "="*60)
    print("Report Preview")
    print("="*60)
    print(report[:1500] + "...\n[truncated]")


def example_4_optimal_risk_config():
    """Example 4: Finding optimal risk configuration."""
    print("\n" + "="*60)
    print("Example 4: Finding Optimal Risk Configuration")
    print("="*60)
    
    # Generate larger dataset
    signals = generate_sample_signals(n_signals=30)
    price_data = generate_sample_price_data()
    
    print(f"\nSearching for optimal risk configuration...")
    print(f"Testing parameter grid...")
    
    # Create parameter grid
    position_sizes = [0.05, 0.10, 0.15]
    portfolio_risks = [0.01, 0.02, 0.03]
    
    best_sharpe = -999
    best_config = None
    best_results = None
    
    configs_tested = 0
    
    for pos_size in position_sizes:
        for portfolio_risk in portfolio_risks:
            config = {
                'max_position_size': pos_size,
                'max_portfolio_risk': portfolio_risk,
                'max_drawdown': 0.15,
                'max_portfolio_heat': 0.10,
                'min_risk_reward': 1.5,
            }
            
            backtest = RiskAwareBacktest(initial_capital=10000, risk_config=config)
            results = backtest.run_backtest(signals, price_data)
            
            configs_tested += 1
            
            if results['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['sharpe_ratio']
                best_config = config
                best_results = results
    
    print(f"\nTested {configs_tested} configurations")
    print(f"\n{'='*60}")
    print("Optimal Configuration")
    print('='*60)
    
    print(f"\nParameters:")
    print(f"  Max Position Size: {best_config['max_position_size']:.1%}")
    print(f"  Max Portfolio Risk: {best_config['max_portfolio_risk']:.1%}")
    
    print(f"\nPerformance:")
    print(f"  Total Return: {best_results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_results['max_drawdown']:.1%}")
    print(f"  Win Rate: {best_results['win_rate']:.1%}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Risk-Aware Backtesting Examples")
    print("Phase 3: Risk Parameter Optimization")
    print("="*60)
    
    # Run all examples
    example_1_basic_backtest()
    example_2_compare_risk_configurations()
    example_3_generate_report()
    example_4_optimal_risk_config()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Risk-aware backtesting simulates real-world risk management")
    print("  2. Different risk configurations produce different results")
    print("  3. Conservative configs reject more signals but may be safer")
    print("  4. Optimize risk parameters alongside strategy parameters")
    print("  5. Use Sharpe ratio and drawdown to compare configurations")
