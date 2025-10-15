#!/usr/bin/env python3
"""
Demo script for the Monitoring and Alerting System.

This script demonstrates how to use the monitoring dashboard,
alert manager, and performance analytics.

Usage:
    python examples/monitoring_demo.py
    
Then open http://localhost:8080 in your browser to see the dashboard.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.dashboard import MonitoringDashboard
from monitoring.alert_manager import AlertManager, AlertPriority, AlertChannel
from monitoring.performance_analytics import PerformanceAnalytics
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def simulate_trading_activity(dashboard: MonitoringDashboard, 
                                   alert_manager: AlertManager):
    """
    Simulate trading activity to demonstrate the monitoring system.
    
    Args:
        dashboard: Monitoring dashboard instance
        alert_manager: Alert manager instance
    """
    logger.info("Starting trading activity simulation...")
    
    # Simulate some trading data
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    total_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    signals = []
    
    for i in range(20):
        # Simulate a trade
        symbol = random.choice(symbols)
        side = random.choice(['buy', 'sell'])
        pnl = random.uniform(-50, 100)
        total_pnl += pnl
        total_trades += 1
        
        if pnl > 0:
            winning_trades += 1
        
        # Add signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'reason': f'Strategy trigger {i}',
            'status': 'executed' if pnl > 0 else 'stopped'
        }
        signals.insert(0, signal)
        signals = signals[:10]  # Keep only last 10
        
        # Update dashboard metrics
        dashboard.update_metrics(
            total_signals=i + 1,
            total_trades=total_trades,
            win_rate=winning_trades / total_trades if total_trades > 0 else 0.0,
            total_pnl=total_pnl,
            recent_signals=signals,
            health_status='healthy' if total_pnl > 0 else 'warning'
        )
        
        # Send alerts for significant events
        if pnl > 50:
            await alert_manager.send_alert(
                title="Big Win!",
                message=f"Large profit on {symbol}: ${pnl:.2f}",
                priority=AlertPriority.HIGH,
                metadata={'symbol': symbol, 'pnl': pnl}
            )
        elif pnl < -30:
            await alert_manager.send_alert(
                title="Significant Loss",
                message=f"Loss on {symbol}: ${pnl:.2f}",
                priority=AlertPriority.MEDIUM,
                metadata={'symbol': symbol, 'pnl': pnl}
            )
        
        logger.info(f"Trade {i+1}: {symbol} {side} - P&L: ${pnl:.2f}")
        
        # Wait a bit before next trade
        await asyncio.sleep(2)
    
    logger.info(f"Simulation complete. Total P&L: ${total_pnl:.2f}")
    logger.info(f"Win Rate: {(winning_trades/total_trades*100):.1f}%")


async def demonstrate_performance_analytics():
    """Demonstrate performance analytics features."""
    logger.info("Demonstrating performance analytics...")
    
    analytics = PerformanceAnalytics()
    
    # Create sample trade data
    trades = [
        {'pnl': 100.0, 'timestamp': '2024-01-01'},
        {'pnl': -50.0, 'timestamp': '2024-01-02'},
        {'pnl': 75.0, 'timestamp': '2024-01-03'},
        {'pnl': -25.0, 'timestamp': '2024-01-04'},
        {'pnl': 150.0, 'timestamp': '2024-01-05'},
    ]
    
    # Generate performance report
    report = analytics.generate_performance_report(trades)
    
    logger.info("Performance Report:")
    logger.info(f"  Total Trades: {report['total_trades']}")
    logger.info(f"  Win Rate: {report['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {report['profit_factor']:.2f}")
    logger.info(f"  Total P&L: ${report['total_pnl']:.2f}")
    logger.info(f"  Avg Trade: ${report['avg_trade']:.2f}")
    
    # Calculate Sharpe ratio
    returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.06])
    sharpe = analytics.calculate_sharpe_ratio(returns)
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")


async def main():
    """Main demo function."""
    logger.info("="*70)
    logger.info("BEARISH ALPHA BOT - MONITORING & ALERTING DEMO")
    logger.info("="*70)
    
    # Initialize components
    logger.info("Initializing monitoring components...")
    
    # Dashboard
    dashboard = MonitoringDashboard(port=8080)
    
    # Alert Manager (with no channels enabled for demo)
    alert_config = {
        'telegram': {'enabled': False},
        'discord': {'enabled': False},
        'email': {'enabled': False},
        'webhook': {'enabled': False}
    }
    alert_manager = AlertManager(alert_config)
    
    try:
        # Start dashboard
        await dashboard.start()
        logger.info("Dashboard started at http://localhost:8080")
        logger.info("Open this URL in your browser to see the live dashboard!")
        
        # Demonstrate performance analytics
        await demonstrate_performance_analytics()
        
        logger.info("")
        logger.info("Starting trading simulation...")
        logger.info("Watch the dashboard update in real-time!")
        logger.info("")
        
        # Run simulation
        await simulate_trading_activity(dashboard, alert_manager)
        
        # Keep dashboard running
        logger.info("")
        logger.info("Simulation complete. Dashboard will stay open for viewing.")
        logger.info("Press Ctrl+C to exit...")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await dashboard.stop()
        logger.info("Dashboard stopped. Goodbye!")


if __name__ == '__main__':
    asyncio.run(main())
