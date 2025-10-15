#!/usr/bin/env python3
"""
Example: Live Trading with Integrated Monitoring

This example shows how to integrate the monitoring system with
the existing live trading infrastructure.

Usage:
    python examples/live_trading_with_monitoring.py
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.dashboard import MonitoringDashboard
from monitoring.alert_manager import AlertManager, AlertPriority, AlertChannel
from monitoring.performance_analytics import PerformanceAnalytics
from core.state import load_state, load_day_stats, save_state
from core.notify import Telegram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedTradingMonitor:
    """
    Integrated trading monitor that combines dashboard, alerts, and analytics.
    """
    
    def __init__(self, config: dict):
        """
        Initialize integrated monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.dashboard = MonitoringDashboard(
            port=config.get('dashboard_port', 8080)
        )
        
        self.alert_manager = AlertManager(
            config.get('alerts', {})
        )
        
        self.analytics = PerformanceAnalytics(
            data_dir=config.get('data_dir', 'data')
        )
        
        logger.info("Integrated trading monitor initialized")
    
    async def start(self):
        """Start all monitoring components."""
        await self.dashboard.start()
        logger.info("Monitoring dashboard started")
    
    async def stop(self):
        """Stop all monitoring components."""
        await self.dashboard.stop()
        logger.info("Monitoring stopped")
    
    async def update_from_state(self):
        """
        Update dashboard metrics from current state files.
        
        This should be called periodically during trading to keep
        the dashboard synchronized with actual trading state.
        """
        try:
            # Load current state
            state = load_state()
            day_stats = load_day_stats()
            
            # Calculate metrics
            open_positions = list(state.get('open', {}).values())
            closed_trades = state.get('closed', [])
            
            # Calculate win rate
            if closed_trades:
                winning_trades = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
                win_rate = winning_trades / len(closed_trades)
            else:
                win_rate = 0.0
            
            # Update dashboard
            self.dashboard.update_metrics(
                total_signals=day_stats.get('signals', 0),
                total_trades=len(closed_trades),
                win_rate=win_rate,
                total_pnl=day_stats.get('pnl', 0.0),
                open_positions=open_positions,
                health_status='healthy'
            )
            
            logger.debug("Dashboard updated from state")
            
        except Exception as e:
            logger.error(f"Error updating from state: {e}")
    
    async def on_signal_generated(self, signal: dict):
        """
        Handle a new trading signal.
        
        Args:
            signal: Signal dictionary with symbol, side, reason, etc.
        """
        # Add to recent signals
        current_metrics = self.dashboard.metrics
        recent_signals = current_metrics.get('recent_signals', [])
        
        signal_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.get('symbol'),
            'side': signal.get('side'),
            'reason': signal.get('reason', 'N/A'),
            'status': 'pending'
        }
        
        recent_signals.insert(0, signal_entry)
        recent_signals = recent_signals[:10]  # Keep last 10
        
        self.dashboard.update_metrics(
            recent_signals=recent_signals,
            total_signals=current_metrics.get('total_signals', 0) + 1
        )
        
        # Send alert for high-confidence signals
        if signal.get('confidence', 0) > 0.8:
            await self.alert_manager.send_alert(
                title=f"High Confidence Signal",
                message=f"{signal['side'].upper()} signal for {signal['symbol']}",
                priority=AlertPriority.HIGH,
                metadata=signal
            )
    
    async def on_trade_executed(self, trade: dict):
        """
        Handle a trade execution.
        
        Args:
            trade: Trade dictionary with symbol, side, size, price, etc.
        """
        await self.alert_manager.send_alert(
            title="Trade Executed",
            message=f"{trade['side'].upper()} {trade['size']} {trade['symbol']} @ {trade['price']}",
            priority=AlertPriority.INFO,
            metadata=trade
        )
        
        # Update metrics
        await self.update_from_state()
    
    async def on_position_closed(self, position: dict):
        """
        Handle a position closure.
        
        Args:
            position: Position dictionary with pnl, etc.
        """
        pnl = position.get('pnl', 0)
        
        # Send alert based on P&L
        if pnl > 100:
            priority = AlertPriority.HIGH
            title = "Big Win! 🎉"
        elif pnl > 0:
            priority = AlertPriority.INFO
            title = "Position Closed - Profit"
        elif pnl > -50:
            priority = AlertPriority.MEDIUM
            title = "Position Closed - Small Loss"
        else:
            priority = AlertPriority.HIGH
            title = "Position Closed - Significant Loss"
        
        await self.alert_manager.send_alert(
            title=title,
            message=f"P&L: ${pnl:.2f} on {position.get('symbol')}",
            priority=priority,
            metadata=position
        )
        
        # Update metrics
        await self.update_from_state()
    
    async def generate_daily_report(self):
        """
        Generate and save daily performance report.
        """
        try:
            state = load_state()
            closed_trades = state.get('closed', [])
            
            if not closed_trades:
                logger.info("No trades to report")
                return
            
            # Generate report
            report = self.analytics.generate_performance_report(closed_trades)
            
            # Save report
            filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            self.analytics.save_report(report, filename)
            
            # Send summary alert
            await self.alert_manager.send_alert(
                title="Daily Performance Report",
                message=f"Win Rate: {report['win_rate']:.1f}% | "
                       f"Total P&L: ${report['total_pnl']:.2f} | "
                       f"Trades: {report['total_trades']}",
                priority=AlertPriority.INFO,
                metadata=report
            )
            
            logger.info(f"Daily report generated: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")


async def main():
    """Main example function."""
    logger.info("="*70)
    logger.info("LIVE TRADING WITH INTEGRATED MONITORING")
    logger.info("="*70)
    
    # Configuration
    config = {
        'dashboard_port': 8080,
        'data_dir': 'data',
        'alerts': {
            'telegram': {
                'enabled': False,  # Set to True and provide credentials
                # 'bot_token': 'YOUR_TOKEN',
                # 'chat_id': 'YOUR_CHAT_ID'
            },
            'discord': {
                'enabled': False,
            },
            'webhook': {
                'enabled': False,
            }
        }
    }
    
    # Initialize monitor
    monitor = IntegratedTradingMonitor(config)
    
    try:
        # Start monitoring
        await monitor.start()
        
        logger.info("")
        logger.info("Monitoring system is running!")
        logger.info("Dashboard: http://localhost:8080")
        logger.info("")
        logger.info("In your trading loop, call:")
        logger.info("  - monitor.update_from_state() - to sync with state files")
        logger.info("  - monitor.on_signal_generated(signal) - when signals are generated")
        logger.info("  - monitor.on_trade_executed(trade) - when trades execute")
        logger.info("  - monitor.on_position_closed(position) - when positions close")
        logger.info("  - monitor.generate_daily_report() - for daily summaries")
        logger.info("")
        
        # Simulation: Update metrics periodically
        logger.info("Simulating periodic updates...")
        for i in range(5):
            await asyncio.sleep(5)
            await monitor.update_from_state()
            logger.info(f"Updated metrics ({i+1}/5)")
        
        logger.info("")
        logger.info("Example complete. In production, keep the monitor running")
        logger.info("throughout your trading session.")
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await monitor.stop()


if __name__ == '__main__':
    asyncio.run(main())
