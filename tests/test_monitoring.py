"""
Tests for monitoring and alerting system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.monitoring.alert_manager import AlertManager, AlertPriority, AlertChannel, Alert
from src.monitoring.performance_analytics import PerformanceAnalytics
from src.monitoring.dashboard import MonitoringDashboard


class TestAlertManager:
    """Test alert manager functionality."""
    
    def test_alert_creation(self):
        """Test basic alert creation."""
        alert = Alert(
            id="test-123",
            timestamp=datetime.now(),
            priority=AlertPriority.INFO,
            title="Test Alert",
            message="This is a test",
            metadata={'key': 'value'},
            channels=[AlertChannel.TELEGRAM]
        )
        
        assert alert.title == "Test Alert"
        assert alert.priority == AlertPriority.INFO
        assert alert.metadata['key'] == 'value'
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        config = {
            'telegram': {'enabled': False},
            'discord': {'enabled': False},
            'email': {'enabled': False},
            'webhook': {'enabled': False}
        }
        
        manager = AlertManager(config)
        assert manager.config == config
        assert len(manager.alert_history) == 0
        assert len(manager.channel_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test alert rate limiting."""
        config = {
            'telegram': {'enabled': False},
            'discord': {'enabled': False}
        }
        
        manager = AlertManager(config)
        
        # First alert should go through
        result1 = await manager.send_alert(
            "Test Alert",
            "First message",
            priority=AlertPriority.INFO
        )
        
        # Second alert should be rate limited
        result2 = await manager.send_alert(
            "Test Alert",
            "Second message",
            priority=AlertPriority.INFO
        )
        
        assert result1 is False  # No channels configured
        assert result2 is False  # Rate limited or no channels
    
    def test_priority_icons(self):
        """Test priority icon mapping."""
        config = {'telegram': {'enabled': False}}
        manager = AlertManager(config)
        
        assert manager._get_priority_icon(AlertPriority.CRITICAL) == "🚨"
        assert manager._get_priority_icon(AlertPriority.HIGH) == "⚠️"
        assert manager._get_priority_icon(AlertPriority.INFO) == "📊"
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        config = {'telegram': {'enabled': False}}
        manager = AlertManager(config)
        
        # Add some test alerts
        for i in range(5):
            alert = Alert(
                id=f"test-{i}",
                timestamp=datetime.now(),
                priority=AlertPriority.INFO,
                title=f"Alert {i}",
                message="Test",
                metadata={},
                channels=[]
            )
            manager.alert_history.append(alert)
        
        summary = manager.get_alert_summary(hours=24)
        assert summary['total_alerts'] == 5
        assert summary['by_priority']['info'] == 5


class TestPerformanceAnalytics:
    """Test performance analytics functionality."""
    
    def test_initialization(self):
        """Test analytics initialization."""
        analytics = PerformanceAnalytics(data_dir='data')
        assert analytics.data_dir.name == 'data'
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        analytics = PerformanceAnalytics()
        
        # Create sample returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        
        sharpe = analytics.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty data."""
        analytics = PerformanceAnalytics()
        returns = pd.Series([])
        
        sharpe = analytics.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        analytics = PerformanceAnalytics()
        
        trades = [
            {'pnl': 10.0},
            {'pnl': -5.0},
            {'pnl': 15.0},
            {'pnl': -3.0},
            {'pnl': 8.0}
        ]
        
        win_rate = analytics.calculate_win_rate(trades)
        assert win_rate == 60.0  # 3 out of 5 wins
    
    def test_win_rate_empty(self):
        """Test win rate with no trades."""
        analytics = PerformanceAnalytics()
        win_rate = analytics.calculate_win_rate([])
        assert win_rate == 0.0
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        analytics = PerformanceAnalytics()
        
        trades = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 80.0},
            {'pnl': -20.0}
        ]
        
        pf = analytics.calculate_profit_factor(trades)
        assert pf > 0
        # (100 + 80) / (50 + 20) = 180 / 70 = 2.57
        assert abs(pf - 2.571) < 0.01
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        analytics = PerformanceAnalytics()
        
        # Create equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 95, 90, 100, 110])
        
        max_dd, start, end = analytics.calculate_max_drawdown(equity)
        assert max_dd > 0
        assert max_dd <= 1.0  # Should be a percentage
    
    def test_performance_report_empty(self):
        """Test performance report with no trades."""
        analytics = PerformanceAnalytics()
        
        report = analytics.generate_performance_report([])
        assert report['total_trades'] == 0
        assert report['win_rate'] == 0.0
        assert report['total_pnl'] == 0.0
    
    def test_performance_report_with_trades(self):
        """Test performance report generation."""
        analytics = PerformanceAnalytics()
        
        trades = [
            {'pnl': 50.0, 'timestamp': '2024-01-01'},
            {'pnl': -20.0, 'timestamp': '2024-01-02'},
            {'pnl': 30.0, 'timestamp': '2024-01-03'},
        ]
        
        report = analytics.generate_performance_report(trades)
        
        assert report['total_trades'] == 3
        assert report['total_pnl'] == 60.0
        assert report['win_rate'] > 0
        assert 'profit_factor' in report
        assert 'avg_trade' in report


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""
    
    @pytest.mark.asyncio
    async def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        dashboard = MonitoringDashboard(port=8081)
        
        assert dashboard.port == 8081
        assert dashboard.metrics['health_status'] == 'healthy'
        assert dashboard.metrics['total_trades'] == 0
    
    def test_metrics_update(self):
        """Test metrics update."""
        dashboard = MonitoringDashboard()
        
        dashboard.metrics['total_trades'] = 5
        dashboard.metrics['total_pnl'] = 100.0
        
        assert dashboard.metrics['total_trades'] == 5
        assert dashboard.metrics['total_pnl'] == 100.0
    
    @pytest.mark.asyncio
    async def test_dashboard_start_stop(self):
        """Test dashboard start and stop."""
        dashboard = MonitoringDashboard(port=8082)
        
        try:
            await dashboard.start()
            # Give it a moment to start
            await asyncio.sleep(0.1)
            
            # Check that it started
            assert dashboard.runner is not None
            
        finally:
            await dashboard.stop()
    
    @pytest.mark.asyncio
    async def test_broadcast_update(self):
        """Test broadcast update functionality."""
        dashboard = MonitoringDashboard()
        
        # Test broadcast with no connected clients
        await dashboard.broadcast_update({'test': 'data'})
        
        # Should not raise any errors
        assert len(dashboard.websockets) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
