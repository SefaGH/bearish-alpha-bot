#!/usr/bin/env python3
"""
Test continuous trading mode and auto-restart functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestContinuousMode:
    """Test continuous mode functionality."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        from live_trading_launcher import HealthMonitor
        
        monitor = HealthMonitor(telegram=None)
        
        assert monitor.health_status == 'healthy'
        assert monitor.metrics['loops_completed'] == 0
        assert monitor.metrics['errors_caught'] == 0
        assert not monitor.monitoring_active
    
    def test_health_monitor_error_recording(self):
        """Test health monitor error recording."""
        from live_trading_launcher import HealthMonitor
        
        monitor = HealthMonitor(telegram=None)
        
        # Record an error
        monitor.record_error("Test error")
        
        assert monitor.metrics['errors_caught'] == 1
        assert monitor.metrics['last_error'] == "Test error"
        assert monitor.metrics['last_error_time'] is not None
        assert monitor.health_status == 'healthy'  # Still healthy with 1 error
        
        # Record many errors to trigger degraded status
        for i in range(15):
            monitor.record_error(f"Error {i}")
        
        assert monitor.health_status == 'degraded'
    
    def test_health_monitor_report(self):
        """Test health monitor report generation."""
        from live_trading_launcher import HealthMonitor
        
        monitor = HealthMonitor(telegram=None)
        monitor.record_error("Test error")
        
        report = monitor.get_health_report()
        
        assert 'status' in report
        assert 'uptime_hours' in report
        assert 'metrics' in report
        assert 'last_heartbeat' in report
        assert report['status'] == 'healthy'
        assert report['metrics']['errors_caught'] == 1
    
    def test_auto_restart_manager_initialization(self):
        """Test auto-restart manager initialization."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=100, restart_delay=30)
        
        assert manager.max_restarts == 100
        assert manager.base_restart_delay == 30
        assert manager.restart_count == 0
        assert manager.consecutive_failures == 0
    
    def test_auto_restart_manager_should_restart(self):
        """Test auto-restart decision logic."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=5, restart_delay=30)
        
        # Should restart initially
        should_restart, reason = manager.should_restart()
        assert should_restart is True
        
        # Simulate reaching max restarts
        manager.restart_count = 5
        should_restart, reason = manager.should_restart()
        assert should_restart is False
        assert "Maximum restart limit" in reason
        
        # Test consecutive failures limit
        manager.restart_count = 0
        manager.consecutive_failures = 11
        should_restart, reason = manager.should_restart()
        assert should_restart is False
        assert "consecutive failures" in reason
    
    def test_auto_restart_manager_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=100, restart_delay=30)
        
        # Initial delay
        assert manager.calculate_restart_delay() == 30
        
        # After 1 failure
        manager.consecutive_failures = 1
        assert manager.calculate_restart_delay() == 60
        
        # After 2 failures
        manager.consecutive_failures = 2
        assert manager.calculate_restart_delay() == 120
        
        # After 3 failures
        manager.consecutive_failures = 3
        assert manager.calculate_restart_delay() == 240
        
        # Should cap at 3600 (1 hour)
        manager.consecutive_failures = 10
        assert manager.calculate_restart_delay() == 3600
    
    def test_auto_restart_manager_record_failure(self):
        """Test failure recording."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=100, restart_delay=30)
        
        manager.record_failure("Test failure")
        
        assert manager.restart_count == 1
        assert manager.consecutive_failures == 1
        assert manager.last_restart_time is not None
    
    def test_auto_restart_manager_record_success(self):
        """Test success recording."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=100, restart_delay=30)
        
        # Simulate failures
        manager.consecutive_failures = 5
        
        # Record success
        manager.record_success()
        
        assert manager.consecutive_failures == 0
    
    def test_auto_restart_manager_status(self):
        """Test status reporting."""
        from live_trading_launcher import AutoRestartManager
        
        manager = AutoRestartManager(max_restarts=100, restart_delay=30)
        manager.record_failure("Test failure")
        
        status = manager.get_status()
        
        assert 'restart_count' in status
        assert 'max_restarts' in status
        assert 'consecutive_failures' in status
        assert 'uptime_seconds' in status
        assert status['restart_count'] == 1
        assert status['max_restarts'] == 100
        assert status['consecutive_failures'] == 1


class TestProductionCoordinatorContinuousMode:
    """Test production coordinator continuous mode."""
    
    @pytest.mark.asyncio
    async def test_continuous_mode_parameter(self):
        """Test that continuous parameter is accepted."""
        from core.production_coordinator import ProductionCoordinator
        
        coordinator = ProductionCoordinator()
        
        # Mock initialization
        coordinator.is_initialized = True
        coordinator.trading_engine = AsyncMock()
        coordinator.trading_engine.start_live_trading = AsyncMock(return_value={'success': True})
        coordinator.trading_engine.stop_live_trading = AsyncMock()
        coordinator.circuit_breaker = AsyncMock()
        coordinator.circuit_breaker.check_circuit_breaker = AsyncMock(return_value={'tripped': False})
        
        # Start with continuous mode and short duration for testing
        task = asyncio.create_task(
            coordinator.run_production_loop(mode='paper', duration=0.1, continuous=True)
        )
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop the loop
        coordinator.is_running = False
        
        # Wait for completion
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Verify trading engine was started
        coordinator.trading_engine.start_live_trading.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
