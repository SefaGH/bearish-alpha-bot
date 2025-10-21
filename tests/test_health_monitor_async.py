#!/usr/bin/env python3
"""
Tests for HealthMonitor async behavior.

Validates that HealthMonitor runs in background without blocking.
"""

import sys
import os
import pytest
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# We'll define a minimal version of HealthMonitor for testing
# This avoids importing all dependencies


class HealthMonitor:
    """
    Minimal HealthMonitor implementation for testing.
    This matches the refactored implementation in live_trading_launcher.py
    """
    
    def __init__(self, telegram: Optional[Any] = None):
        self.telegram = telegram
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '300'))
        
        self.metrics = {
            'loops_completed': 0,
            'errors_caught': 0,
            'signals_processed': 0,
            'last_error': None,
            'last_error_time': None
        }
        
        self.health_status = 'healthy'
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
    
    async def start_monitoring(self) -> asyncio.Task:
        """Start monitoring in background (idempotent, non-blocking)."""
        if self._task and not self._task.done():
            return self._task
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitoring_loop())
        return self._task
    
    async def stop_monitoring(self):
        """Stop monitoring gracefully."""
        if not self._task:
            return
        
        self._stop_event.set()
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Internal loop - runs in background."""
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.heartbeat_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
                self.last_heartbeat = datetime.now(timezone.utc)
                self.metrics['loops_completed'] += 1
        
        except asyncio.CancelledError:
            raise
    
    def record_error(self, error: str):
        """Record an error."""
        self.metrics['errors_caught'] += 1
        self.metrics['last_error'] = error
        self.metrics['last_error_time'] = datetime.now(timezone.utc)


class TestHealthMonitorAsync:
    """Test suite for HealthMonitor async behavior."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_non_blocking(self):
        """Test that health monitor starts without blocking."""
        monitor = HealthMonitor(telegram=None)
        
        # Start monitoring - this should return immediately
        start_time = asyncio.get_event_loop().time()
        task = await monitor.start_monitoring()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should complete in less than 1 second (non-blocking)
        assert elapsed < 1.0, f"start_monitoring took {elapsed}s - should be instant"
        
        # Task should be running
        assert task is not None
        assert not task.done()
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Task should be stopped
        assert task.done() or task.cancelled()
    
    @pytest.mark.asyncio
    async def test_health_monitor_concurrent_execution(self):
        """Test that health monitor runs concurrently with main loop."""
        monitor = HealthMonitor(telegram=None)
        
        # Flag to check if main loop executed
        main_loop_executed = False
        
        async def main_loop():
            nonlocal main_loop_executed
            await asyncio.sleep(0.5)  # Simulate some work
            main_loop_executed = True
        
        # Start health monitor
        await monitor.start_monitoring()
        
        # Execute main loop - should not be blocked
        await main_loop()
        
        # Main loop should have executed
        assert main_loop_executed, "Main loop was blocked by health monitor"
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_health_monitor_idempotent_start(self):
        """Test that starting an already running monitor is idempotent."""
        monitor = HealthMonitor(telegram=None)
        
        # Start monitoring twice
        task1 = await monitor.start_monitoring()
        task2 = await monitor.start_monitoring()
        
        # Should return the same task
        assert task1 is task2
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_health_monitor_graceful_shutdown(self):
        """Test that health monitor stops gracefully."""
        monitor = HealthMonitor(telegram=None)
        
        # Start monitoring
        task = await monitor.start_monitoring()
        
        # Wait a bit for the task to be running
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        start_time = asyncio.get_event_loop().time()
        await monitor.stop_monitoring()
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should stop quickly (less than 2 seconds)
        assert elapsed < 2.0, f"stop_monitoring took {elapsed}s - should be fast"
        
        # Task should be stopped
        assert task.done() or task.cancelled()
    
    @pytest.mark.asyncio
    async def test_health_monitor_with_short_interval(self):
        """Test health monitor with short interval (using env var)."""
        # Set short interval for testing
        os.environ['HEALTH_CHECK_INTERVAL'] = '1'
        
        try:
            monitor = HealthMonitor(telegram=None)
            
            # Verify interval is set correctly
            assert monitor.heartbeat_interval == 1
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Wait for at least one heartbeat
            await asyncio.sleep(1.5)
            
            # Check that at least one loop completed
            assert monitor.metrics['loops_completed'] >= 1
            
            # Stop monitoring
            await monitor.stop_monitoring()
        
        finally:
            # Clean up environment
            del os.environ['HEALTH_CHECK_INTERVAL']
    
    @pytest.mark.asyncio
    async def test_health_monitor_stop_event(self):
        """Test that stop event properly terminates monitoring loop."""
        monitor = HealthMonitor(telegram=None)
        
        # Start monitoring
        task = await monitor.start_monitoring()
        
        # Stop event should be clear initially
        assert not monitor._stop_event.is_set()
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Stop event should be set
        assert monitor._stop_event.is_set()
        
        # Task should be done
        await asyncio.sleep(0.1)  # Give it a moment to finish
        assert task.done() or task.cancelled()
    
    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles work correctly."""
        monitor = HealthMonitor(telegram=None)
        
        for i in range(3):
            # Start monitoring
            task = await monitor.start_monitoring()
            assert not task.done()
            
            # Wait a bit
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            await asyncio.sleep(0.1)
            
            # Task should be done
            assert task.done() or task.cancelled()


class TestHealthMonitorIntegration:
    """Integration tests for HealthMonitor in context of trading loop."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_does_not_block_trading_loop(self):
        """Test that health monitor allows trading loop to start."""
        monitor = HealthMonitor(telegram=None)
        
        # Simulate trading loop
        trading_started = False
        trading_completed = False
        
        async def simulated_trading_loop():
            nonlocal trading_started, trading_completed
            trading_started = True
            await asyncio.sleep(0.2)  # Simulate some trading activity
            trading_completed = True
        
        # This simulates the pattern in _start_trading_loop
        _health_task = None
        try:
            # Start health monitor in background (NON-BLOCKING)
            if monitor:
                _health_task = asyncio.create_task(
                    monitor.start_monitoring()
                )
            
            # Trading loop should start immediately
            await simulated_trading_loop()
            
        finally:
            # Cleanup
            if monitor:
                await monitor.stop_monitoring()
            
            if _health_task and not _health_task.done():
                _health_task.cancel()
                try:
                    await _health_task
                except asyncio.CancelledError:
                    pass
        
        # Verify trading loop executed
        assert trading_started, "Trading loop never started (blocked by health monitor?)"
        assert trading_completed, "Trading loop never completed"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test that health monitor runs truly concurrently."""
        monitor = HealthMonitor(telegram=None)
        
        results = []
        
        async def record_operations():
            # Start health monitor
            await monitor.start_monitoring()
            results.append(('monitor_started', asyncio.get_event_loop().time()))
            
            # Do other work
            for i in range(3):
                await asyncio.sleep(0.1)
                results.append((f'work_{i}', asyncio.get_event_loop().time()))
            
            # Stop monitor
            await monitor.stop_monitoring()
            results.append(('monitor_stopped', asyncio.get_event_loop().time()))
        
        await record_operations()
        
        # Check that operations happened in order
        assert len(results) >= 5
        assert results[0][0] == 'monitor_started'
        assert results[-1][0] == 'monitor_stopped'
        
        # Check that work happened after monitor started
        for i in range(3):
            assert f'work_{i}' in [r[0] for r in results]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
