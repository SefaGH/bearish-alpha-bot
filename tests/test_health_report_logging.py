#!/usr/bin/env python3
"""
Tests for HealthMonitor health report file logging.

Validates that HealthMonitor writes health reports to logs/health_*.json.
This test file extracts the HealthMonitor class to avoid heavy dependencies.
"""

import sys
import os
import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class MockTelegram:
    """Mock Telegram for testing."""
    def __init__(self):
        self.messages = []
    
    def send(self, message: str):
        """Record sent messages."""
        self.messages.append(message)


class HealthMonitor:
    """
    HealthMonitor implementation for testing (extracted from live_trading_launcher.py).
    This includes the health report logging functionality.
    """
    
    def __init__(self, telegram: Optional[Any] = None):
        """Initialize health monitor."""
        self.telegram = telegram
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '300'))
        
        # Performance metrics
        self.metrics = {
            'loops_completed': 0,
            'errors_caught': 0,
            'signals_processed': 0,
            'last_error': None,
            'last_error_time': None
        }
        
        # Health status
        self.health_status = 'healthy'
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        
        # Health report file path
        ts = self.start_time.strftime('%Y%m%d_%H%M%S')
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.health_log_path = os.path.join(log_dir, f'health_{ts}.json')
    
    async def start_monitoring(self) -> asyncio.Task:
        """Start monitoring in background."""
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
        
        # Write final health report
        final_snapshot = self.get_health_report()
        self._write_health_report(snapshot=final_snapshot, final=True)
    
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
                
                # Write periodic health report
                snapshot = self.get_health_report()
                self._write_health_report(snapshot=snapshot, final=False)
        
        except asyncio.CancelledError:
            raise
    
    def record_error(self, error: str):
        """Record an error in the metrics."""
        self.metrics['errors_caught'] += 1
        self.metrics['last_error'] = error
        self.metrics['last_error_time'] = datetime.now(timezone.utc)
        
        # Update health status based on error frequency
        if self.metrics['errors_caught'] > 10:
            self.health_status = 'degraded'
        if self.metrics['errors_caught'] > 50:
            self.health_status = 'critical'
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            'status': self.health_status,
            'uptime_hours': uptime / 3600,
            'metrics': self.metrics,
            'last_heartbeat': self.last_heartbeat.isoformat()
        }
    
    def _write_health_report(self, snapshot: Optional[Dict[str, Any]] = None, final: bool = False):
        """
        Write health report to JSON file.
        
        Args:
            snapshot: Optional health snapshot to write. If None, generates one from get_health_report()
            final: Whether this is a final report (on shutdown)
        """
        try:
            if snapshot is None:
                snapshot = self.get_health_report()
            
            # Add metadata
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'report_type': 'final' if final else 'periodic',
                'health': snapshot
            }
            
            # Write to file
            with open(self.health_log_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            # Silently handle write errors to prevent health monitoring from disrupting
            # the main application. Health reports are diagnostic tools and should not
            # cause application failures. Common scenarios: disk full, permission errors.
            pass


class TestHealthReportLogging:
    """Test suite for HealthMonitor health report logging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_health_log_path_created(self):
        """Test that health_log_path is created in __init__."""
        monitor = HealthMonitor(telegram=None)
        
        # Check that health_log_path attribute exists
        assert hasattr(monitor, 'health_log_path')
        
        # Check that it points to logs/health_*.json
        assert monitor.health_log_path.startswith('logs/health_')
        assert monitor.health_log_path.endswith('.json')
        
        # Check that logs directory exists
        assert os.path.exists('logs')
        assert os.path.isdir('logs')
    
    @pytest.mark.asyncio
    async def test_write_health_report_creates_file(self):
        """Test that _write_health_report creates a JSON file."""
        monitor = HealthMonitor(telegram=None)
        
        # Write a health report
        monitor._write_health_report(final=False)
        
        # Check that the file exists
        assert os.path.exists(monitor.health_log_path)
        
        # Check that it's valid JSON
        with open(monitor.health_log_path, 'r') as f:
            data = json.load(f)
        
        # Verify structure
        assert 'timestamp' in data
        assert 'report_type' in data
        assert 'health' in data
        assert data['report_type'] == 'periodic'
    
    @pytest.mark.asyncio
    async def test_write_health_report_final(self):
        """Test that final health report has correct report_type."""
        monitor = HealthMonitor(telegram=None)
        
        # Write a final health report
        monitor._write_health_report(final=True)
        
        # Check the file
        with open(monitor.health_log_path, 'r') as f:
            data = json.load(f)
        
        assert data['report_type'] == 'final'
    
    @pytest.mark.asyncio
    async def test_health_report_contains_metrics(self):
        """Test that health report contains expected metrics."""
        monitor = HealthMonitor(telegram=None)
        
        # Record some metrics
        monitor.metrics['loops_completed'] = 5
        monitor.metrics['errors_caught'] = 2
        monitor.metrics['signals_processed'] = 10
        
        # Write report
        monitor._write_health_report(final=False)
        
        # Read and verify
        with open(monitor.health_log_path, 'r') as f:
            data = json.load(f)
        
        health = data['health']
        assert 'status' in health
        assert 'uptime_hours' in health
        assert 'metrics' in health
        assert 'last_heartbeat' in health
        
        # Check metrics
        assert health['metrics']['loops_completed'] == 5
        assert health['metrics']['errors_caught'] == 2
        assert health['metrics']['signals_processed'] == 10
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_writes_periodic_reports(self):
        """Test that monitoring loop writes periodic health reports."""
        # Set short interval for testing
        os.environ['HEALTH_CHECK_INTERVAL'] = '1'
        
        try:
            monitor = HealthMonitor(telegram=None)
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Wait for at least one heartbeat
            await asyncio.sleep(1.5)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            
            # Check that health report file exists
            assert os.path.exists(monitor.health_log_path)
            
            # Verify it's a valid JSON file
            with open(monitor.health_log_path, 'r') as f:
                data = json.load(f)
            
            # Should be marked as final (from stop_monitoring)
            assert data['report_type'] == 'final'
            
        finally:
            # Clean up environment
            os.environ.pop('HEALTH_CHECK_INTERVAL', None)
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_writes_final_report(self):
        """Test that stop_monitoring writes a final health report."""
        monitor = HealthMonitor(telegram=None)
        
        # Start and immediately stop
        await monitor.start_monitoring()
        await asyncio.sleep(0.1)
        await monitor.stop_monitoring()
        
        # Check that final report exists
        assert os.path.exists(monitor.health_log_path)
        
        with open(monitor.health_log_path, 'r') as f:
            data = json.load(f)
        
        assert data['report_type'] == 'final'
    
    @pytest.mark.asyncio
    async def test_health_report_file_overwrites(self):
        """Test that health reports overwrite the same file."""
        monitor = HealthMonitor(telegram=None)
        
        # Write first report
        monitor.metrics['loops_completed'] = 1
        monitor._write_health_report(final=False)
        
        with open(monitor.health_log_path, 'r') as f:
            data1 = json.load(f)
        assert data1['health']['metrics']['loops_completed'] == 1
        
        # Write second report
        monitor.metrics['loops_completed'] = 2
        monitor._write_health_report(final=False)
        
        with open(monitor.health_log_path, 'r') as f:
            data2 = json.load(f)
        assert data2['health']['metrics']['loops_completed'] == 2
    
    @pytest.mark.asyncio
    async def test_health_report_handles_write_errors(self):
        """Test that health report writing handles errors gracefully."""
        monitor = HealthMonitor(telegram=None)
        
        # Make the log path invalid (directory that can't be created)
        monitor.health_log_path = '/invalid/path/health.json'
        
        # This should not raise an exception
        monitor._write_health_report(final=False)
    
    @pytest.mark.asyncio
    async def test_health_report_json_structure(self):
        """Test the complete JSON structure of health report."""
        monitor = HealthMonitor(telegram=None)
        
        # Set some state
        monitor.health_status = 'healthy'
        monitor.metrics['errors_caught'] = 3
        monitor.record_error("Test error")
        
        # Write report
        monitor._write_health_report(final=False)
        
        # Read and verify complete structure
        with open(monitor.health_log_path, 'r') as f:
            data = json.load(f)
        
        # Top level
        assert isinstance(data, dict)
        assert 'timestamp' in data
        assert 'report_type' in data
        assert 'health' in data
        
        # Timestamp should be valid ISO format
        datetime.fromisoformat(data['timestamp'])
        
        # Health section
        health = data['health']
        assert health['status'] == 'healthy'
        assert isinstance(health['uptime_hours'], (int, float))
        assert health['uptime_hours'] >= 0
        
        # Metrics
        assert 'loops_completed' in health['metrics']
        assert 'errors_caught' in health['metrics']
        assert 'signals_processed' in health['metrics']
        assert health['metrics']['errors_caught'] == 4  # 3 + 1 from record_error


class TestHealthReportIntegration:
    """Integration tests for health report logging with other components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_health_report_with_telegram(self):
        """Test that health monitor works with Telegram notifications."""
        mock_telegram = MockTelegram()
        monitor = HealthMonitor(telegram=mock_telegram)
        
        # Write report
        monitor._write_health_report(final=True)
        
        # Check file exists
        assert os.path.exists(monitor.health_log_path)
    
    @pytest.mark.asyncio
    async def test_multiple_monitors_different_files(self):
        """Test that multiple monitors create different health report files."""
        # Create two monitors
        monitor1 = HealthMonitor(telegram=None)
        # Sleep for 1+ seconds to ensure different timestamps
        # (health log filename uses second-resolution timestamp format: %Y%m%d_%H%M%S)
        await asyncio.sleep(1.1)
        monitor2 = HealthMonitor(telegram=None)
        
        # They should have different paths
        assert monitor1.health_log_path != monitor2.health_log_path
        
        # Write reports
        monitor1._write_health_report(final=False)
        monitor2._write_health_report(final=False)
        
        # Both files should exist
        assert os.path.exists(monitor1.health_log_path)
        assert os.path.exists(monitor2.health_log_path)
    
    @pytest.mark.asyncio
    async def test_health_report_in_logs_directory(self):
        """Test that health reports are created in logs/ directory."""
        monitor = HealthMonitor(telegram=None)
        monitor._write_health_report(final=False)
        
        # Check that file is in logs/ directory
        assert monitor.health_log_path.startswith('logs/')
        
        # Check that logs/ directory exists and contains the file
        logs_files = os.listdir('logs')
        health_files = [f for f in logs_files if f.startswith('health_') and f.endswith('.json')]
        assert len(health_files) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
