"""
Comprehensive tests for WebSocket Data Integration (Issue #120).
Tests WebSocket priority, REST fallback, metrics tracking, and health monitoring.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, MagicMock

# Mock ccxt before importing modules that need it
sys.modules['ccxt.pro'] = MagicMock()
sys.modules['ccxt'] = MagicMock()

from core.live_trading_engine import LiveTradingEngine
from core.websocket_manager import WebSocketManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestWebSocketPriority:
    """Test WebSocket priority data fetching."""
    
    def test_websocket_priority_when_available(self):
        """Test that WebSocket is used when data is fresh and available."""
        # Create mock WebSocket manager with fresh data
        ws_manager = Mock()
        ws_manager.is_data_fresh.return_value = True
        ws_manager.get_latest_data.return_value = {
            'ohlcv': [[1234567890000, 50000, 51000, 49000, 50500, 100]] * 100
        }
        ws_manager.get_connection_health.return_value = {
            'status': 'healthy',
            'active_streams': 5,
            'total_messages': 1000
        }
        
        # Create mock exchange clients
        exchange_clients = {
            'test_exchange': Mock()
        }
        
        # Initialize engine
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Fetch data with priority
        df = engine._get_ohlcv_with_priority('BTC/USDT:USDT', '1m', limit=100)
        
        # Verify WebSocket was called
        ws_manager.is_data_fresh.assert_called_once()
        ws_manager.get_latest_data.assert_called_once()
        
        # Verify stats recorded WebSocket fetch
        assert engine.ws_stats['websocket_fetches'] == 1
        assert engine.ws_stats['rest_fetches'] == 0
        assert engine.ws_stats['websocket_success_rate'] == 100.0
        
        logger.info("✓ WebSocket priority working when data is fresh")
    
    def test_rest_fallback_when_websocket_unavailable(self):
        """Test REST fallback when WebSocket is unavailable."""
        # Create mock WebSocket manager that returns no data
        ws_manager = Mock()
        ws_manager.is_data_fresh.return_value = False
        
        # Create mock exchange client with REST data
        mock_client = Mock()
        mock_client.fetch_ohlcv.return_value = [
            [1234567890000, 50000, 51000, 49000, 50500, 100]
        ] * 100
        
        exchange_clients = {
            'test_exchange': mock_client
        }
        
        # Initialize engine
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Fetch data with priority (should fall back to REST)
        df = engine._get_ohlcv_with_priority('BTC/USDT:USDT', '1m', limit=100)
        
        # Verify REST was called
        mock_client.fetch_ohlcv.assert_called_once()
        
        # Verify stats recorded REST fetch
        assert engine.ws_stats['websocket_fetches'] == 0
        assert engine.ws_stats['rest_fetches'] == 1
        
        logger.info("✓ REST fallback working when WebSocket unavailable")
    
    def test_rest_fallback_when_data_stale(self):
        """Test REST fallback when WebSocket data is stale."""
        # Create mock WebSocket manager with stale data
        ws_manager = Mock()
        ws_manager.is_data_fresh.return_value = False  # Data is stale
        
        # Create mock exchange client
        mock_client = Mock()
        mock_client.fetch_ohlcv.return_value = [
            [1234567890000, 50000, 51000, 49000, 50500, 100]
        ] * 100
        
        exchange_clients = {
            'test_exchange': mock_client
        }
        
        # Initialize engine
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Fetch data (should use REST due to stale data)
        df = engine._get_ohlcv_with_priority('BTC/USDT:USDT', '1m', limit=100)
        
        # Verify WebSocket freshness was checked
        ws_manager.is_data_fresh.assert_called_once()
        
        # Verify REST was used
        assert engine.ws_stats['rest_fetches'] == 1
        
        logger.info("✓ REST fallback working when data is stale")


class TestStatisticsTracking:
    """Test WebSocket statistics tracking."""
    
    def test_statistics_accuracy(self):
        """Test that statistics are accurately tracked."""
        # Create mock managers
        ws_manager = Mock()
        ws_manager.is_data_fresh.return_value = True
        ws_manager.get_latest_data.return_value = {
            'ohlcv': [[1234567890000, 50000, 51000, 49000, 50500, 100]] * 100
        }
        ws_manager.get_connection_health.return_value = {
            'status': 'healthy'
        }
        
        exchange_clients = {
            'test_exchange': Mock()
        }
        
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Make multiple fetches
        for _ in range(5):
            engine._get_ohlcv_with_priority('BTC/USDT:USDT', '1m', limit=100)
        
        # Verify statistics
        assert engine.ws_stats['websocket_fetches'] == 5
        assert engine.ws_stats['websocket_success_rate'] == 100.0
        assert engine.ws_stats['avg_latency_ws'] > 0
        
        logger.info("✓ Statistics tracking accurate")
    
    def test_latency_metrics(self):
        """Test that latency metrics are calculated correctly."""
        ws_manager = Mock()
        ws_manager.is_data_fresh.return_value = True
        ws_manager.get_latest_data.return_value = {
            'ohlcv': [[1234567890000, 50000, 51000, 49000, 50500, 100]] * 100
        }
        ws_manager.get_connection_health.return_value = {'status': 'healthy'}
        
        exchange_clients = {'test_exchange': Mock()}
        
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Make a fetch
        engine._get_ohlcv_with_priority('BTC/USDT:USDT', '1m', limit=100)
        
        # Check latency was recorded
        assert engine.ws_stats['avg_latency_ws'] > 0
        assert engine.ws_stats['total_latency_ws'] > 0
        
        logger.info("✓ Latency metrics calculated correctly")
    
    def test_success_rate_calculation(self):
        """Test success rate calculation with mixed results."""
        ws_manager = Mock()
        exchange_clients = {'test_exchange': Mock()}
        
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Simulate some successes and failures
        engine._record_ws_fetch(10.0, success=True)
        engine._record_ws_fetch(15.0, success=True)
        engine._record_ws_fetch(0, success=False)
        engine._record_ws_fetch(12.0, success=True)
        
        # Check success rate
        assert engine.ws_stats['websocket_fetches'] == 3
        assert engine.ws_stats['websocket_failures'] == 1
        assert engine.ws_stats['websocket_success_rate'] == 75.0
        
        logger.info("✓ Success rate calculated correctly")


class TestDataFreshness:
    """Test data freshness validation."""
    
    def test_data_freshness_check(self):
        """Test that data freshness is properly validated."""
        manager = WebSocketManager()
        
        # Mock data collector with recent data
        manager._data_collector = Mock()
        manager._data_collector.get_latest_ohlcv.return_value = [
            [int(time.time() * 1000), 50000, 51000, 49000, 50500, 100]
        ]
        
        # Mock active streams
        manager._active_streams = {'test_exchange': {'BTC/USDT:USDT_1m'}}
        
        # Check freshness
        is_fresh = manager.is_data_fresh('BTC/USDT:USDT', '1m', max_age_seconds=60)
        
        # Data should be fresh (just created)
        assert is_fresh
        
        logger.info("✓ Data freshness validation working")
    
    def test_stale_data_detection(self):
        """Test detection of stale data."""
        manager = WebSocketManager()
        
        # Mock data collector with old data (2 minutes old)
        old_timestamp = int((time.time() - 120) * 1000)
        manager._data_collector = Mock()
        manager._data_collector.get_latest_ohlcv.return_value = [
            [old_timestamp, 50000, 51000, 49000, 50500, 100]
        ]
        
        manager._active_streams = {'test_exchange': {'BTC/USDT:USDT_1m'}}
        
        # Check freshness with 60 second threshold
        is_fresh = manager.is_data_fresh('BTC/USDT:USDT', '1m', max_age_seconds=60)
        
        # Data should be stale
        assert not is_fresh
        
        logger.info("✓ Stale data detection working")


class TestConnectionHealth:
    """Test WebSocket connection health monitoring."""
    
    def test_health_status_healthy(self):
        """Test health status reporting when healthy."""
        manager = WebSocketManager()
        manager.message_count = 100
        manager.streams = {'BTC/USDT:USDT': True, 'ETH/USDT:USDT': True}
        
        health = manager.get_connection_health()
        
        assert health['status'] == 'healthy'
        assert health['active_streams'] == 2
        assert health['total_messages'] == 100
        assert 'uptime_seconds' in health
        
        logger.info("✓ Health status reporting correctly (healthy)")
    
    def test_health_status_disconnected(self):
        """Test health status when disconnected."""
        manager = WebSocketManager()
        manager.message_count = 0
        manager.streams = {}
        
        health = manager.get_connection_health()
        
        assert health['status'] == 'disconnected'
        assert health['active_streams'] == 0
        
        logger.info("✓ Health status reporting correctly (disconnected)")
    
    def test_health_status_warming_up(self):
        """Test health status during warmup phase."""
        manager = WebSocketManager()
        manager.message_count = 5  # Less than 10
        manager.streams = {'BTC/USDT:USDT': True}
        
        health = manager.get_connection_health()
        
        assert health['status'] == 'warming_up'
        
        logger.info("✓ Health status reporting correctly (warming up)")


class TestQualityScoring:
    """Test data quality scoring."""
    
    def test_quality_score_high_quality(self):
        """Test quality score for high quality data."""
        manager = WebSocketManager()
        
        # Setup fresh, complete data
        manager._data_collector = Mock()
        manager._data_collector.get_latest_ohlcv.return_value = [
            [int(time.time() * 1000), 50000, 51000, 49000, 50500, 100]
        ] * 100
        manager._active_streams = {'test_exchange': {'BTC/USDT:USDT_1m'}}
        manager.last_message_time = {'BTC/USDT:USDT': time.time()}
        
        score = manager.get_data_quality_score('BTC/USDT:USDT', '1m')
        
        # Should be high quality (fresh + complete + frequent updates)
        assert score >= 70  # At least 70% quality
        
        logger.info(f"✓ Quality scoring working (score: {score})")
    
    def test_quality_score_no_data(self):
        """Test quality score when no data available."""
        manager = WebSocketManager()
        manager._data_collector = Mock()
        manager._data_collector.get_latest_ohlcv.return_value = None
        
        score = manager.get_data_quality_score('BTC/USDT:USDT', '1m')
        
        assert score == 0.0
        
        logger.info("✓ Quality scoring correct for missing data")


class TestWebSocketStatsAPI:
    """Test get_websocket_stats API."""
    
    def test_comprehensive_stats(self):
        """Test comprehensive statistics retrieval."""
        ws_manager = Mock()
        ws_manager.get_connection_health.return_value = {
            'status': 'healthy',
            'active_streams': 5
        }
        
        exchange_clients = {'test_exchange': Mock()}
        
        engine = LiveTradingEngine(
            mode='paper',
            websocket_manager=ws_manager,
            exchange_clients=exchange_clients
        )
        
        # Record some metrics
        engine._record_ws_fetch(10.0, success=True)
        engine._record_ws_fetch(15.0, success=True)
        engine._record_rest_fetch(200.0, success=True)
        
        # Get stats
        stats = engine.get_websocket_stats()
        
        # Verify comprehensive stats
        assert 'websocket_fetches' in stats
        assert 'rest_fetches' in stats
        assert 'websocket_usage_ratio' in stats
        assert 'connection_health' in stats
        assert 'latency_improvement_pct' in stats
        
        # Check usage ratio calculation
        assert stats['websocket_usage_ratio'] > 0
        
        logger.info("✓ Comprehensive stats API working")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
