"""
Tests for WebSocket Client Connection State Tracking
Tests the enhanced connection state tracking features added to websocket_client.py
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from src.core.websocket_client import WebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConnectionStateTracking:
    """Test connection state tracking functionality."""
    
    def test_initial_connection_state(self):
        """Test that client starts in disconnected state."""
        client = WebSocketClient('kucoinfutures')
        
        # Check initial state
        assert client._is_connected == False
        assert client._first_message_received == False
        assert client._last_message_time is None
        assert client.is_connected() == False
        
        logger.info("✓ Initial connection state is correct")
    
    def test_error_history_initialization(self):
        """Test that error history is initialized correctly."""
        client = WebSocketClient('kucoinfutures')
        
        assert hasattr(client, 'error_history')
        assert isinstance(client.error_history, list)
        assert len(client.error_history) == 0
        assert client.max_error_history == 100
        
        logger.info("✓ Error history initialized correctly")
    
    def test_log_error_method(self):
        """Test _log_error method adds errors to history."""
        client = WebSocketClient('kucoinfutures')
        
        # Log some errors
        client._log_error('parse_frame', 'Test parse_frame error')
        client._log_error('AttributeError', 'Test attribute error')
        
        assert len(client.error_history) == 2
        assert client.error_history[0]['type'] == 'parse_frame'
        assert client.error_history[1]['type'] == 'AttributeError'
        assert 'timestamp' in client.error_history[0]
        assert 'message' in client.error_history[0]
        
        logger.info("✓ Error logging works correctly")
    
    def test_error_history_limit(self):
        """Test that error history respects max size limit."""
        client = WebSocketClient('kucoinfutures')
        
        # Add more than max_error_history errors
        for i in range(150):
            client._log_error('test_error', f'Error {i}')
        
        # Should keep only the last 100
        assert len(client.error_history) == 100
        # The last error should be Error 149
        assert 'Error 149' in client.error_history[-1]['message']
        
        logger.info("✓ Error history respects size limit")
    
    def test_get_recent_error_count(self):
        """Test _get_recent_error_count method."""
        client = WebSocketClient('kucoinfutures')
        
        # Add some errors with timestamps
        now = datetime.now()
        
        # Add old error (10 minutes ago)
        client.error_history.append({
            'timestamp': now - timedelta(minutes=10),
            'type': 'old_error',
            'message': 'Old error'
        })
        
        # Add recent errors (within last 5 minutes)
        for i in range(5):
            client.error_history.append({
                'timestamp': now - timedelta(seconds=i * 30),
                'type': 'recent_error',
                'message': f'Recent error {i}'
            })
        
        # Count errors in last 5 minutes (300 seconds)
        recent_count = client._get_recent_error_count(300)
        
        assert recent_count == 5  # Should not count the 10-minute-old error
        
        # Count errors in last 1 minute (60 seconds)
        recent_count_1min = client._get_recent_error_count(60)
        assert recent_count_1min <= 5  # Should count only very recent errors
        
        logger.info("✓ Recent error count calculation works correctly")
    
    @pytest.mark.asyncio
    async def test_connection_state_on_close(self):
        """Test that connection state is reset when client is closed."""
        client = WebSocketClient('kucoinfutures')
        
        # Manually set connection state to simulate active connection
        client._is_connected = True
        client._first_message_received = True
        client._running = True
        
        # Close the client
        await client.close()
        
        # Check that connection state is reset
        assert client._is_connected == False
        assert client._running == False
        assert client.is_connected() == False
        
        logger.info("✓ Connection state reset on close")


class TestEnhancedHealthStatus:
    """Test enhanced health status reporting."""
    
    def test_health_status_disconnected(self):
        """Test health status shows disconnected when not connected."""
        client = WebSocketClient('kucoinfutures')
        
        status = client.get_health_status()
        
        assert 'connected' in status
        assert 'streaming' in status
        assert 'last_message_time' in status
        assert 'recent_errors_5min' in status
        
        assert status['connected'] == False
        assert status['streaming'] == False
        assert status['status'] == 'disconnected'
        assert status['last_message_time'] is None
        
        logger.info("✓ Health status shows disconnected correctly")
    
    def test_health_status_connecting(self):
        """Test health status shows connecting when connected but no messages yet."""
        client = WebSocketClient('kucoinfutures')
        
        # Simulate connection without messages
        client._is_connected = True
        client._first_message_received = False
        
        status = client.get_health_status()
        
        assert status['connected'] == True
        assert status['streaming'] == False
        assert status['status'] == 'connecting'
        
        logger.info("✓ Health status shows connecting correctly")
    
    def test_health_status_healthy(self):
        """Test health status shows healthy when fully connected and streaming."""
        client = WebSocketClient('kucoinfutures')
        
        # Simulate full connection
        client._is_connected = True
        client._first_message_received = True
        client._last_message_time = datetime.now()
        
        status = client.get_health_status()
        
        assert status['connected'] == True
        assert status['streaming'] == True
        assert status['status'] == 'healthy'
        assert status['last_message_time'] is not None
        
        logger.info("✓ Health status shows healthy correctly")
    
    def test_health_status_with_errors(self):
        """Test health status degrades with errors."""
        client = WebSocketClient('kucoinfutures')
        
        # Simulate connection with errors
        client._is_connected = True
        client._first_message_received = True
        
        # Add moderate errors
        client.parse_frame_errors['BTC/USDT:USDT_1m'] = 10
        
        status = client.get_health_status()
        
        assert status['status'] in ['warning', 'degraded']
        assert status['total_parse_frame_errors'] == 10
        
        logger.info("✓ Health status degrades with errors")
    
    def test_health_status_critical(self):
        """Test health status shows critical with many errors."""
        client = WebSocketClient('kucoinfutures')
        
        # Simulate connection with critical errors
        client._is_connected = True
        client._first_message_received = True
        client.parse_frame_errors['BTC/USDT:USDT_1m'] = 60
        
        status = client.get_health_status()
        
        assert status['status'] == 'critical'
        assert status['total_parse_frame_errors'] == 60
        
        logger.info("✓ Health status shows critical with many errors")
    
    def test_health_status_recent_errors_tracking(self):
        """Test that health status includes recent error count."""
        client = WebSocketClient('kucoinfutures')
        
        # Add some errors
        now = datetime.now()
        for i in range(3):
            client.error_history.append({
                'timestamp': now - timedelta(seconds=i * 30),
                'type': 'test_error',
                'message': f'Error {i}'
            })
        
        status = client.get_health_status()
        
        assert 'recent_errors_5min' in status
        assert status['recent_errors_5min'] == 3
        
        logger.info("✓ Recent errors tracking in health status works")


class TestIsConnectedMethod:
    """Test is_connected() method."""
    
    def test_is_connected_false_when_not_running(self):
        """Test is_connected returns False when not running."""
        client = WebSocketClient('kucoinfutures')
        
        client._is_connected = True
        client._first_message_received = True
        client._running = False
        
        assert client.is_connected() == False
        
        logger.info("✓ is_connected() returns False when not running")
    
    def test_is_connected_false_when_no_messages(self):
        """Test is_connected returns False when no messages received."""
        client = WebSocketClient('kucoinfutures')
        
        client._is_connected = True
        client._first_message_received = False
        client._running = True
        
        assert client.is_connected() == False
        
        logger.info("✓ is_connected() returns False when no messages received")
    
    def test_is_connected_false_when_disconnected(self):
        """Test is_connected returns False when disconnected."""
        client = WebSocketClient('kucoinfutures')
        
        client._is_connected = False
        client._first_message_received = True
        client._running = True
        
        assert client.is_connected() == False
        
        logger.info("✓ is_connected() returns False when disconnected")
    
    def test_is_connected_true_when_fully_connected(self):
        """Test is_connected returns True when fully connected."""
        client = WebSocketClient('kucoinfutures')
        
        client._is_connected = True
        client._first_message_received = True
        client._running = True
        
        assert client.is_connected() == True
        
        logger.info("✓ is_connected() returns True when fully connected")


class TestConnectionStateIntegration:
    """Integration tests for connection state tracking."""
    
    @pytest.mark.asyncio
    async def test_watch_ohlcv_updates_connection_state(self):
        """Test that watch_ohlcv updates connection state on success."""
        client = WebSocketClient('kucoinfutures')
        
        # Mock the exchange method to return data
        mock_ohlcv = [[1634567890000, 50000, 51000, 49000, 50500, 100]]
        client.ex.watch_ohlcv = AsyncMock(return_value=mock_ohlcv)
        
        # Call watch_ohlcv
        result = await client.watch_ohlcv('BTC/USDT:USDT', '1m')
        
        # Check connection state was updated
        assert client._is_connected == True
        assert client._first_message_received == True
        assert client._last_message_time is not None
        assert result == mock_ohlcv
        
        logger.info("✓ watch_ohlcv updates connection state on success")
    
    @pytest.mark.asyncio
    async def test_watch_ohlcv_logs_errors(self):
        """Test that watch_ohlcv logs errors to history."""
        client = WebSocketClient('kucoinfutures')
        
        # Mock the exchange method to raise an error
        client.ex.watch_ohlcv = AsyncMock(side_effect=Exception("Test error"))
        
        # Call watch_ohlcv and expect exception
        with pytest.raises(Exception):
            await client.watch_ohlcv('BTC/USDT:USDT', '1m')
        
        # Check error was logged
        assert len(client.error_history) > 0
        assert client.error_history[-1]['type'] == 'Exception'
        assert client._is_connected == False
        
        logger.info("✓ watch_ohlcv logs errors to history")
    
    @pytest.mark.asyncio
    async def test_watch_ticker_updates_connection_state(self):
        """Test that watch_ticker updates connection state on success."""
        client = WebSocketClient('kucoinfutures')
        
        # Mock the exchange method to return data
        mock_ticker = {'symbol': 'BTC/USDT:USDT', 'last': 50000}
        client.ex.watch_ticker = AsyncMock(return_value=mock_ticker)
        
        # Call watch_ticker
        result = await client.watch_ticker('BTC/USDT:USDT')
        
        # Check connection state was updated
        assert client._is_connected == True
        assert client._first_message_received == True
        assert client._last_message_time is not None
        assert result == mock_ticker
        
        logger.info("✓ watch_ticker updates connection state on success")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
