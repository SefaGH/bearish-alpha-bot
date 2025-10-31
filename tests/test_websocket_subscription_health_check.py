"""
Integration test for WebSocket subscription health check fix (Issue #259)

This test demonstrates that subscription confirmations are now properly
counted as active streams, allowing health checks to pass.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from core.websocket_manager import WebSocketManager


class MockBingXWebSocket:
    """Mock BingX WebSocket that simulates subscription behavior."""
    
    def __init__(self):
        self.subscriptions = {}
        self.pending_subscriptions = {}
        self._is_connected = False
        self.message_count = 0
        self.last_message_time = None
        self._ws_thread = Mock()
        self._ws_thread.is_alive = Mock(return_value=True)
    
    def is_connected(self):
        return self._is_connected
    
    def add_subscription(self, sub_id: str):
        """Simulate a subscription confirmation."""
        self.subscriptions[sub_id] = {'confirmed': True}


class MockWebSocketClientWithSubscriptions:
    """Mock WebSocket client that delegates to BingXWebSocket."""
    
    def __init__(self):
        self.bingx_ws = MockBingXWebSocket()
        self._is_connected = False
    
    def get_subscription_count(self) -> int:
        """Return actual subscription count from underlying BingX client."""
        return len(self.bingx_ws.subscriptions)
    
    def get_health_status(self):
        """Return health status including subscription count."""
        return {
            'connected': self.bingx_ws._is_connected,
            'listen_task_status': 'running' if self.bingx_ws._ws_thread.is_alive() else 'stopped',
            'subscriptions': len(self.bingx_ws.subscriptions),
            'message_count': self.bingx_ws.message_count,
            'last_message_time': self.bingx_ws.last_message_time
        }


def test_health_check_passes_after_subscriptions_confirmed():
    """
    Test that health check passes once subscription confirmations are received.
    
    This simulates the real-world scenario where:
    1. WebSocket connections are established
    2. Subscription requests are sent
    3. Subscription confirmations arrive
    4. Health check should now see active streams and pass
    """
    # Setup WebSocketManager with mock client
    manager = WebSocketManager(exchanges={}, config={})
    mock_client = MockWebSocketClientWithSubscriptions()
    manager.clients = {'bingx': mock_client}
    
    # Initially, no subscriptions (health check would fail)
    assert manager.get_active_stream_count() == 0
    
    # Simulate subscription confirmations arriving
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_1m')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_5m')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_30m')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_1h')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_4h')
    
    # Now health check should pass - it sees 5 active streams
    active_count = manager.get_active_stream_count()
    assert active_count == 5, f"Expected 5 active streams after subscriptions, got {active_count}"
    
    # Verify health status reflects subscriptions
    health = mock_client.get_health_status()
    assert health['subscriptions'] == 5


def test_health_check_reflects_multiple_symbols():
    """Test that health check counts subscriptions across multiple symbols."""
    manager = WebSocketManager(exchanges={}, config={})
    mock_client = MockWebSocketClientWithSubscriptions()
    manager.clients = {'bingx': mock_client}
    
    # Simulate subscriptions for multiple symbols across multiple timeframes
    symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
    timeframes = ['1m', '5m', '30m', '1h', '4h']
    
    for symbol in symbols:
        for tf in timeframes:
            mock_client.bingx_ws.add_subscription(f'{symbol}@kline_{tf}')
    
    # Should have 3 symbols * 5 timeframes = 15 subscriptions
    expected_count = len(symbols) * len(timeframes)
    actual_count = manager.get_active_stream_count()
    assert actual_count == expected_count, f"Expected {expected_count} subscriptions, got {actual_count}"


def test_health_check_handles_gradual_subscription_confirmations():
    """Test that count updates as subscription confirmations arrive gradually."""
    manager = WebSocketManager(exchanges={}, config={})
    mock_client = MockWebSocketClientWithSubscriptions()
    manager.clients = {'bingx': mock_client}
    
    # Initially zero
    assert manager.get_active_stream_count() == 0
    
    # First confirmation arrives
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_1m')
    assert manager.get_active_stream_count() == 1
    
    # Second confirmation arrives
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_5m')
    assert manager.get_active_stream_count() == 2
    
    # More confirmations arrive
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_30m')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_1h')
    mock_client.bingx_ws.add_subscription('BTC-USDT@kline_4h')
    assert manager.get_active_stream_count() == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
