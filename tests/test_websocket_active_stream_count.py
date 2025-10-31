"""
Test for WebSocket Active Stream Count Fix (Issue #259)

This test verifies that WebSocket subscription confirmations are properly
counted as active streams by the WebSocketManager.
"""
import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from core.websocket_manager import WebSocketManager


class MockWebSocketClient:
    """Mock WebSocket client for testing."""
    
    def __init__(self, subscription_count=0):
        self._subscription_count = subscription_count
        self.bingx_ws = Mock()
        self.bingx_ws.subscriptions = {f"sub_{i}": {} for i in range(subscription_count)}
    
    def get_subscription_count(self) -> int:
        """Return the mock subscription count."""
        return len(self.bingx_ws.subscriptions)


def test_get_active_stream_count_with_no_clients():
    """Test that get_active_stream_count returns 0 when there are no clients."""
    manager = WebSocketManager(exchanges={}, config={})
    assert manager.get_active_stream_count() == 0


def test_get_active_stream_count_with_single_client():
    """Test that get_active_stream_count correctly counts subscriptions from a single client."""
    # Create manager with mock client that has 5 subscriptions
    mock_client = MockWebSocketClient(subscription_count=5)
    manager = WebSocketManager(exchanges={}, config={})
    manager.clients = {'bingx': mock_client}
    
    # Verify the count
    assert manager.get_active_stream_count() == 5


def test_get_active_stream_count_with_multiple_clients():
    """Test that get_active_stream_count correctly aggregates subscriptions from multiple clients."""
    # Create manager with two mock clients
    mock_client1 = MockWebSocketClient(subscription_count=3)
    mock_client2 = MockWebSocketClient(subscription_count=7)
    
    manager = WebSocketManager(exchanges={}, config={})
    manager.clients = {
        'bingx': mock_client1,
        'binance': mock_client2
    }
    
    # Verify the aggregated count
    assert manager.get_active_stream_count() == 10


def test_get_active_stream_count_with_client_without_method():
    """Test that get_active_stream_count handles clients without get_subscription_count method gracefully."""
    # Create a client without get_subscription_count method
    mock_client_without_method = Mock()
    mock_client_without_method.get_subscription_count = None  # Not a method
    
    mock_client_with_method = MockWebSocketClient(subscription_count=5)
    
    manager = WebSocketManager(exchanges={}, config={})
    manager.clients = {
        'broken_client': mock_client_without_method,
        'working_client': mock_client_with_method
    }
    
    # Should still count the working client
    assert manager.get_active_stream_count() == 5


def test_get_active_stream_count_after_subscription_added():
    """Test that count updates when subscriptions are added dynamically."""
    mock_client = MockWebSocketClient(subscription_count=3)
    manager = WebSocketManager(exchanges={}, config={})
    manager.clients = {'bingx': mock_client}
    
    # Initial count
    assert manager.get_active_stream_count() == 3
    
    # Add more subscriptions
    mock_client.bingx_ws.subscriptions['sub_3'] = {}
    mock_client.bingx_ws.subscriptions['sub_4'] = {}
    
    # Verify updated count
    assert manager.get_active_stream_count() == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
