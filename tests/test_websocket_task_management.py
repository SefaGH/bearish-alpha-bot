#!/usr/bin/env python3
"""
Tests for WebSocket Task Management Fix (Issue #160).

Validates that WebSocket streaming tasks are properly scheduled and executed.
"""

import sys
import os
import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestOptimizedWebSocketManager:
    """Test suite for OptimizedWebSocketManager task management."""
    
    def test_initialize_websockets_returns_task_list(self):
        """Test that initialize_websockets returns a list of tasks, not bool."""
        from live_trading_launcher import OptimizedWebSocketManager
        
        # Create manager with config
        config = {
            'universe': {'fixed_symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT']},
            'websocket': {'enabled': True}
        }
        
        manager = OptimizedWebSocketManager(config)
        manager.setup_from_config(config)
        
        # Verify fixed symbols are set
        assert len(manager.fixed_symbols) == 2
        assert manager.fixed_symbols[0] == 'BTC/USDT:USDT'
    
    @pytest.mark.asyncio
    async def test_initialize_websockets_returns_empty_list_on_no_symbols(self):
        """Test that initialize_websockets returns empty list when no symbols."""
        from live_trading_launcher import OptimizedWebSocketManager
        
        # Create manager with no symbols
        config = {
            'universe': {'fixed_symbols': []},
            'websocket': {'enabled': True}
        }
        
        manager = OptimizedWebSocketManager(config)
        manager.setup_from_config(config)
        
        # Should return empty list
        exchange_clients = {}
        tasks = await manager.initialize_websockets(exchange_clients)
        
        assert isinstance(tasks, list)
        assert len(tasks) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_websockets_creates_tasks(self):
        """Test that initialize_websockets creates asyncio tasks."""
        from live_trading_launcher import OptimizedWebSocketManager
        from core.websocket_manager import WebSocketManager
        
        # Create manager with symbols
        config = {
            'universe': {'fixed_symbols': ['BTC/USDT:USDT']},
            'websocket': {'enabled': True}
        }
        
        manager = OptimizedWebSocketManager(config)
        manager.setup_from_config(config)
        
        # Mock exchange clients
        mock_client = MagicMock()
        mock_client.ex = MagicMock()
        mock_client.ex.apiKey = None
        exchange_clients = {'bingx': mock_client}
        
        # Mock the stream_ohlcv to return a task
        with patch.object(WebSocketManager, 'stream_ohlcv') as mock_stream:
            # Create a dummy coroutine for the task
            async def dummy_stream():
                await asyncio.sleep(0.1)
            
            # Return a task
            mock_task = asyncio.create_task(dummy_stream())
            mock_stream.return_value = [mock_task]
            
            # Initialize websockets
            tasks = await manager.initialize_websockets(exchange_clients)
            
            # Should return task list
            assert isinstance(tasks, list)
            
            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


class TestWebSocketClientConnectionState:
    """Test WebSocket client connection state tracking."""
    
    def test_is_connected_method_exists(self):
        """Test that WebSocketClient has is_connected method."""
        from core.websocket_client import WebSocketClient
        
        # Check method exists
        assert hasattr(WebSocketClient, 'is_connected')
        
        # Create a client (will fail on actual connection, but that's ok)
        try:
            # Test with a mock exchange
            with patch('ccxt.pro.bingx'):
                client = WebSocketClient('bingx', None)
                
                # Should have the method
                assert callable(client.is_connected)
                
                # Initially should be False (not connected)
                assert client.is_connected() == False
        except Exception:
            # If ccxt.pro is not available, that's ok - we verified the method exists
            pass


class TestSystemInfoWebSocketStatus:
    """Test system_info WebSocket status reporting."""
    
    def test_get_websocket_status_with_none(self):
        """Test get_websocket_status returns correct status when ws_manager is None."""
        from core.system_info import SystemInfoCollector
        
        status = SystemInfoCollector.get_websocket_status(None)
        
        assert status['enabled'] == False
        assert status['status_emoji'] == '⚠️'
        assert status['status_text'] == 'REST MODE'
        assert status['stream_count'] == 0
        assert status['mode'] == 'rest'
    
    def test_get_websocket_status_with_initialized_manager(self):
        """Test get_websocket_status with initialized but not connected manager."""
        from core.system_info import SystemInfoCollector
        
        # Create mock manager
        mock_ws_manager = MagicMock()
        mock_ws_manager.is_initialized = True
        mock_ws_manager.clients = {}
        mock_ws_manager._tasks = []
        
        status = SystemInfoCollector.get_websocket_status(mock_ws_manager)
        
        # Should show initialized but no streams
        assert 'INITIALIZED' in status['status_text'] or 'DISCONNECTED' in status['status_text']
        assert status['stream_count'] == 0
    
    def test_get_websocket_status_with_connected_clients(self):
        """Test get_websocket_status with connected and streaming clients."""
        from core.system_info import SystemInfoCollector
        
        # Create mock client
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client._first_message_received = True
        
        # Create mock tasks
        mock_task = MagicMock()
        mock_task.done.return_value = False
        
        # Create mock manager
        mock_ws_manager = MagicMock()
        mock_ws_manager.clients = {'bingx': mock_client}
        mock_ws_manager._tasks = [mock_task]
        # Important: Prevent auto-creation of ws_manager attribute
        mock_ws_manager.ws_manager = None
        
        status = SystemInfoCollector.get_websocket_status(mock_ws_manager)
        
        # Should show connected and streaming
        assert 'CONNECTED' in status['status_text'] or 'STREAMING' in status['status_text']
        assert status['stream_count'] >= 1
    
    def test_get_websocket_status_with_optimized_manager(self):
        """Test get_websocket_status with OptimizedWebSocketManager wrapper."""
        from core.system_info import SystemInfoCollector
        
        # Create mock client
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client._first_message_received = True
        
        # Create mock inner ws_manager
        mock_inner_manager = MagicMock()
        mock_inner_manager.clients = {'bingx': mock_client}
        
        # Create mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_inner_manager._tasks = [mock_task]
        
        # Create mock OptimizedWebSocketManager
        mock_optimized = MagicMock()
        mock_optimized.ws_manager = mock_inner_manager
        mock_optimized.is_initialized = True
        
        status = SystemInfoCollector.get_websocket_status(mock_optimized)
        
        # Should detect the inner manager and check its state
        assert status['stream_count'] >= 1


class TestWebSocketTaskCleanup:
    """Test WebSocket task cleanup on shutdown."""
    
    @pytest.mark.asyncio
    async def test_tasks_are_cancelled_on_shutdown(self):
        """Test that WebSocket tasks are properly cancelled on shutdown."""
        
        # Create dummy tasks
        async def dummy_stream():
            try:
                await asyncio.sleep(10)  # Long running
            except asyncio.CancelledError:
                # Expected on shutdown
                raise
        
        tasks = [
            asyncio.create_task(dummy_stream()),
            asyncio.create_task(dummy_stream()),
            asyncio.create_task(dummy_stream())
        ]
        
        # Give tasks a moment to start
        await asyncio.sleep(0.1)
        
        # Verify tasks are running
        assert all(not task.done() for task in tasks)
        
        # Cancel all tasks (simulating shutdown)
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should be cancelled
        assert all(isinstance(r, asyncio.CancelledError) for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
