"""
Test WebSocket Architecture Fixes
Tests for issue: WebSocketManager/BingX integration and trade loop compatibility
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.core.websocket_client import WebSocketClient
from src.core.websocket_client_bingx import WebSocketClient as BingxClient
from src.core.websocket_manager import WebSocketManager
from src.core.production_coordinator import ProductionCoordinator


class TestWebSocketClientDiagnostics:
    """Test diagnostic fields initialization in WebSocketClient"""
    
    def test_diagnostic_fields_initialized(self):
        """Test that all diagnostic fields are properly initialized"""
        client = WebSocketClient('kucoinfutures', None)
        
        # Check all diagnostic fields exist
        assert hasattr(client, 'error_history')
        assert hasattr(client, 'max_error_history')
        assert hasattr(client, 'parse_frame_errors')
        assert hasattr(client, 'max_parse_frame_retries')
        assert hasattr(client, 'reconnect_delay')
        assert hasattr(client, 'reconnect_count')
        assert hasattr(client, 'last_reconnect')
        assert hasattr(client, 'use_rest_fallback')
        
        # Check types and initial values
        assert isinstance(client.error_history, list)
        assert client.max_error_history == 100
        assert isinstance(client.parse_frame_errors, dict)
        assert client.max_parse_frame_retries == 3
        assert client.reconnect_delay == 5.0
        assert client.reconnect_count == 0
        assert client.last_reconnect is None
        assert isinstance(client.use_rest_fallback, bool)
    
    def test_tasks_list_initialized(self):
        """Test that tasks list is initialized for tracking"""
        client = WebSocketClient('kucoinfutures', None)
        assert hasattr(client, '_tasks')
        assert isinstance(client._tasks, list)
        assert len(client._tasks) == 0


class TestBingXWebSocketClientDiagnostics:
    """Test diagnostic fields initialization in BingX-specific client"""
    
    def test_bingx_diagnostic_fields_initialized(self):
        """Test that BingX client has all diagnostic fields"""
        client = BingxClient('bingx', None)
        
        # Check all diagnostic fields exist
        assert hasattr(client, 'error_history')
        assert hasattr(client, 'max_error_history')
        assert hasattr(client, 'parse_frame_errors')
        assert hasattr(client, 'max_parse_frame_retries')
        assert hasattr(client, 'reconnect_delay')
        assert hasattr(client, 'reconnect_count')
        assert hasattr(client, 'last_reconnect')
        assert hasattr(client, 'use_rest_fallback')
        
        # Check initial values match specification
        assert client.max_error_history == 100
        assert client.max_parse_frame_retries == 3
        assert client.reconnect_delay == 5.0
        assert client.reconnect_count == 0


class TestWebSocketManagerWrappers:
    """Test wrapper methods in WebSocketManager"""
    
    def test_ohlcv_wrapper_method_exists(self):
        """Test that _make_ohlcv_wrapper method exists"""
        manager = WebSocketManager(exchanges={'test': None})
        assert hasattr(manager, '_make_ohlcv_wrapper')
        assert callable(manager._make_ohlcv_wrapper)
    
    def test_ticker_wrapper_method_exists(self):
        """Test that _make_ticker_wrapper method exists"""
        manager = WebSocketManager(exchanges={'test': None})
        assert hasattr(manager, '_make_ticker_wrapper')
        assert callable(manager._make_ticker_wrapper)
    
    @pytest.mark.asyncio
    async def test_ohlcv_wrapper_calls_watch_ohlcv(self):
        """Test that OHLCV wrapper properly calls watch_ohlcv"""
        manager = WebSocketManager(exchanges={'test': None})
        
        # Create a mock client
        mock_client = Mock()
        mock_client.watch_ohlcv = AsyncMock(return_value=[[1, 2, 3, 4, 5, 6]])
        mock_client.reconnect_delay = 1
        mock_client.name = 'test'
        
        # Create wrapper
        wrapper = manager._make_ohlcv_wrapper(
            mock_client, 'BTC/USDT', '1m', None, max_iterations=1
        )
        
        # Run wrapper
        manager._running = True
        await wrapper()
        
        # Verify watch_ohlcv was called
        mock_client.watch_ohlcv.assert_called()
    
    @pytest.mark.asyncio
    async def test_ticker_wrapper_calls_watch_ticker(self):
        """Test that ticker wrapper properly calls watch_ticker"""
        manager = WebSocketManager(exchanges={'test': None})
        
        # Create a mock client
        mock_client = Mock()
        mock_client.watch_ticker = AsyncMock(return_value={'last': 50000})
        mock_client.reconnect_delay = 1
        mock_client.name = 'test'
        
        # Create wrapper
        wrapper = manager._make_ticker_wrapper(
            mock_client, 'BTC/USDT', None, max_iterations=1
        )
        
        # Run wrapper
        manager._running = True
        await wrapper()
        
        # Verify watch_ticker was called
        mock_client.watch_ticker.assert_called()


class TestProductionCoordinatorExchangeKeys:
    """Test exchange key normalization in ProductionCoordinator"""
    
    @pytest.mark.asyncio
    async def test_exchange_keys_normalized_to_lowercase(self):
        """Test that exchange keys are normalized to lowercase"""
        coordinator = ProductionCoordinator()
        
        # Create mock exchange clients with mixed case keys
        mock_clients = {
            'BingX': Mock(),
            'KuCoinFutures': Mock(),
            'Binance': Mock()
        }
        
        # Initialize with mixed case keys
        result = await coordinator.initialize_production_system(
            exchange_clients=mock_clients,
            mode='paper'
        )
        
        # Check that keys are normalized
        assert 'bingx' in coordinator.exchange_clients
        assert 'kucoinfutures' in coordinator.exchange_clients
        assert 'binance' in coordinator.exchange_clients
        
        # Check that original case keys are NOT present
        assert 'BingX' not in coordinator.exchange_clients
        assert 'KuCoinFutures' not in coordinator.exchange_clients
        assert 'Binance' not in coordinator.exchange_clients


class TestWebSocketManagerClientSelection:
    """Test exchange-specific client selection"""
    
    def test_bingx_client_selection_attempted(self):
        """Test that BingX-specific client is attempted to be loaded"""
        # This should attempt to import websocket_client_bingx
        with patch('src.core.websocket_manager.logger') as mock_logger:
            manager = WebSocketManager(exchanges={'bingx': None})
            
            # Check that BingX client was initialized
            assert 'bingx' in manager.clients
            
            # The log message should indicate BingX initialization
            # (This verifies the flow went through BingX-specific path)
            calls = [str(call) for call in mock_logger.info.call_args_list]
            bingx_init_logged = any('bingx' in str(call).lower() for call in calls)
            assert bingx_init_logged


class TestStreamOhlcvFallback:
    """Test stream_ohlcv fallback to wrapper when watch_ohlcv_loop is not available"""
    
    @pytest.mark.asyncio
    async def test_stream_ohlcv_uses_loop_when_available(self):
        """Test that stream_ohlcv prefers watch_ohlcv_loop when available"""
        manager = WebSocketManager(exchanges={'test': None})
        
        # Create a mock client WITH watch_ohlcv_loop
        mock_client = Mock()
        mock_client.name = 'test'
        mock_client.watch_ohlcv_loop = AsyncMock()
        manager.clients['test'] = mock_client
        
        # Call stream_ohlcv
        manager._running = True
        tasks = await manager.stream_ohlcv(
            {'test': ['BTC/USDT']}, 
            '1m', 
            max_iterations=1
        )
        
        # Verify watch_ohlcv_loop was used
        assert len(tasks) > 0
        # Note: we can't directly check if watch_ohlcv_loop was called here
        # because the task is created but not necessarily executed yet
    
    @pytest.mark.asyncio
    async def test_stream_ohlcv_uses_wrapper_when_loop_not_available(self):
        """Test that stream_ohlcv falls back to wrapper when watch_ohlcv_loop not available"""
        manager = WebSocketManager(exchanges={'test': None})
        
        # Create a mock client WITHOUT watch_ohlcv_loop
        mock_client = Mock()
        mock_client.name = 'test'
        mock_client.watch_ohlcv = AsyncMock(return_value=[[1, 2, 3, 4, 5, 6]])
        # Explicitly remove watch_ohlcv_loop to force wrapper usage
        if hasattr(mock_client, 'watch_ohlcv_loop'):
            delattr(mock_client, 'watch_ohlcv_loop')
        manager.clients['test'] = mock_client
        
        # Mock the wrapper method
        manager._make_ohlcv_wrapper = Mock(return_value=AsyncMock())
        
        # Call stream_ohlcv
        manager._running = True
        tasks = await manager.stream_ohlcv(
            {'test': ['BTC/USDT']}, 
            '1m', 
            max_iterations=1
        )
        
        # Verify wrapper was created
        assert manager._make_ohlcv_wrapper.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
