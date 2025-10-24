"""
Simple smoke test for WebSocket integration fixes
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.core.websocket_manager import WebSocketManager
from src.core.websocket_client import WebSocketClient
from src.core.websocket_client_bingx import WebSocketClient as BingxClient


@pytest.mark.asyncio
async def test_websocket_manager_initialization():
    """Smoke test: WebSocketManager can be initialized with mixed case keys"""
    exchanges = {
        'BingX': {'apiKey': 'test', 'secret': 'test'},
        'KuCoinFutures': None
    }
    
    manager = WebSocketManager(exchanges=exchanges)
    
    # Check that clients are created (keys should be lowercase internally)
    assert len(manager.clients) == 2
    assert 'bingx' in manager.clients or 'kucoinfutures' in manager.clients


@pytest.mark.asyncio
async def test_websocket_client_close_cleanup():
    """Smoke test: WebSocketClient can track and cleanup tasks"""
    client = WebSocketClient('kucoinfutures', None)
    
    # Add a real async task
    async def dummy_task():
        await asyncio.sleep(10)  # Long running task
    
    task = asyncio.create_task(dummy_task())
    client._tasks.append(task)
    
    # Close should cancel tasks
    await client.close()
    
    # Verify task was cancelled
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_bingx_client_has_loop_methods():
    """Smoke test: BingX client has watch_ohlcv_loop and watch_ticker_loop"""
    client = BingxClient('bingx', None)
    
    assert hasattr(client, 'watch_ohlcv_loop')
    assert callable(client.watch_ohlcv_loop)
    assert hasattr(client, 'watch_ticker_loop')
    assert callable(client.watch_ticker_loop)


@pytest.mark.asyncio
async def test_websocket_manager_stream_with_fallback():
    """Smoke test: WebSocketManager handles clients with and without loop methods"""
    manager = WebSocketManager(exchanges={'test': None})
    
    # Create two types of clients
    client_with_loop = Mock()
    client_with_loop.name = 'test1'
    client_with_loop.watch_ohlcv_loop = AsyncMock()
    
    client_without_loop = Mock()
    client_without_loop.name = 'test2'
    client_without_loop.watch_ohlcv = AsyncMock(return_value=[[1, 2, 3, 4, 5, 6]])
    # Ensure watch_ohlcv_loop doesn't exist
    if hasattr(client_without_loop, 'watch_ohlcv_loop'):
        delattr(client_without_loop, 'watch_ohlcv_loop')
    
    manager.clients['test1'] = client_with_loop
    manager.clients['test2'] = client_without_loop
    
    # Both should work
    manager._running = True
    tasks1 = await manager.stream_ohlcv({'test1': ['BTC/USDT']}, '1m', max_iterations=1)
    tasks2 = await manager.stream_ohlcv({'test2': ['BTC/USDT']}, '1m', max_iterations=1)
    
    assert len(tasks1) > 0
    assert len(tasks2) > 0


@pytest.mark.asyncio
async def test_get_health_status_no_attribute_error():
    """Smoke test: get_health_status doesn't raise AttributeError"""
    client = WebSocketClient('kucoinfutures', None)
    
    # Should not raise AttributeError
    status = client.get_health_status()
    
    # Should return a dict with expected keys
    assert isinstance(status, dict)
    assert 'exchange' in status
    assert 'status' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
