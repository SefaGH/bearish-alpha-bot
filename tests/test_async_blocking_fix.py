"""
Test to verify the fix for blocking I/O in async context (bot freeze issue).

This test verifies that:
1. _fetch_ohlcv is properly async and doesn't block the event loop
2. process_symbol properly awaits async operations
3. Timeout protection works correctly
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator


@pytest.mark.asyncio
async def test_fetch_ohlcv_is_async():
    """Test that _fetch_ohlcv runs in thread pool and doesn't block."""
    coordinator = ProductionCoordinator()
    
    # Mock exchange client
    mock_client = Mock()
    mock_client.ohlcv = Mock(return_value=[
        [1234567890, 100, 105, 98, 102, 1000],
        [1234567900, 102, 106, 100, 104, 1100],
    ])
    
    # Mock _ohlcv_to_dataframe
    coordinator._ohlcv_to_dataframe = Mock(return_value=pd.DataFrame({
        'open': [100, 102],
        'high': [105, 106],
        'low': [98, 100],
        'close': [102, 104],
        'volume': [1000, 1100]
    }))
    
    # Test that the method is async
    result = coordinator._fetch_ohlcv(mock_client, "BTC/USDT", "1h")
    assert asyncio.iscoroutine(result), "_fetch_ohlcv should return a coroutine"
    
    # Await the result
    df = await result
    assert df is not None
    assert len(df) == 2
    mock_client.ohlcv.assert_called_once_with("BTC/USDT", "1h", limit=200)


@pytest.mark.asyncio
async def test_fetch_ohlcv_handles_blocking_call():
    """Test that _fetch_ohlcv properly handles blocking calls without freezing."""
    coordinator = ProductionCoordinator()
    
    # Create a mock client that simulates a slow blocking call
    mock_client = Mock()
    
    def slow_blocking_call(*args, **kwargs):
        """Simulate a slow blocking network call."""
        import time
        time.sleep(0.1)  # Simulate network delay
        return [[1234567890, 100, 105, 98, 102, 1000]]
    
    mock_client.ohlcv = Mock(side_effect=slow_blocking_call)
    coordinator._ohlcv_to_dataframe = Mock(return_value=pd.DataFrame({
        'open': [100], 'high': [105], 'low': [98], 'close': [102], 'volume': [1000]
    }))
    
    # This should complete without blocking the event loop
    start = asyncio.get_event_loop().time()
    
    # Run multiple fetches concurrently to verify non-blocking behavior
    tasks = [
        coordinator._fetch_ohlcv(mock_client, f"PAIR{i}/USDT", "1h")
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    
    elapsed = asyncio.get_event_loop().time() - start
    
    # Should complete in roughly 0.1s (parallel) not 0.3s (sequential)
    assert elapsed < 0.25, f"Took {elapsed}s - tasks may have run sequentially (blocking)"
    assert all(df is not None for df in results)


@pytest.mark.asyncio
async def test_process_symbol_timeout():
    """Test that process_symbol has timeout protection."""
    coordinator = ProductionCoordinator()
    coordinator.exchange_clients = {}
    coordinator.websocket_manager = None
    coordinator.processed_symbols_count = 0
    
    # This should return None quickly without data
    result = await coordinator.process_symbol("BTC/USDT")
    assert result is None


@pytest.mark.asyncio
async def test_process_trading_loop_timeout():
    """Test that _process_trading_loop has timeout protection for each symbol."""
    coordinator = ProductionCoordinator()
    coordinator.active_symbols = ["BTC/USDT", "ETH/USDT"]
    coordinator.processed_symbols_count = 0
    coordinator.exchange_clients = {}
    coordinator.websocket_manager = None
    
    # Mock process_symbol to simulate a hanging call
    async def hanging_process(symbol):
        if symbol == "BTC/USDT":
            await asyncio.sleep(100)  # Simulate hang
        return None
    
    with patch.object(coordinator, 'process_symbol', side_effect=hanging_process):
        # This should complete within reasonable time due to timeout
        start = asyncio.get_event_loop().time()
        await coordinator._process_trading_loop()
        elapsed = asyncio.get_event_loop().time() - start
        
        # Should timeout after 30s for BTC/USDT, then process ETH/USDT quickly
        assert elapsed < 35, f"Took {elapsed}s - timeout may not be working"


@pytest.mark.asyncio
async def test_fetch_ohlcv_error_handling():
    """Test that _fetch_ohlcv handles errors gracefully."""
    coordinator = ProductionCoordinator()
    
    # Mock client that raises an exception
    mock_client = Mock()
    mock_client.ohlcv = Mock(side_effect=Exception("Network error"))
    
    # Should return None on error, not raise
    result = await coordinator._fetch_ohlcv(mock_client, "BTC/USDT", "1h")
    assert result is None


@pytest.mark.asyncio
async def test_multiple_concurrent_fetches():
    """Test that multiple fetches can run concurrently without blocking."""
    coordinator = ProductionCoordinator()
    
    # Create multiple mock clients
    mock_clients = []
    for i in range(5):
        client = Mock()
        client.ohlcv = Mock(return_value=[
            [1234567890 + i, 100 + i, 105 + i, 98 + i, 102 + i, 1000 + i]
        ])
        mock_clients.append(client)
    
    coordinator._ohlcv_to_dataframe = Mock(return_value=pd.DataFrame({
        'open': [100], 'high': [105], 'low': [98], 'close': [102], 'volume': [1000]
    }))
    
    # Run all fetches concurrently
    start = asyncio.get_event_loop().time()
    tasks = [
        coordinator._fetch_ohlcv(client, f"PAIR{i}/USDT", "1h")
        for i, client in enumerate(mock_clients)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start
    
    # All should complete successfully
    assert len(results) == 5
    assert all(df is not None for df in results)
    
    # Should complete quickly (concurrently)
    assert elapsed < 1.0, f"Took {elapsed}s - may be running sequentially"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
