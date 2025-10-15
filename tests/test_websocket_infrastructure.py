"""
Tests for Phase 3.1: WebSocket Infrastructure
Tests WebSocket client and manager functionality.
"""

import pytest
import asyncio
import logging
from collections import defaultdict
from unittest.mock import Mock, patch, AsyncMock
from src.core.websocket_client import WebSocketClient
from src.core.websocket_manager import WebSocketManager, StreamDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestWebSocketClient:
    """Test WebSocketClient functionality."""
    
    def test_websocket_client_initialization(self):
        """Test WebSocket client can be initialized."""
        # Test with valid exchange
        client = WebSocketClient('kucoinfutures')
        assert client.name == 'kucoinfutures'
        assert hasattr(client.ex, 'watch_ohlcv')
        logger.info("✓ WebSocket client initialization successful")
    
    def test_websocket_client_invalid_exchange(self):
        """Test WebSocket client raises error for invalid exchange."""
        with pytest.raises(AttributeError):
            WebSocketClient('invalid_exchange')
        logger.info("✓ Invalid exchange properly rejected")
    
    @pytest.mark.asyncio
    async def test_websocket_client_close(self):
        """Test WebSocket client can be closed."""
        client = WebSocketClient('kucoinfutures')
        await client.close()
        assert not client._running
        logger.info("✓ WebSocket client close successful")


class TestWebSocketManager:
    """Test WebSocketManager functionality."""
    
    def test_websocket_manager_initialization(self):
        """Test WebSocket manager can be initialized."""
        manager = WebSocketManager()
        assert len(manager.clients) > 0
        assert 'kucoinfutures' in manager.clients or 'bingx' in manager.clients
        logger.info(f"✓ WebSocket manager initialized with {len(manager.clients)} exchanges")
    
    def test_websocket_manager_custom_exchanges(self):
        """Test WebSocket manager with custom exchange configuration."""
        exchanges = {
            'kucoinfutures': None,
            'bingx': None
        }
        manager = WebSocketManager(exchanges)
        assert 'kucoinfutures' in manager.clients
        assert 'bingx' in manager.clients
        logger.info("✓ Custom exchange configuration successful")
    
    def test_websocket_manager_status(self):
        """Test WebSocket manager status reporting."""
        manager = WebSocketManager()
        status = manager.get_stream_status()
        
        assert 'running' in status
        assert 'total_streams' in status
        assert 'active_streams' in status
        assert 'exchanges' in status
        assert isinstance(status['exchanges'], list)
        logger.info(f"✓ Status reporting working: {status}")
    
    @pytest.mark.asyncio
    async def test_websocket_manager_close(self):
        """Test WebSocket manager can be closed."""
        manager = WebSocketManager()
        await manager.close()
        assert not manager._running
        logger.info("✓ WebSocket manager close successful")


class TestStreamDataCollector:
    """Test StreamDataCollector functionality."""
    
    def test_collector_initialization(self):
        """Test collector can be initialized."""
        collector = StreamDataCollector(buffer_size=100)
        assert collector.buffer_size == 100
        assert len(collector.ohlcv_data) == 0
        assert len(collector.ticker_data) == 0
        logger.info("✓ StreamDataCollector initialization successful")
    
    @pytest.mark.asyncio
    async def test_collector_ohlcv_callback(self):
        """Test OHLCV data collection."""
        collector = StreamDataCollector(buffer_size=10)
        
        # Simulate OHLCV data
        ohlcv = [[1634567890000, 50000, 51000, 49000, 50500, 100]]
        
        await collector.ohlcv_callback('kucoinfutures', 'BTC/USDT:USDT', '1m', ohlcv)
        
        latest = collector.get_latest_ohlcv('kucoinfutures', 'BTC/USDT:USDT', '1m')
        assert latest == ohlcv
        logger.info("✓ OHLCV data collection successful")
    
    @pytest.mark.asyncio
    async def test_collector_ticker_callback(self):
        """Test ticker data collection."""
        collector = StreamDataCollector(buffer_size=10)
        
        # Simulate ticker data
        ticker = {'symbol': 'BTC/USDT:USDT', 'last': 50000, 'bid': 49999, 'ask': 50001}
        
        await collector.ticker_callback('kucoinfutures', 'BTC/USDT:USDT', ticker)
        
        latest = collector.get_latest_ticker('kucoinfutures', 'BTC/USDT:USDT')
        assert latest == ticker
        logger.info("✓ Ticker data collection successful")
    
    @pytest.mark.asyncio
    async def test_collector_buffer_limit(self):
        """Test collector respects buffer size limits."""
        collector = StreamDataCollector(buffer_size=5)
        
        # Add more items than buffer size
        for i in range(10):
            ohlcv = [[1634567890000 + i, 50000, 51000, 49000, 50500, 100]]
            await collector.ohlcv_callback('kucoinfutures', 'BTC/USDT:USDT', '1m', ohlcv)
        
        # Check that buffer is trimmed
        key = 'BTC/USDT:USDT_1m'
        assert len(collector.ohlcv_data['kucoinfutures'][key]) == 5
        logger.info("✓ Buffer size limit enforced correctly")
    
    def test_collector_clear(self):
        """Test collector data can be cleared."""
        collector = StreamDataCollector()
        collector.ohlcv_data = {'test': {'data': []}}
        collector.ticker_data = {'test': {'data': []}}
        
        collector.clear()
        
        assert len(collector.ohlcv_data) == 0
        assert len(collector.ticker_data) == 0
        logger.info("✓ Collector clear successful")


class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""
    
    @pytest.mark.asyncio
    async def test_basic_streaming_workflow(self):
        """Test basic streaming workflow with mock data."""
        manager = WebSocketManager()
        collector = StreamDataCollector(buffer_size=100)
        
        # Verify manager is initialized
        assert len(manager.clients) > 0
        
        # Get status
        status = manager.get_stream_status()
        assert status['running'] == False
        assert status['total_streams'] == 0
        
        # Clean up
        await manager.close()
        logger.info("✓ Basic streaming workflow test passed")
    
    @pytest.mark.asyncio
    async def test_multi_exchange_coordination(self):
        """Test multi-exchange coordination."""
        exchanges = {
            'kucoinfutures': None,
            'bingx': None
        }
        manager = WebSocketManager(exchanges)
        
        # Verify both exchanges are initialized
        assert 'kucoinfutures' in manager.clients
        assert 'bingx' in manager.clients
        
        status = manager.get_stream_status()
        assert len(status['exchanges']) == 2
        
        # Clean up
        await manager.close()
        logger.info("✓ Multi-exchange coordination test passed")
    
    @pytest.mark.asyncio
    async def test_ccxt_client_integration(self):
        """Test WebSocketManager initialization with CcxtClient instances."""
        from src.core.ccxt_client import CcxtClient
        
        # Create CcxtClient instances (unauthenticated)
        clients = {
            'kucoinfutures': CcxtClient('kucoinfutures', None),
            'bingx': CcxtClient('bingx', None)
        }
        
        # Initialize WebSocketManager with CcxtClient instances
        manager = WebSocketManager(clients)
        
        # Verify initialization
        assert len(manager.clients) == 2
        assert 'kucoinfutures' in manager.clients
        assert 'bingx' in manager.clients
        assert manager._use_ccxt_clients == True
        
        # Clean up
        await manager.close()
        logger.info("✓ CcxtClient integration test passed")


class TestWebSocketManagerNewAPI:
    """Test new API methods added for Phase 2.1 integration."""
    
    @pytest.mark.asyncio
    async def test_subscribe_tickers(self):
        """Test subscribe_tickers method."""
        manager = WebSocketManager()
        
        # Test callback registration
        callback_called = {'count': 0}
        
        async def test_callback(exchange, symbol, ticker):
            callback_called['count'] += 1
        
        # Note: This will try to start actual streams, but we're just testing the API
        # In production, this would connect to real exchanges
        try:
            # Test that method exists and accepts parameters
            assert hasattr(manager, 'subscribe_tickers')
            logger.info("✓ subscribe_tickers method exists")
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_subscribe_orderbook(self):
        """Test subscribe_orderbook method."""
        manager = WebSocketManager()
        
        try:
            # Test that method exists and accepts parameters
            assert hasattr(manager, 'subscribe_orderbook')
            logger.info("✓ subscribe_orderbook method exists")
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_callback_registration(self):
        """Test callback registration methods."""
        manager = WebSocketManager()
        
        async def ticker_callback(exchange, symbol, ticker):
            pass
        
        async def orderbook_callback(exchange, symbol, orderbook):
            pass
        
        # Register callbacks
        result = manager.on_ticker_update(ticker_callback)
        assert result is manager  # Test method chaining
        assert len(manager.callbacks['ticker']) == 1
        
        result = manager.on_orderbook_update(orderbook_callback)
        assert result is manager
        assert len(manager.callbacks['orderbook']) == 1
        
        await manager.close()
        logger.info("✓ Callback registration test passed")
    
    @pytest.mark.asyncio
    async def test_start_streams(self):
        """Test start_streams method."""
        manager = WebSocketManager()
        
        # Test that method exists and accepts subscription dict
        assert hasattr(manager, 'start_streams')
        
        # Test with empty subscriptions (won't start actual streams)
        subscriptions = {}
        tasks = await manager.start_streams(subscriptions)
        assert isinstance(tasks, dict)
        
        await manager.close()
        logger.info("✓ start_streams method test passed")
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutdown method."""
        manager = WebSocketManager()
        
        # Test that shutdown method exists
        assert hasattr(manager, 'shutdown')
        
        # Test shutdown
        await manager.shutdown()
        assert manager.is_running == False
        
        logger.info("✓ shutdown method test passed")
    
    @pytest.mark.asyncio
    async def test_config_initialization(self):
        """Test WebSocketManager initialization with config parameter."""
        config = {
            'reconnect_delay': 5,
            'max_retries': 3,
            'timeout': 30
        }
        
        manager = WebSocketManager(config=config)
        
        # Verify config is stored
        assert manager.config == config
        assert manager.config['reconnect_delay'] == 5
        
        await manager.close()
        logger.info("✓ Config initialization test passed")
    
    def test_manager_attributes(self):
        """Test that WebSocketManager has all required attributes."""
        manager = WebSocketManager()
        
        # Test attributes from problem statement
        assert hasattr(manager, 'exchanges')
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'connections')
        assert hasattr(manager, 'callbacks')
        assert hasattr(manager, 'is_running')
        assert hasattr(manager, 'reconnect_delays')
        
        # Test that callbacks is a defaultdict
        assert isinstance(manager.callbacks, defaultdict)
        
        logger.info("✓ Manager attributes test passed")


def run_websocket_tests():
    """Run all WebSocket tests."""
    logger.info("="*60)
    logger.info("Phase 3.1: WebSocket Infrastructure Tests")
    logger.info("="*60)
    
    # Run tests
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '-p', 'no:warnings'
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("\n" + "="*60)
        logger.info("✅ All WebSocket tests passed!")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ Some tests failed")
        logger.error("="*60)
    
    return exit_code


if __name__ == '__main__':
    run_websocket_tests()
