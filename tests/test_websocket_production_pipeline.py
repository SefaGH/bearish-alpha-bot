"""
Test Phase-4 WebSocket → Production Data Pipeline
Tests the complete flow from WebSocket to ProductionCoordinator.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from src.core.websocket_client import WebSocketClient
from src.core.websocket_manager import WebSocketManager, StreamDataCollector
from src.core.bingx_websocket import BingXWebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestWebSocketLoopMethods:
    """Test WebSocket loop methods for continuous streaming."""
    
    @pytest.mark.asyncio
    async def test_watch_ohlcv_loop_exists(self):
        """Test that watch_ohlcv_loop method exists."""
        client = WebSocketClient('bingx')
        assert hasattr(client, 'watch_ohlcv_loop')
        logger.info("✓ watch_ohlcv_loop method exists")
    
    @pytest.mark.asyncio
    async def test_watch_ticker_loop_exists(self):
        """Test that watch_ticker_loop method exists."""
        client = WebSocketClient('bingx')
        assert hasattr(client, 'watch_ticker_loop')
        logger.info("✓ watch_ticker_loop method exists")
    
    @pytest.mark.asyncio
    async def test_watch_ohlcv_loop_with_iterations(self):
        """Test watch_ohlcv_loop with limited iterations."""
        client = WebSocketClient('bingx')
        
        # Track callback invocations
        callback_count = 0
        
        async def test_callback(symbol, timeframe, ohlcv):
            nonlocal callback_count
            callback_count += 1
            logger.info(f"Callback invoked: {symbol} {timeframe} (count: {callback_count})")
        
        # Mock the watch_ohlcv to return test data
        async def mock_watch_ohlcv(symbol, timeframe, callback=None):
            return [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
        
        client.watch_ohlcv = mock_watch_ohlcv
        
        # Run loop with max 3 iterations
        await client.watch_ohlcv_loop('BTC/USDT:USDT', '1m', test_callback, max_iterations=3)
        
        # Should have been called 3 times
        assert callback_count == 3
        logger.info(f"✓ watch_ohlcv_loop completed {callback_count} iterations")


class TestStreamDataCollector:
    """Test StreamDataCollector integration."""
    
    def test_collector_initialization(self):
        """Test StreamDataCollector can be initialized."""
        collector = StreamDataCollector(buffer_size=100)
        assert collector.buffer_size == 100
        assert hasattr(collector, 'ohlcv_data')
        assert hasattr(collector, 'ticker_data')
        logger.info("✓ StreamDataCollector initialized")
    
    @pytest.mark.asyncio
    async def test_collector_ohlcv_callback(self):
        """Test collector OHLCV callback stores data."""
        collector = StreamDataCollector(buffer_size=100)
        
        # Test data
        test_ohlcv = [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
        
        # Call callback
        await collector.ohlcv_callback('bingx', 'BTC/USDT:USDT', '1m', test_ohlcv)
        
        # Verify data stored
        assert 'bingx' in collector.ohlcv_data
        assert 'BTC/USDT:USDT_1m' in collector.ohlcv_data['bingx']
        assert len(collector.ohlcv_data['bingx']['BTC/USDT:USDT_1m']) == 1
        
        logger.info("✓ Collector stores OHLCV data correctly")
    
    @pytest.mark.asyncio
    async def test_collector_get_latest_ohlcv(self):
        """Test getting latest OHLCV from collector."""
        collector = StreamDataCollector(buffer_size=100)
        
        # Store test data
        test_ohlcv = [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
        await collector.ohlcv_callback('bingx', 'BTC/USDT:USDT', '1m', test_ohlcv)
        
        # Get latest
        latest = collector.get_latest_ohlcv('bingx', 'BTC/USDT:USDT', '1m')
        
        assert latest is not None
        assert latest == test_ohlcv
        
        logger.info("✓ get_latest_ohlcv returns correct data")


class TestBingXCollectorBridge:
    """Test BingX → StreamDataCollector bridge."""
    
    def test_bingx_accepts_collector(self):
        """Test BingXWebSocket accepts collector parameter."""
        collector = StreamDataCollector(buffer_size=100)
        
        ws = BingXWebSocket(
            api_key=None,
            api_secret=None,
            futures=True,
            collector=collector
        )
        
        assert ws.collector is collector
        logger.info("✓ BingXWebSocket accepts collector parameter")
    
    @pytest.mark.asyncio
    async def test_bingx_bridges_data_to_collector(self):
        """Test BingX bridges kline data to collector."""
        collector = StreamDataCollector(buffer_size=100)
        
        ws = BingXWebSocket(
            api_key=None,
            api_secret=None,
            futures=True,
            collector=collector
        )
        
        # Mock kline message (BingX format)
        mock_message = {
            "code": 0,
            "dataType": "BTC-USDT@kline_1m",
            "s": "BTC-USDT",
            "data": [
                {
                    "c": "110267.6",
                    "o": "110298.6",
                    "h": "110298.6",
                    "l": "110265.1",
                    "v": "2.0741",
                    "T": 1761327420000
                }
            ]
        }
        
        # Process message
        await ws._process_message(mock_message)
        
        # Verify data was bridged to collector
        latest = collector.get_latest_ohlcv('bingx', 'BTC/USDT:USDT', '1m')
        
        assert latest is not None
        assert len(latest) > 0
        # Check first candle
        candle = latest[0]
        assert candle[0] == 1761327420000  # timestamp
        assert candle[1] == 110298.6  # open
        assert candle[4] == 110267.6  # close
        
        logger.info("✓ BingX bridges data to collector correctly")


class TestWebSocketManagerCollectorInjection:
    """Test WebSocketManager injects collector into clients."""
    
    def test_manager_has_collector(self):
        """Test WebSocketManager initializes with collector."""
        manager = WebSocketManager()
        
        assert hasattr(manager, '_data_collector')
        assert manager._data_collector is not None
        
        logger.info("✓ WebSocketManager has collector")
    
    def test_manager_passes_collector_to_clients(self):
        """Test WebSocketManager passes collector to clients."""
        manager = WebSocketManager()
        
        # Check if BingX client has collector
        if 'bingx' in manager.clients:
            client = manager.clients['bingx']
            assert hasattr(client, 'collector')
            assert client.collector is manager._data_collector
            
            logger.info("✓ WebSocketManager passes collector to BingX client")
        else:
            logger.warning("BingX client not initialized, skipping collector test")


class TestWebSocketManagerGetLatestData:
    """Test WebSocketManager.get_latest_data() API."""
    
    @pytest.mark.asyncio
    async def test_get_latest_data_exists(self):
        """Test get_latest_data method exists."""
        manager = WebSocketManager()
        
        assert hasattr(manager, 'get_latest_data')
        logger.info("✓ get_latest_data method exists")
    
    @pytest.mark.asyncio
    async def test_get_latest_data_returns_none_when_empty(self):
        """Test get_latest_data returns None when no data available."""
        manager = WebSocketManager()
        
        result = manager.get_latest_data('BTC/USDT:USDT', '1m')
        
        # Should return None when no data
        assert result is None
        logger.info("✓ get_latest_data returns None when empty")
    
    @pytest.mark.asyncio
    async def test_get_latest_data_returns_data_after_collection(self):
        """Test get_latest_data returns data after collector receives it."""
        manager = WebSocketManager()
        
        # Manually add data to collector
        test_ohlcv = [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
        await manager._data_collector.ohlcv_callback('bingx', 'BTC/USDT:USDT', '1m', test_ohlcv)
        
        # Mark stream as active
        manager._active_streams['bingx'].add('BTC/USDT:USDT_1m')
        
        # Get data
        result = manager.get_latest_data('BTC/USDT:USDT', '1m')
        
        assert result is not None
        assert 'exchange' in result
        assert 'symbol' in result
        assert 'timeframe' in result
        assert 'ohlcv' in result
        assert result['symbol'] == 'BTC/USDT:USDT'
        assert result['timeframe'] == '1m'
        assert result['ohlcv'] == test_ohlcv
        
        logger.info("✓ get_latest_data returns collected data")


class TestTimeframeConfiguration:
    """Test timeframe configuration."""
    
    def test_config_has_required_timeframes(self):
        """Test config file has all required timeframes."""
        import yaml
        
        with open('config/config.example.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        stream_timeframes = config.get('websocket', {}).get('stream_timeframes', [])
        
        # Check all required timeframes are present
        required_timeframes = ['1m', '5m', '30m', '1h', '4h']
        for tf in required_timeframes:
            assert tf in stream_timeframes, f"Timeframe {tf} missing from config"
        
        logger.info(f"✓ Config has all required timeframes: {stream_timeframes}")


class TestEndToEndDataFlow:
    """Test end-to-end data flow from WebSocket to get_latest_data."""
    
    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self):
        """Test complete data pipeline from collection to retrieval."""
        manager = WebSocketManager()
        
        # Simulate data collection for multiple timeframes
        timeframes = ['1m', '5m', '30m', '1h', '4h']
        symbol = 'BTC/USDT:USDT'
        
        for tf in timeframes:
            # Create test data
            test_ohlcv = [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
            
            # Simulate data collection
            await manager._data_collector.ohlcv_callback('bingx', symbol, tf, test_ohlcv)
            
            # Mark stream as active
            manager._active_streams['bingx'].add(f'{symbol}_{tf}')
        
        # Test retrieval for each timeframe
        for tf in timeframes:
            result = manager.get_latest_data(symbol, tf)
            
            assert result is not None, f"No data for {tf}"
            assert result['timeframe'] == tf
            assert result['symbol'] == symbol
            assert len(result['ohlcv']) > 0
            
            logger.info(f"✓ Data pipeline working for {tf}")
        
        logger.info("✓ Complete data pipeline test passed")
    
    @pytest.mark.asyncio
    async def test_production_coordinator_can_access_data(self):
        """Test ProductionCoordinator-like access pattern."""
        manager = WebSocketManager()
        
        # Simulate WebSocket data collection
        symbol = 'BTC/USDT:USDT'
        timeframes = ['30m', '1h', '4h']
        
        for tf in timeframes:
            test_ohlcv = [[1234567890000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
            await manager._data_collector.ohlcv_callback('bingx', symbol, tf, test_ohlcv)
            manager._active_streams['bingx'].add(f'{symbol}_{tf}')
        
        # Simulate ProductionCoordinator access pattern
        data_30m = manager.get_latest_data(symbol, '30m')
        data_1h = manager.get_latest_data(symbol, '1h')
        data_4h = manager.get_latest_data(symbol, '4h')
        
        # All three should have data
        assert data_30m is not None, "Missing 30m data"
        assert data_1h is not None, "Missing 1h data"
        assert data_4h is not None, "Missing 4h data"
        
        # Verify structure
        for data in [data_30m, data_1h, data_4h]:
            assert 'exchange' in data
            assert 'symbol' in data
            assert 'timeframe' in data
            assert 'ohlcv' in data
            assert 'timestamp' in data
        
        logger.info("✓ ProductionCoordinator can access all required timeframes")


def run_all_tests():
    """Run all tests in this module."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
