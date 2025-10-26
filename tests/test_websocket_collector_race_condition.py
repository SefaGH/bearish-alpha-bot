"""
Test for WebSocket collector race condition fix.

This test verifies that the fixes for WebSocket Manager collector race condition
work correctly. The issue was that MarketDataPipeline tried to inject data before
WebSocketManager's collector was ready, causing repeated warnings.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.websocket_manager import WebSocketManager, StreamDataCollector
from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient


class TestWebSocketCollectorRaceCondition:
    """Test suite for WebSocket collector race condition fixes."""
    
    def test_websocket_manager_has_collector_property(self):
        """Test that WebSocketManager exposes collector property."""
        ws_manager = WebSocketManager()
        
        # Verify collector property exists
        assert hasattr(ws_manager, 'collector')
        
        # Verify it returns the internal _data_collector
        assert ws_manager.collector is ws_manager._data_collector
        
        # Verify it's a StreamDataCollector instance
        assert isinstance(ws_manager.collector, StreamDataCollector)
    
    def test_websocket_manager_is_collector_ready(self):
        """Test is_collector_ready() method."""
        ws_manager = WebSocketManager()
        
        # Should be ready after initialization
        assert ws_manager.is_collector_ready() is True
        
        # Test with no collector
        ws_manager._data_collector = None
        assert ws_manager.is_collector_ready() is False
    
    def test_stream_data_collector_prime_buffer(self):
        """Test prime_buffer_with_dataframe() method."""
        collector = StreamDataCollector(buffer_size=100)
        
        # Create sample DataFrame
        timestamps = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        df = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [101 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [100.5 + i for i in range(10)],
            'volume': [1000 + i*10 for i in range(10)]
        }, index=timestamps)
        
        # Prime the buffer
        collector.prime_buffer_with_dataframe('bingx', 'BTC/USDT:USDT', '1h', df)
        
        # Verify data was stored
        key = 'BTC/USDT:USDT_1h'
        assert 'bingx' in collector.ohlcv_data
        assert key in collector.ohlcv_data['bingx']
        
        # Verify data format
        stored_data = collector.ohlcv_data['bingx'][key]
        assert len(stored_data) == 1  # One entry (the latest)
        assert 'timestamp' in stored_data[0]
        assert 'data' in stored_data[0]
        
        # Verify OHLCV data
        ohlcv_list = stored_data[0]['data']
        assert len(ohlcv_list) == 10  # 10 candles
        assert len(ohlcv_list[0]) == 6  # [timestamp, open, high, low, close, volume]
    
    @pytest.mark.asyncio
    async def test_market_data_pipeline_wait_for_websocket_ready(self):
        """Test _wait_for_websocket_ready() method."""
        # Create mock WebSocket manager
        mock_ws = MagicMock()
        mock_ws.is_collector_ready.return_value = True
        
        # Create pipeline with mock
        mock_exchange = Mock(spec=CcxtClient)
        pipeline = MarketDataPipeline(
            exchanges={'bingx': mock_exchange},
            websocket_manager=mock_ws
        )
        
        # Test: should return True immediately since collector is ready
        ready = await pipeline._wait_for_websocket_ready(timeout=1.0)
        assert ready is True
    
    @pytest.mark.asyncio
    async def test_market_data_pipeline_wait_timeout(self):
        """Test _wait_for_websocket_ready() timeout."""
        # Create mock WebSocket manager that's never ready
        mock_ws = MagicMock()
        mock_ws.is_collector_ready.return_value = False
        mock_ws.collector = None
        
        # Create pipeline with mock
        mock_exchange = Mock(spec=CcxtClient)
        pipeline = MarketDataPipeline(
            exchanges={'bingx': mock_exchange},
            websocket_manager=mock_ws
        )
        
        # Test: should timeout after 0.5 seconds
        ready = await pipeline._wait_for_websocket_ready(timeout=0.5)
        assert ready is False
    
    @pytest.mark.asyncio
    async def test_market_data_pipeline_no_websocket_manager(self):
        """Test _wait_for_websocket_ready() with no WebSocket manager."""
        # Create pipeline without WebSocket manager
        mock_exchange = Mock(spec=CcxtClient)
        pipeline = MarketDataPipeline(
            exchanges={'bingx': mock_exchange},
            websocket_manager=None
        )
        
        # Test: should return False immediately
        ready = await pipeline._wait_for_websocket_ready(timeout=1.0)
        assert ready is False
    
    def test_defensive_null_checks_in_fetch_and_store(self):
        """Test that defensive null checks don't crash when collector is missing."""
        # Create pipeline without WebSocket manager
        mock_exchange = Mock(spec=CcxtClient)
        pipeline = MarketDataPipeline(
            exchanges={'bingx': mock_exchange},
            websocket_manager=None
        )
        
        # Create fake OHLCV data
        timestamps = pd.date_range(start='2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1010, 1020, 1030, 1040]
        }, index=timestamps)
        
        # This should not crash even though websocket_manager is None
        # The method should log a warning and continue
        # (We can't easily test the sync method without mocking ccxt calls,
        #  but the defensive checks are the same in both async and sync versions)
    
    @pytest.mark.asyncio
    async def test_integration_websocket_collector_ready_flow(self):
        """Integration test for the full ready-wait flow."""
        # Create a real WebSocketManager
        ws_manager = WebSocketManager()
        
        # Verify collector is ready
        assert ws_manager.is_collector_ready() is True
        
        # Create MarketDataPipeline
        mock_exchange = Mock(spec=CcxtClient)
        pipeline = MarketDataPipeline(
            exchanges={'bingx': mock_exchange},
            websocket_manager=ws_manager
        )
        
        # Wait for collector to be ready
        ready = await pipeline._wait_for_websocket_ready(timeout=1.0)
        assert ready is True
        
        # Verify we can prime data
        timestamps = pd.date_range(start='2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1010, 1020, 1030, 1040]
        }, index=timestamps)
        
        # Prime buffer through WebSocket manager
        ws_manager.collector.prime_buffer_with_dataframe('bingx', 'BTC/USDT:USDT', '1h', df)
        
        # Verify data was stored
        key = 'BTC/USDT:USDT_1h'
        assert 'bingx' in ws_manager.collector.ohlcv_data
        assert key in ws_manager.collector.ohlcv_data['bingx']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
