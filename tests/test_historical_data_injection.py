#!/usr/bin/env python3
"""
Integration test for historical data injection into WebSocket cache.

This test validates the complete flow described in the GitHub issue:
- Historical data fetched via MarketDataPipeline
- Data injected into WebSocketManager's StreamDataCollector
- Data retrievable from the central cache
- Indicators calculated correctly on retrieval
"""

import sys
import os
from unittest.mock import Mock
import pytest
import pandas as pd
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.websocket_manager import WebSocketManager, StreamDataCollector
from core.ccxt_client import CcxtClient


@pytest.fixture
def generate_ohlcv_data():
    """Generate test OHLCV data."""
    def _generate(count=250, interval_minutes=60, base_price=50000):
        data = []
        timestamp = int(time.time() * 1000)
        for i in range(count):
            data.append([
                timestamp - (i * interval_minutes * 60 * 1000),
                base_price + i * 10,      # open
                base_price + i * 10 + 100,  # high
                base_price + i * 10 - 100,  # low
                base_price + i * 10 + 50,   # close
                1000 + i                   # volume
            ])
        return list(reversed(data))
    return _generate


@pytest.fixture
def mock_ccxt_client(generate_ohlcv_data):
    """Create a mock CcxtClient for testing."""
    client = Mock(spec=CcxtClient)
    client.name = 'test_exchange'
    client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Generate 250 candles (enough for EMA200 calculation)
    sample_data = generate_ohlcv_data(count=250)
    
    client.ohlcv.return_value = sample_data
    client.fetch_ohlcv_bulk.return_value = sample_data
    
    return client


@pytest.mark.asyncio
async def test_end_to_end_historical_data_injection(mock_ccxt_client):
    """
    Test complete historical data injection flow.
    
    This validates the architecture described in the GitHub issue:
    1. MarketDataPipeline fetches historical data via REST
    2. Data is injected into WebSocketManager's StreamDataCollector
    3. Data can be retrieved from the central cache
    4. Indicators are calculated correctly
    """
    # Step 1: Create WebSocketManager with StreamDataCollector
    ws_manager = WebSocketManager()
    assert ws_manager.collector is not None, "StreamDataCollector should be initialized"
    assert ws_manager.is_collector_ready(), "Collector should be ready"
    
    # Step 2: Create MarketDataPipeline with WebSocketManager
    exchanges = {'test_exchange': mock_ccxt_client}
    config = {
        'indicators': {
            'rsi_period': 14,
            'ema_fast': 21,
            'ema_mid': 50,
            'ema_slow': 200
        }
    }
    pipeline = MarketDataPipeline(exchanges, config=config, websocket_manager=ws_manager)
    
    # Step 3: Prime data buffers (simulates startup)
    symbols = ['BTC/USDT:USDT']
    timeframes = ['1h', '4h']
    await pipeline.prime_data_buffers_async(symbols, timeframes)
    
    # Step 4: Verify data was injected into WebSocketManager
    for timeframe in timeframes:
        # Check that data exists in the collector
        ohlcv = ws_manager.collector.get_latest_ohlcv('test_exchange', 'BTC/USDT:USDT', timeframe)
        assert ohlcv is not None, f"Data should be in collector for {timeframe}"
        assert len(ohlcv) == 250, f"Should have 250 candles for {timeframe}"
        
        # Verify OHLCV format
        assert len(ohlcv[0]) == 6, "Each candle should have 6 elements (timestamp, o, h, l, c, v)"
    
    # Step 5: Retrieve data via pipeline's get_latest_ohlcv
    df_1h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '1h')
    assert df_1h is not None, "Should retrieve 1h data"
    assert not df_1h.empty, "DataFrame should not be empty"
    assert len(df_1h) == 250, "Should have 250 rows"
    
    # Step 6: Verify indicators were calculated
    assert 'rsi' in df_1h.columns, "RSI indicator should be present"
    assert 'ema_fast' in df_1h.columns, "EMA fast should be present"
    assert 'ema_mid' in df_1h.columns, "EMA mid should be present"
    assert 'ema_slow' in df_1h.columns, "EMA slow should be present"
    
    # Step 7: Verify indicators have valid values (not all NaN)
    # EMA 200 needs 200 candles, so the last rows should have values
    assert not pd.isna(df_1h['ema_slow'].iloc[-1]), "EMA200 should be calculated for last candle"
    assert not pd.isna(df_1h['rsi'].iloc[-1]), "RSI should be calculated for last candle"
    
    # Step 8: Verify data for second timeframe
    df_4h = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '4h')
    assert df_4h is not None, "Should retrieve 4h data"
    assert len(df_4h) == 250, "Should have 250 rows"
    assert 'rsi' in df_4h.columns, "Indicators should be present"
    
    print("✅ End-to-end historical data injection test passed!")
    print(f"   - Data successfully injected into WebSocketManager")
    print(f"   - {len(symbols)} symbols, {len(timeframes)} timeframes")
    print(f"   - 250 historical candles per timeframe")
    print(f"   - All indicators calculated correctly")


@pytest.mark.asyncio
async def test_data_not_stored_locally_in_pipeline(mock_ccxt_client):
    """
    Test that data is NOT stored in pipeline's local data_streams.
    
    This validates that the "data silo" problem is fixed.
    """
    ws_manager = WebSocketManager()
    exchanges = {'test_exchange': mock_ccxt_client}
    pipeline = MarketDataPipeline(exchanges, websocket_manager=ws_manager)
    
    # Prime data
    await pipeline.prime_data_buffers_async(['BTC/USDT:USDT'], ['1h'])
    
    # Verify data_streams is empty or deprecated (not used for storage)
    # The data should only be in WebSocketManager, not in pipeline's data_streams
    assert len(pipeline.data_streams) == 0 or not pipeline.data_streams, \
        "Pipeline should not store data locally - it should use WebSocketManager"
    
    # But data should be retrievable via WebSocketManager
    df = pipeline.get_latest_ohlcv('BTC/USDT:USDT', '1h')
    assert df is not None, "Data should be retrievable from WebSocketManager"
    assert len(df) > 0, "Should have data"
    
    print("✅ Data silo prevention test passed!")
    print("   - Data NOT stored in pipeline's local cache")
    print("   - Data ONLY in WebSocketManager (single source of truth)")


def test_websocket_manager_collector_initialization():
    """
    Test that WebSocketManager properly initializes StreamDataCollector.
    
    This validates the fix mentioned in the issue description.
    """
    ws_manager = WebSocketManager()
    
    # Verify collector is initialized in __init__
    assert hasattr(ws_manager, '_data_collector'), "Should have _data_collector attribute"
    assert ws_manager._data_collector is not None, "Collector should be initialized"
    
    # Verify public accessor works
    assert ws_manager.collector is not None, "Public collector property should work"
    assert isinstance(ws_manager.collector, StreamDataCollector), "Should be StreamDataCollector instance"
    
    # Verify is_collector_ready works
    assert ws_manager.is_collector_ready(), "Collector should be ready"
    
    print("✅ WebSocketManager collector initialization test passed!")
    print("   - StreamDataCollector initialized in __init__")
    print("   - Public accessor working correctly")


def test_stream_data_collector_prime_buffer():
    """
    Test StreamDataCollector.prime_buffer_with_dataframe method.
    
    This validates the core injection mechanism.
    """
    collector = StreamDataCollector(buffer_size=1000)
    
    # Create test DataFrame
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    }, index=pd.to_datetime([1234567890000, 1234567950000, 1234568010000], unit='ms'))
    
    # Prime the buffer
    collector.prime_buffer_with_dataframe('test_exchange', 'BTC/USDT:USDT', '1h', df)
    
    # Verify data was stored
    ohlcv = collector.get_latest_ohlcv('test_exchange', 'BTC/USDT:USDT', '1h')
    assert ohlcv is not None, "Data should be stored"
    assert len(ohlcv) == 3, "Should have 3 candles"
    assert ohlcv[0][0] == 1234567890000, "First timestamp should match"
    assert ohlcv[0][1] == 100.0, "First open should match"
    assert ohlcv[-1][4] == 104.0, "Last close should match"
    
    print("✅ prime_buffer_with_dataframe test passed!")
    print("   - DataFrame successfully converted to OHLCV list")
    print("   - Data stored in collector buffer")
    print("   - Data retrievable via get_latest_ohlcv")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
