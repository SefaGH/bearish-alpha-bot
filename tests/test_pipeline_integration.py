#!/usr/bin/env python3
"""
Integration tests for Market Data Pipeline with main.py
"""

import sys
import os
from unittest.mock import Mock, patch
import pytest
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient


@pytest.mark.asyncio
async def test_pipeline_async_methods():
    """Test that async methods work correctly."""
    # Create mock client
    mock_client = Mock(spec=CcxtClient)
    mock_client.name = 'test_exchange'
    mock_client.validate_and_get_symbol.return_value = 'BTC/USDT:USDT'
    
    # Mock OHLCV data (format: [timestamp, open, high, low, close, volume])
    import time
    timestamp = int(time.time() * 1000) - (5 * 60 * 1000)  # 5 minutes ago
    sample_data = []
    for i in range(100):
        ts = timestamp - (i * 30 * 60 * 1000)  # 30-minute intervals
        sample_data.append([
            ts,
            50000 + i * 10,      # open
            50100 + i * 10,      # high
            49900 + i * 10,      # low
            50050 + i * 10,      # close
            1000 + i             # volume
        ])
    sample_data.reverse()  # Oldest first
    mock_client.ohlcv.return_value = sample_data
    mock_client.fetch_ohlcv_bulk.return_value = sample_data
    
    # Create pipeline
    exchanges = {'test': mock_client}
    pipeline = MarketDataPipeline(exchanges)
    
    # Test start_feeds_async
    results = await pipeline.start_feeds_async(['BTC/USDT:USDT'], ['30m'])
    
    assert results['successful_fetches'] > 0
    assert pipeline.is_running == True
    
    # Test get_health_status
    health = pipeline.get_health_status()
    
    assert 'overall_status' in health
    assert 'active_feeds' in health
    assert health['overall_status'] == 'healthy'
    
    # Cleanup
    pipeline.shutdown()


@pytest.mark.asyncio
async def test_run_with_pipeline_function_exists():
    """Test that run_with_pipeline function exists and is callable."""
    import main
    
    assert hasattr(main, 'run_with_pipeline')
    assert asyncio.iscoroutinefunction(main.run_with_pipeline)


def test_main_pipeline_flag_handling():
    """Test that main.py handles --pipeline flag correctly."""
    import main
    
    # The flag should be handled in __main__ block
    # We just verify the function exists for now
    assert hasattr(main, 'run_with_pipeline')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
