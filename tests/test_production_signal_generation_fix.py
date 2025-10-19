"""
Test for Critical Signal Generation Fix in Production Loop.

Validates:
1. No double sleep in _process_trading_loop()
2. Strategy execution logs are at INFO level (not DEBUG)
3. Symbol processing logs are visible
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, name):
        self.name = name
        self.call_count = 0
    
    def signal(self, df_30m, df_1h, regime_data=None):
        """Generate a mock signal."""
        self.call_count += 1
        return {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': self.name
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_trading_loop_no_double_sleep():
    """Test that _process_trading_loop() does not sleep internally."""
    coordinator = ProductionCoordinator()
    
    # Set active symbols
    coordinator.active_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    
    # Mock process_symbol to avoid complex dependencies
    async def mock_process_symbol(symbol):
        return None  # No signal
    
    coordinator.process_symbol = mock_process_symbol
    
    # Time the execution - should NOT include sleep
    import time
    start = time.time()
    
    await coordinator._process_trading_loop()
    
    elapsed = time.time() - start
    
    # Should complete quickly (< 5 seconds) without the 30s sleep
    assert elapsed < 5.0, f"Loop took {elapsed}s - should not sleep internally"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strategy_execution_logs_at_info_level(caplog):
    """Test that strategy execution logs are at INFO level, not DEBUG."""
    coordinator = ProductionCoordinator()
    
    # Set log level to INFO to check if logs appear
    caplog.set_level(logging.INFO)
    
    # Mock exchange clients and initialize
    mock_client = Mock()
    mock_client.fetch_ticker = Mock(return_value={'last': 50000.0})
    
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Register a strategy
    strategy = MockStrategy('test_strategy')
    coordinator.register_strategy('test_strategy', strategy, 0.25)
    
    # Mock data fetching
    mock_df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50100] * 100,
        'low': [49900] * 100,
        'close': [50000] * 100,
        'volume': [1000] * 100
    })
    mock_df['rsi'] = 50.0
    mock_df['ema_50'] = 50000.0
    mock_df['ema_200'] = 49500.0
    
    with patch.object(coordinator, '_fetch_ohlcv', return_value=mock_df):
        # Process a symbol
        await coordinator.process_symbol('BTC/USDT:USDT')
    
    # Check that INFO level logs are present
    log_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    
    # Should see strategy count log
    assert any('Registered strategies count: 1' in msg for msg in log_messages), \
        "Strategy count log not found at INFO level"
    
    # Should see strategy execution log
    assert any('Executing 1 strategies for BTC/USDT:USDT' in msg for msg in log_messages), \
        "Strategy execution log not found at INFO level"
    
    # Should see calling strategy log
    assert any('Calling test_strategy' in msg for msg in log_messages), \
        "Strategy call log not found at INFO level"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_symbol_processing_log_visible(caplog):
    """Test that symbol processing log is visible at INFO level."""
    coordinator = ProductionCoordinator()
    
    # Set log level to INFO
    caplog.set_level(logging.INFO)
    
    # Set active symbols
    coordinator.active_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    # Mock process_symbol to avoid complex dependencies
    async def mock_process_symbol(symbol):
        return None  # No signal
    
    coordinator.process_symbol = mock_process_symbol
    
    # Run the trading loop
    await coordinator._process_trading_loop()
    
    # Check that INFO level logs contain symbol processing info
    log_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    
    # Should see the processing log with count and list
    assert any('Processing 3 symbols' in msg for msg in log_messages), \
        "Symbol processing count log not found"
    
    assert any("['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']" in msg for msg in log_messages), \
        "Symbol list not found in log"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_sleep_allows_30s_interval_in_main_loop():
    """Test that removing sleep from _process_trading_loop allows proper 30s interval."""
    coordinator = ProductionCoordinator()
    
    # Set active symbols
    coordinator.active_symbols = ['BTC/USDT:USDT']
    
    # Mock process_symbol
    async def mock_process_symbol(symbol):
        return None
    
    coordinator.process_symbol = mock_process_symbol
    
    # Call _process_trading_loop multiple times in sequence
    import time
    start = time.time()
    
    # Simulate 3 iterations like the main loop would do
    for _ in range(3):
        await coordinator._process_trading_loop()
        # Main loop in run_production_loop would sleep here
        # But we're testing that _process_trading_loop itself doesn't sleep
    
    elapsed = time.time() - start
    
    # All 3 iterations should complete quickly (< 5s total) without internal sleep
    assert elapsed < 5.0, f"3 iterations took {elapsed}s - should not include sleep"


@pytest.mark.unit
@pytest.mark.asyncio  
async def test_logs_visible_with_multiple_strategies(caplog):
    """Test that logs are visible when multiple strategies are registered."""
    coordinator = ProductionCoordinator()
    
    # Set log level to INFO
    caplog.set_level(logging.INFO)
    
    # Mock exchange clients and initialize
    mock_client = Mock()
    mock_client.fetch_ticker = Mock(return_value={'last': 50000.0})
    
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Register multiple strategies
    strategy1 = MockStrategy('strategy_1')
    strategy2 = MockStrategy('strategy_2')
    coordinator.register_strategy('strategy_1', strategy1, 0.25)
    coordinator.register_strategy('strategy_2', strategy2, 0.25)
    
    # Mock data fetching
    mock_df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50100] * 100,
        'low': [49900] * 100,
        'close': [50000] * 100,
        'volume': [1000] * 100
    })
    mock_df['rsi'] = 50.0
    mock_df['ema_50'] = 50000.0
    mock_df['ema_200'] = 49500.0
    
    with patch.object(coordinator, '_fetch_ohlcv', return_value=mock_df):
        await coordinator.process_symbol('BTC/USDT:USDT')
    
    # Check logs
    log_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    
    # Should see count of 2 strategies
    assert any('Registered strategies count: 2' in msg for msg in log_messages), \
        "Strategy count log not found"
    
    # Should see execution log mentioning 2 strategies
    assert any('Executing 2 strategies' in msg for msg in log_messages), \
        "Strategy execution log not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
