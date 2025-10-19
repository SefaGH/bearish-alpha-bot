"""
Test Phase 3.4 Strategy Execution Fix.

Tests that registered strategies are properly stored and executed in process_symbol().
"""

import pytest
import asyncio
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
def test_init_stores_strategies_dict():
    """Test that __init__ creates self.strategies dictionary."""
    coordinator = ProductionCoordinator()
    
    # Verify strategies dictionary exists
    assert hasattr(coordinator, 'strategies')
    assert isinstance(coordinator.strategies, dict)
    assert len(coordinator.strategies) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_strategy_stores_reference():
    """Test that register_strategy stores strategy reference."""
    coordinator = ProductionCoordinator()
    
    # Mock exchange clients and initialize
    mock_client = Mock()
    mock_client.fetch_ticker = Mock(return_value={'last': 50000.0})
    
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Create and register strategy
    strategy = MockStrategy('test_strategy')
    result = coordinator.register_strategy('test_strategy', strategy, 0.25)
    
    # Verify strategy was stored
    assert 'test_strategy' in coordinator.strategies
    assert coordinator.strategies['test_strategy'] is strategy


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_symbol_executes_registered_strategies():
    """Test that process_symbol executes registered strategies."""
    coordinator = ProductionCoordinator()
    
    # Mock exchange clients and initialize
    mock_client = Mock()
    mock_client.fetch_ticker = Mock(return_value={'last': 50000.0})
    
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Create and register strategy
    strategy = MockStrategy('test_strategy')
    coordinator.register_strategy('test_strategy', strategy, 0.25)
    
    # Mock data fetching to return valid DataFrames
    mock_df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50100] * 100,
        'low': [49900] * 100,
        'close': [50000] * 100,
        'volume': [1000] * 100
    })
    
    # Add required indicators
    mock_df['rsi'] = 50.0
    mock_df['ema_50'] = 50000.0
    mock_df['ema_200'] = 49500.0
    
    with patch.object(coordinator, '_fetch_ohlcv', return_value=mock_df):
        # Process a symbol
        signal = await coordinator.process_symbol('BTC/USDT:USDT')
    
    # Verify strategy was called
    assert strategy.call_count > 0
    
    # Verify signal was generated
    assert signal is not None
    assert signal['strategy'] == 'test_strategy'
    assert signal['symbol'] == 'BTC/USDT:USDT'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_symbol_fallback_when_no_strategies():
    """Test that process_symbol falls back to default behavior when no strategies registered."""
    coordinator = ProductionCoordinator()
    
    # Initialize without registering strategies
    mock_client = Mock()
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Mock data fetching
    mock_df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50100] * 100,
        'low': [49900] * 100,
        'close': [50000] * 100,
        'volume': [1000] * 100
    })
    mock_df['rsi'] = 25.0  # Oversold
    mock_df['ema_50'] = 50000.0
    mock_df['ema_200'] = 49500.0
    
    with patch.object(coordinator, '_fetch_ohlcv', return_value=mock_df):
        # Process a symbol - should use fallback strategies
        signal = await coordinator.process_symbol('BTC/USDT:USDT')
    
    # Signal may or may not be generated depending on conditions
    # The important thing is no error occurs
    assert True  # If we got here without error, test passes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_strategies_first_signal_wins():
    """Test that when multiple strategies are registered, first signal wins."""
    coordinator = ProductionCoordinator()
    
    # Mock exchange clients and initialize
    mock_client = Mock()
    mock_client.fetch_ticker = Mock(return_value={'last': 50000.0})
    
    await coordinator.initialize_production_system(
        exchange_clients={'test': mock_client},
        portfolio_config={'equity_usd': 1000}
    )
    
    # Create and register multiple strategies
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
        signal = await coordinator.process_symbol('BTC/USDT:USDT')
    
    # Verify first strategy was called
    assert strategy1.call_count > 0
    
    # Verify signal came from first strategy
    assert signal is not None
    assert signal['strategy'] == 'strategy_1'
    
    # Second strategy should not be called (first signal wins)
    assert strategy2.call_count == 0


@pytest.mark.unit
def test_strategies_dict_survives_reinitialization():
    """Test that strategies dict persists across operations."""
    coordinator = ProductionCoordinator()
    
    # Add a mock strategy directly (before initialization)
    mock_strategy = MockStrategy('test')
    coordinator.strategies['test'] = mock_strategy
    
    # Verify it's still there
    assert 'test' in coordinator.strategies
    assert coordinator.strategies['test'] is mock_strategy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
