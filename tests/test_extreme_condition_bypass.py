"""
Test Extreme Condition Bypass functionality.

Tests that RSI-based bypass logic correctly prevents RL Agent veto
when RSI reaches extreme oversold/overbought levels.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.strategy_coordinator import StrategyCoordinator


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self, config=None):
        self.cfg = config or {}
        self.exchange_clients = {}
    
    def get_strategy_allocation(self, strategy_name):
        return 0.25
    
    @property
    def performance_monitor(self):
        return None


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def __init__(self):
        self.active_positions = {}
    
    async def validate_new_position(self, signal, portfolio_manager=None):
        return True, "OK", {}


class MockMarketDataPipeline:
    """Mock market data pipeline for testing."""
    
    def __init__(self, rsi_value=50.0):
        self.rsi_value = rsi_value
    
    async def get_latest_ohlcv(self, symbol, timeframe, exchange=None):
        """Return mock OHLCV data with specified RSI."""
        return pd.DataFrame({
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100,
            'rsi': [self.rsi_value] * 100,  # All rows have the same RSI for simplicity
            'atr': [200] * 100
        })


class MockRLAgent:
    """Mock RL Agent that always returns HOLD."""
    
    def __init__(self):
        self.decision = 1  # 0=buy, 1=hold, 2=sell
    
    def act(self, state_features, market_regime=None):
        """Always return HOLD to test bypass."""
        return self.decision


@pytest.mark.unit
def test_bypass_config_structure():
    """Test that bypass config is properly structured."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        config=config
    )
    
    # Verify config is accessible
    bypass_config = coordinator.portfolio_manager.cfg.get('signals', {}).get('bypass', {})
    assert bypass_config.get('enabled') is True
    assert bypass_config.get('rsi_oversold_threshold') == 20
    assert bypass_config.get('rsi_overbought_threshold') == 80


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_rsi_from_market_data_extreme_oversold():
    """Test RSI extraction when RSI is at extreme oversold levels."""
    config = {'signals': {'bypass': {'enabled': True}}}
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    
    # Create coordinator with RSI = 15 (extreme oversold)
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Extract RSI
    rsi = await coordinator._extract_rsi_from_market_data('BTC/USDT:USDT')
    
    # Verify RSI is extracted correctly
    assert rsi is not None
    assert rsi == 15.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_rsi_from_market_data_extreme_overbought():
    """Test RSI extraction when RSI is at extreme overbought levels."""
    config = {'signals': {'bypass': {'enabled': True}}}
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    
    # Create coordinator with RSI = 85 (extreme overbought)
    market_pipeline = MockMarketDataPipeline(rsi_value=85.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Extract RSI
    rsi = await coordinator._extract_rsi_from_market_data('BTC/USDT:USDT')
    
    # Verify RSI is extracted correctly
    assert rsi is not None
    assert rsi == 85.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extreme_oversold_bypass_triggers_on_buy_signal():
    """Test that extreme oversold condition triggers bypass for BUY signal."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)  # Extreme oversold
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create BUY signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 15.0, 'BTC/USDT:USDT', 'buy'
    )
    
    # Verify bypass is triggered
    assert bypass_triggered is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extreme_overbought_bypass_triggers_on_sell_signal():
    """Test that extreme overbought condition triggers bypass for SELL signal."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=85.0)  # Extreme overbought
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create SELL signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0,
        'stop': 51000.0,
        'target': 48000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 85.0, 'BTC/USDT:USDT', 'sell'
    )
    
    # Verify bypass is triggered
    assert bypass_triggered is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_bypass_on_normal_rsi_buy_signal():
    """Test that bypass does NOT trigger for normal RSI with BUY signal."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=45.0)  # Normal RSI
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create BUY signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 45.0, 'BTC/USDT:USDT', 'buy'
    )
    
    # Verify bypass is NOT triggered
    assert bypass_triggered is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_bypass_on_mismatched_signal_type():
    """Test that bypass does NOT trigger when signal type doesn't match RSI condition."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)  # Extreme oversold
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create SELL signal (opposite of what RSI suggests)
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0,
        'stop': 51000.0,
        'target': 48000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass - should NOT trigger because SELL signal with oversold RSI
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 15.0, 'BTC/USDT:USDT', 'sell'
    )
    
    # Verify bypass is NOT triggered
    assert bypass_triggered is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bypass_disabled_in_config():
    """Test that bypass does NOT trigger when disabled in config."""
    config = {
        'signals': {
            'bypass': {
                'enabled': False,  # Disabled
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)  # Extreme oversold
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create BUY signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass - should NOT trigger because bypass is disabled
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 15.0, 'BTC/USDT:USDT', 'buy'
    )
    
    # Verify bypass is NOT triggered
    assert bypass_triggered is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_custom_thresholds():
    """Test that custom RSI thresholds are respected."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 25,  # Custom threshold
                'rsi_overbought_threshold': 75  # Custom threshold
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=24.0)  # Just below custom threshold
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create BUY signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass with RSI=24 (below custom threshold of 25)
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 24.0, 'BTC/USDT:USDT', 'buy'
    )
    
    # Verify bypass is triggered with custom threshold
    assert bypass_triggered is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_long_signal_synonym():
    """Test that 'long' signal type works same as 'buy'."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create LONG signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',  # Synonym for 'buy'
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 15.0, 'BTC/USDT:USDT', 'long'
    )
    
    # Verify bypass is triggered
    assert bypass_triggered is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_signal_synonym():
    """Test that 'short' signal type works same as 'sell'."""
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 80
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=85.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    # Create SHORT signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'short',  # Synonym for 'sell'
        'entry': 50000.0,
        'stop': 51000.0,
        'target': 48000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 85.0, 'BTC/USDT:USDT', 'short'
    )
    
    # Verify bypass is triggered
    assert bypass_triggered is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_threshold_validation():
    """Test that invalid thresholds are properly validated and bypass is not triggered."""
    # Test case 1: oversold >= overbought (invalid)
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 80,  # Invalid: higher than overbought
                'rsi_overbought_threshold': 20
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=15.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'buy',
        'entry': 50000.0,
        'stop': 49000.0,
        'target': 52000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass - should NOT trigger due to invalid thresholds
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 15.0, 'BTC/USDT:USDT', 'buy'
    )
    
    assert bypass_triggered is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_out_of_range_threshold_validation():
    """Test that out-of-range thresholds (< 0 or > 100) are rejected."""
    # Test case: threshold > 100 (invalid)
    config = {
        'signals': {
            'bypass': {
                'enabled': True,
                'rsi_oversold_threshold': 20,
                'rsi_overbought_threshold': 150  # Invalid: > 100
            }
        }
    }
    
    portfolio_mgr = MockPortfolioManager(config)
    risk_mgr = MockRiskManager()
    market_pipeline = MockMarketDataPipeline(rsi_value=85.0)
    
    coordinator = StrategyCoordinator(
        portfolio_manager=portfolio_mgr,
        risk_manager=risk_mgr,
        market_data_pipeline=market_pipeline,
        config=config
    )
    
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'sell',
        'entry': 50000.0,
        'stop': 51000.0,
        'target': 48000.0,
        'strategy_name': 'test_strategy'
    }
    
    # Check bypass - should NOT trigger due to out-of-range threshold
    bypass_triggered = await coordinator._check_extreme_condition_bypass(
        signal, 85.0, 'BTC/USDT:USDT', 'sell'
    )
    
    assert bypass_triggered is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
