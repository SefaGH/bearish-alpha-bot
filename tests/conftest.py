"""
Shared pytest fixtures for all test types (unit and integration tests).

This module provides common fixtures and utilities for testing the
Bearish Alpha Bot trading system.
"""

import pytest
import os
import sys
import asyncio
import inspect
from typing import Generator, Dict, Any
from pathlib import Path
from unittest.mock import MagicMock, Mock

# Add src and scripts to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))


# ---------------------------------------------------------------------------
# Lightweight asyncio support (fallback when pytest-asyncio is unavailable)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the custom asyncio marker used across the test suite."""
    config.addinivalue_line(
        "markers",
        "asyncio: execute the test inside an asyncio event loop (fallback)",
    )


def pytest_pyfunc_call(pyfuncitem):
    """Execute coroutine tests inside a dedicated event loop."""

    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_function(**pyfuncitem.funcargs))
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    return True


# ============================================================================
# Session-scoped fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def test_data_dir(repo_root) -> Path:
    """Return the test data directory path."""
    data_dir = repo_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ============================================================================
# Function-scoped fixtures (run for each test)
# ============================================================================

@pytest.fixture(scope="function")
def clean_env() -> Generator[None, None, None]:
    """
    Provide a clean environment for each test.
    
    Saves original environment variables, sets test defaults,
    and restores original values after test completion.
    
    Usage:
        def test_something(clean_env):
            os.environ['TRADING_MODE'] = 'paper'
            # ... test code ...
    """
    # Save original ENV
    original_env = os.environ.copy()
    
    # Set test defaults
    test_env = {
        'TRADING_MODE': 'paper',
        'CAPITAL_USDT': '100',
        'TRADING_SYMBOLS': 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT',
        'CONFIG_PATH': 'config/config.example.yaml',
        'SKIP_PYTHON_VERSION_CHECK': '1',  # Allow tests to run on any Python version
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Restore original ENV
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def cleanup_tasks():
    """Placeholder fixture kept for backwards compatibility."""
    yield


@pytest.fixture
def mock_exchange_api():
    """
    Mock external exchange API calls (not the ccxt library itself).
    
    Returns a configured mock exchange object that simulates realistic
    responses without making real API calls.
    
    Usage:
        def test_something(mock_exchange_api):
            # mock_exchange_api.fetch_ticker() returns mock data
            pass
    """
    mock_exchange = MagicMock()
    
    # Mock common exchange methods with realistic data
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT:USDT',
        'last': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'bid': 49950.0,
        'ask': 50050.0,
        'volume': 1000000.0,
    }
    
    mock_exchange.fetch_ohlcv.return_value = [
        [1609459200000, 29000, 29100, 28900, 29050, 100],  # [timestamp, O, H, L, C, V]
        [1609462800000, 29050, 29200, 29000, 29150, 120],
        [1609466400000, 29150, 29250, 29100, 29200, 110],
    ]
    
    mock_exchange.fetch_balance.return_value = {
        'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}
    }
    
    mock_exchange.get_bingx_balance.return_value = {
        'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}
    }
    
    # Mock connection status
    mock_exchange.is_connected = Mock(return_value=True)
    
    return mock_exchange


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide a sample configuration dictionary for testing.
    
    Returns a minimal valid configuration that can be used in tests.
    
    Usage:
        def test_something(sample_config):
            config = sample_config
            # ... use config in test ...
    """
    return {
        'universe': {
            'fixed_symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
            'auto_select': False,
        },
        'risk': {
            'max_position_size': 0.2,
            'max_portfolio_risk': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.015,
            'max_drawdown': 0.10,
        },
        'execution': {
            'mode': 'paper',
            'exchange': 'bingx',
        },
        'adaptive_strategies': {
            'ob': {
                'enabled': True,
                'base_threshold': 45,
                'threshold_range': 10,
            },
            'str': {
                'enabled': True,
                'base_threshold': 55,
                'threshold_range': 10,
            },
        },
        'signals': {
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.5,
        },
        'websocket': {
            'enabled': False,
        },
    }


# ============================================================================
# Pytest configuration hooks
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook - runs before test collection.
    
    Registers custom markers and configures test environment.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (slow, end-to-end)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (takes >5 seconds)"
    )
    config.addinivalue_line(
        "markers", "phase3: mark test as Phase 3 ML/AI feature test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test items during collection.
    
    Automatically marks integration tests and adds markers based on file location.
    """
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark phase3 tests
        if any(keyword in item.nodeid for keyword in ['ml_', 'regime_', 'price_pred', 'strategy_opt']):
            item.add_marker(pytest.mark.phase3)


# ============================================================================
# Utility functions for tests
# ============================================================================

def get_test_symbols() -> list:
    """Return list of test trading symbols."""
    return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']


def get_test_capital() -> float:
    """Return test capital amount in USDT."""
    return 100.0


# ============================================================================
# Asyncio event loop configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.get_event_loop_policy()
