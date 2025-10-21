"""
Shared fixtures for integration tests.

These fixtures provide common setup and teardown for integration tests.
"""

import pytest
import os
import sys
import asyncio
from typing import Generator

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))


@pytest.fixture(scope="function")
def integration_env() -> Generator[None, None, None]:
    """
    Setup and teardown environment variables for integration tests.
    
    This fixture ensures tests run with consistent environment configuration
    and cleans up after each test.
    """
    # Store original env vars
    original_env = {}
    test_env_keys = [
        'TRADING_SYMBOLS',
        'TRADING_MODE',
        'CAPITAL_USDT',
        'RSI_THRESHOLD_BTC',
        'CONFIG_PATH',
        'SKIP_PYTHON_VERSION_CHECK'
    ]
    
    for key in test_env_keys:
        original_env[key] = os.environ.get(key)
    
    # Set default test environment
    os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT'
    os.environ['TRADING_MODE'] = 'paper'
    os.environ['CAPITAL_USDT'] = '100'
    os.environ['CONFIG_PATH'] = 'config/config.example.yaml'
    # Allow tests to run even if Python version doesn't match exactly
    os.environ['SKIP_PYTHON_VERSION_CHECK'] = '1'
    
    yield
    
    # Restore original env vars
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
async def cleanup_tasks():
    """
    Fixture to ensure all async tasks are cleaned up after test.
    
    This prevents task leaks between tests.
    """
    yield
    
    # Cancel any remaining tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        if not task.done():
            task.cancel()
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
