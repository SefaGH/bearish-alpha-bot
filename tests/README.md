# Test Suite Documentation

## Overview

This directory contains the comprehensive test suite for the Bearish Alpha Bot trading system. The tests are designed to be reliable, maintainable, and verify actual system behavior.

## Test Categories

### Unit Tests (`tests/test_*.py`)
- **Speed:** Fast (~1 second per test)
- **Isolation:** Test individual functions/classes in isolation
- **Mocking:** Heavy mocking allowed
- **Purpose:** Verify individual component behavior

**Example unit tests:**
- `test_utils.py` - Utility function tests
- `test_import_compatibility.py` - Import mechanism tests
- `test_risk_management.py` - Risk calculation tests

### Integration Tests (`tests/integration/`)
- **Speed:** Slow (~60 seconds total for all integration tests)
- **Isolation:** Test full system behavior end-to-end
- **Mocking:** Minimal mocking (only external APIs like exchange connections)
- **Purpose:** Verify system integration and detect real issues (freeze, deadlock, data flow)

**Example integration tests:**
- `test_launcher_integration.py` - Full launcher initialization and execution
- `test_websocket_streaming.py` - WebSocket connection and data delivery
- `test_config_consistency.py` - Configuration loading and priority

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Unit Tests Only (Fast)
```bash
pytest tests/ -v -m "not integration"
```

### Run Integration Tests Only (Slow)
```bash
pytest tests/integration/ -v -m integration
```

### Run Specific Test File
```bash
pytest tests/integration/test_launcher_integration.py -v
```

### Run Specific Test
```bash
pytest tests/integration/test_launcher_integration.py::test_launcher_runs_without_freeze -v -s
```

### Run with Output
```bash
pytest tests/ -v -s  # -s shows print statements
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "config" -v  # Run all tests with "config" in name
```

## Test Configuration

### pytest.ini
The test suite is configured in `pytest.ini` at the repository root:
- Async mode: auto (for async/await tests)
- Test paths: `tests/`
- Timeout: 300 seconds per test
- Markers: integration, unit, slow, phase3

### conftest.py Files

#### tests/conftest.py (Top-level)
Shared fixtures for all tests:
- `clean_env`: Clean environment variables
- `cleanup_tasks`: Async task cleanup
- `mock_exchange_api`: Mock exchange API responses
- `sample_config`: Sample configuration dictionary

#### tests/integration/conftest.py
Integration-specific fixtures:
- `integration_env`: Integration test environment setup
- Python version check bypass (SKIP_PYTHON_VERSION_CHECK)

## Writing New Tests

### DO:
- ✅ **Mock external APIs** (exchanges, external services, network calls)
- ✅ **Test actual behavior** (not assumed behavior)
- ✅ **Use real async execution** (don't mock asyncio primitives)
- ✅ **Verify with Python 3.11+** (project requirement)
- ✅ **Use existing fixtures** from conftest.py
- ✅ **Follow naming convention** (`test_*.py` for files, `test_*` for functions)

### DON'T:
- ❌ **Don't mock core libraries** (ccxt, aiohttp, asyncio) - mock the API calls instead
- ❌ **Don't mock during import time** - import after setting up patches
- ❌ **Don't over-mock** - integration tests should test real integration
- ❌ **Don't test assumptions** - test what the code actually does
- ❌ **Don't use deprecated test patterns** - follow current examples

## Test Patterns

### Unit Test Pattern
```python
import pytest
from my_module import my_function

def test_my_function():
    """Test my_function with valid input."""
    result = my_function(input_value)
    assert result == expected_value
```

### Async Unit Test Pattern
```python
import pytest

@pytest.mark.asyncio
async def test_async_function(cleanup_tasks):
    """Test async function."""
    result = await my_async_function()
    assert result is not None
```

### Integration Test Pattern (Correct)
```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.integration
@pytest.mark.asyncio
async def test_launcher_integration(integration_env, cleanup_tasks):
    """Test full launcher initialization and execution."""
    
    # Mock external dependencies BEFORE import
    with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
         patch('core.notify.Telegram') as mock_telegram:
        
        # Import AFTER patching
        from live_trading_launcher import LiveTradingLauncher
        
        # Setup mocks
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
        mock_ccxt.return_value = mock_exchange
        
        # Test full initialization
        launcher = LiveTradingLauncher(mode='paper')
        result = await launcher.run(duration=5)
        
        assert result == 0  # Success
```

### Integration Test Pattern (Wrong - Don't Do This)
```python
# ❌ WRONG: Import before patching
from live_trading_launcher import LiveTradingLauncher

@pytest.mark.integration
async def test_launcher_wrong():
    with patch('core.ccxt_client.CcxtClient'):  # Too late!
        launcher = LiveTradingLauncher()  # Already imported
```

## Common Test Fixtures

### `clean_env`
Provides clean environment variables for tests:
```python
def test_with_clean_env(clean_env):
    os.environ['TRADING_MODE'] = 'paper'
    # Original env restored after test
```

### `cleanup_tasks`
Ensures async tasks are cleaned up:
```python
@pytest.mark.asyncio
async def test_async_cleanup(cleanup_tasks):
    task = asyncio.create_task(my_coroutine())
    # Tasks automatically cancelled and cleaned up
```

### `mock_exchange_api`
Provides mock exchange with realistic responses:
```python
def test_with_mock_exchange(mock_exchange_api):
    price = mock_exchange_api.fetch_ticker()['last']
    assert price == 50000.0
```

### `sample_config`
Provides sample configuration:
```python
def test_with_config(sample_config):
    symbols = sample_config['universe']['fixed_symbols']
    assert len(symbols) == 2
```

## Test Markers

### @pytest.mark.integration
Marks test as integration test (slow, end-to-end):
```python
@pytest.mark.integration
async def test_full_system():
    # Test full system behavior
    pass
```

### @pytest.mark.unit
Marks test as unit test (fast, isolated):
```python
@pytest.mark.unit
def test_single_function():
    # Test single function
    pass
```

### @pytest.mark.slow
Marks test as slow (>5 seconds):
```python
@pytest.mark.slow
def test_long_operation():
    # Test that takes time
    pass
```

### @pytest.mark.asyncio
Required for async tests:
```python
@pytest.mark.asyncio
async def test_async():
    result = await async_function()
    assert result is not None
```

## Debugging Tests

### Run with Print Output
```bash
pytest tests/test_file.py -v -s
```

### Run with Python Debugger
```bash
pytest tests/test_file.py --pdb
```

### Run Single Test with Verbose Output
```bash
pytest tests/test_file.py::test_name -vv -s
```

### Show Local Variables on Failure
```bash
pytest tests/test_file.py -l
```

## Test Coverage

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### View Coverage Report
```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Environment Requirements

### Python Version
- **Required:** Python 3.11.x
- **Note:** Python 3.12+ not supported due to aiohttp 3.8.6 compatibility
- **Testing:** Tests can run on Python 3.12 with SKIP_PYTHON_VERSION_CHECK=1

### Dependencies
All test dependencies are in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Core test dependencies:
- `pytest>=7.0.0` - Test framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-timeout>=2.1.0` - Test timeout support
- `pytest-mock>=3.11.0` - Mock support

## Best Practices

### 1. Test Real Behavior
```python
# ✅ GOOD: Test what actually happens
def test_actual_behavior():
    result = function_under_test()
    assert result == actual_expected_value

# ❌ BAD: Test assumed behavior
def test_assumed_behavior():
    result = function_under_test()
    assert result == what_i_think_it_should_be
```

### 2. Mock at the Right Level
```python
# ✅ GOOD: Mock external API calls
with patch('ccxt.pro.bingx.fetch_ticker') as mock:
    mock.return_value = {'last': 50000.0}
    # Test uses real ccxt library, mock data source

# ❌ BAD: Mock entire library
with patch('ccxt.pro.bingx') as mock:
    # Breaks library functionality
```

### 3. Use Fixtures for Common Setup
```python
# ✅ GOOD: Use fixture
def test_with_fixture(clean_env, mock_exchange_api):
    # Setup done automatically
    pass

# ❌ BAD: Duplicate setup in every test
def test_manual_setup():
    # Manually set up env
    # Manually create mocks
    # Lots of repeated code
```

### 4. Test One Thing Per Test
```python
# ✅ GOOD: Single purpose
def test_function_returns_correct_value():
    assert function() == expected

def test_function_handles_error():
    with pytest.raises(ValueError):
        function(invalid_input)

# ❌ BAD: Multiple unrelated assertions
def test_everything():
    assert function() == expected
    assert other_function() == other_expected
    # Confusing if it fails
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Manual trigger via GitHub Actions

CI configuration: `.github/workflows/test.yml`

## Troubleshooting

### Tests Hang/Freeze
- Check for missing `await` in async code
- Verify cleanup_tasks fixture is used
- Look for blocking I/O in async context

### Import Errors
- Ensure `sys.path` includes src and scripts directories
- Check if mocks are set up before imports
- Verify dependencies are installed

### Mock Issues
- Mock at the import point, not definition point
- Use `patch.object` for specific attributes
- Check mock call_args if mock not called as expected

### Timeout Errors
- Increase timeout in pytest.ini if legitimately slow
- Check for infinite loops or deadlocks
- Verify async tasks complete properly

## Contributing

When adding new tests:
1. Follow existing patterns in similar tests
2. Use appropriate fixtures from conftest.py
3. Add docstrings explaining what's being tested
4. Ensure tests pass locally before committing
5. Update this README if adding new test categories

## Questions?

For questions about testing:
- Check existing tests for examples
- Review conftest.py for available fixtures
- See pytest documentation: https://docs.pytest.org/
- Open an issue with the `testing` label
