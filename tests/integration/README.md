# Integration Tests

This directory contains end-to-end integration tests that verify the bot's behavior in production-like scenarios.

## Purpose

These tests address real-world issues that unit tests with mocks cannot catch:
- **Bot freezes** due to deadlocks or blocking code
- **WebSocket streams not delivering data** (tasks not scheduled correctly)
- **Async event loop blocks** (improperly awaited coroutines)
- **Cross-module config inconsistencies**
- **Timeout issues** in production scenarios

## Test Files

### `test_launcher_integration.py`
- `test_launcher_runs_without_freeze()` - **CRITICAL**: Verifies bot runs 30s without freezing (Issue #153)
- `test_async_tasks_properly_scheduled()` - Verifies async task management (Issue #160)
- `test_launcher_initialization_phases()` - Verifies all 8 initialization phases complete

### `test_websocket_streaming.py`
- `test_websocket_streams_deliver_data()` - Verifies WebSocket data delivery (Issue #160)
- `test_websocket_connection_state_tracking()` - Verifies connection state tracking (Issue #159)
- `test_websocket_error_handling()` - Verifies error recovery mechanisms

### `test_config_consistency.py`
- `test_config_consistency_across_all_modules()` - Verifies unified config (Issue #157)
- `test_env_priority_over_yaml()` - Verifies ENV overrides YAML
- `test_config_validation()` - Verifies config validation
- `test_runtime_config_consistency()` - Verifies no config drift

## Running Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v -m integration
```

### Run specific test:
```bash
pytest tests/integration/test_launcher_integration.py::test_launcher_runs_without_freeze -v -s
```

### Run with full output:
```bash
pytest tests/integration/ -v -s --capture=no
```

## Environment Requirements

### Minimal (for mocked tests):
- Python 3.11+
- pytest>=7.0.0
- pytest-asyncio>=0.21.0
- pytest-timeout>=2.1.0
- pytest-mock>=3.11.0
- pandas, numpy, pyyaml, python-dotenv

### Full (for real integration tests):
- All dependencies from requirements.txt
- Valid config/config.example.yaml file

## Test Strategy

The integration tests use a hybrid approach:
1. **Mock external APIs** (exchange connections, Telegram) to avoid real API calls
2. **Run actual code paths** (launcher, coordinator, strategies) to catch integration bugs
3. **Verify behavior** (no freezes, proper async scheduling, config consistency)

This approach catches real integration issues while keeping tests fast and reliable.

## CI/CD Integration

Integration tests run automatically on:
- Pull requests (when code in src/, scripts/, or tests/integration/ changes)
- Push to main/develop branches
- Nightly at 2 AM UTC
- Manual workflow dispatch

See `.github/workflows/integration-tests.yml` for details.

## Related Issues

- Issue #153: Bot Freeze at Startup
- Issue #157: Configuration Loading Priority Conflict
- Issue #159: WebSocket Connection State Tracking
- Issue #160: WebSocket Task Management
- Issue #165: Integration Tests for Bot Freeze Detection (this implementation)
