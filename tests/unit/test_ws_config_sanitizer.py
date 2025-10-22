"""Unit test for WebSocket config sanitizer to handle malformed configs gracefully."""

import pytest
import asyncio

from scripts.live_trading_launcher import OptimizedWebSocketManager


@pytest.mark.asyncio
async def test_malformed_config_sanitized_returns_empty_tasks():
    """A malformed config with placeholder strings should be sanitized and not raise."""
    mgr = OptimizedWebSocketManager(config={'websocket': 'dict', 'universe': 'dict'})

    # Call setup_from_config explicitly (the constructor may set config but we test the sanitizer)
    mgr.setup_from_config({'websocket': 'dict', 'universe': 'dict'})

    # Provide empty exchange_clients to avoid importing real WebSocketManager
    tasks = await mgr.initialize_websockets(exchange_clients={})

    assert isinstance(tasks, list)
    assert tasks == []
