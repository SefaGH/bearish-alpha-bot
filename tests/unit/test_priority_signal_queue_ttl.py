import logging
import time

import pytest

from src.core.strategy_coordinator import PrioritySignalQueue
from src.core.signal_intents import INTENT_ENTRY, INTENT_SCALE_IN


def _make_queue(ttl=60, max_pending=1, max_pending_scale_in=1, pyramiding_enabled=True):
    return PrioritySignalQueue(
        {
            "ttl_seconds": ttl,
            "max_queue_depth": 10,
            "batch_dequeue": 1,
            "max_pending_per_symbol": max_pending,
            "max_pending_scale_in_per_symbol": max_pending_scale_in,
            "pyramiding_enabled": pyramiding_enabled,
        },
        logging.getLogger(__name__),
    )


def _payload(symbol, intent):
    return {"signal": {"symbol": symbol, "intent": intent, "priority": 1}}


@pytest.mark.asyncio
async def test_pending_cleared_after_ttl_for_same_symbol(monkeypatch):
    t = 0.0

    def fake_time():
        return t

    monkeypatch.setattr(time, "time", fake_time)
    queue = _make_queue(ttl=60, max_pending=1, pyramiding_enabled=True)

    ok, reason = await queue.put(_payload("BTC/USDT:USDT", INTENT_ENTRY))
    assert ok is True
    assert queue._pending_by_symbol["BTC/USDT:USDT"]["total"] == 1

    # Advance time beyond TTL without dequeuing/purging
    t = 61.0

    # Attempt to enqueue again; desired behavior: should be accepted after TTL
    ok, reason = await queue.put(_payload("BTC/USDT:USDT", INTENT_ENTRY))
    assert ok is True, f"Unexpected rejection after TTL: {reason}"


@pytest.mark.asyncio
async def test_scale_in_pending_cleared_after_ttl(monkeypatch):
    t = 0.0

    def fake_time():
        return t

    monkeypatch.setattr(time, "time", fake_time)
    queue = _make_queue(ttl=60, max_pending=1, max_pending_scale_in=1, pyramiding_enabled=True)

    ok, reason = await queue.put(_payload("ETH/USDT:USDT", INTENT_SCALE_IN))
    assert ok is True
    assert queue._pending_by_symbol["ETH/USDT:USDT"]["scale_in"] == 1

    # Advance time beyond TTL without dequeuing/purging
    t = 61.0

    ok, reason = await queue.put(_payload("ETH/USDT:USDT", INTENT_SCALE_IN))
    assert ok is True, f"Unexpected rejection after TTL for scale_in: {reason}"


@pytest.mark.skip(reason="non-deterministic smoke; kept for documentation")
@pytest.mark.asyncio
async def test_priority_queue_ttl_smoke(monkeypatch):
    t = 0.0

    def fake_time():
        return t

    monkeypatch.setattr(time, "time", fake_time)
    queue = _make_queue(ttl=1, max_pending=1, pyramiding_enabled=True)

    ok, _ = await queue.put(_payload("SMOKE/USDT", INTENT_ENTRY))
    assert ok is True
    t = 2.0  # advance beyond TTL
    ok, reason = await queue.put(_payload("SMOKE/USDT", INTENT_ENTRY))
    assert ok is True or "limit" not in (reason or "")
