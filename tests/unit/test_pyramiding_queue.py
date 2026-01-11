import asyncio
import logging
import pytest

from src.core.strategy_coordinator import PrioritySignalQueue
from src.core.signal_intents import INTENT_ENTRY, INTENT_SCALE_IN


def _make_payload(symbol, intent, score=0.5):
    return {
        "signal": {
            "symbol": symbol,
            "intent": intent,
            "priority": score,
        }
    }


@pytest.mark.asyncio
async def test_pyramiding_disabled_pending_limit_applies_to_scale_in():
    queue = PrioritySignalQueue(
        {
            "ttl_seconds": 60,
            "max_queue_depth": 5,
            "max_pending_per_symbol": 1,
            "max_pending_scale_in_per_symbol": 2,
            "pyramiding_enabled": False,
        },
        logging.getLogger(__name__),
    )
    ok, _, _ = await queue.put(_make_payload("BTC", INTENT_SCALE_IN))
    assert ok is True
    ok, reason, _ = await queue.put(_make_payload("BTC", INTENT_SCALE_IN))
    assert ok is False
    assert "limit" in reason.lower()


@pytest.mark.asyncio
async def test_pyramiding_enabled_allows_extra_scale_in_pending():
    queue = PrioritySignalQueue(
        {
            "ttl_seconds": 60,
            "max_queue_depth": 10,
            "max_pending_per_symbol": 1,
            "max_pending_scale_in_per_symbol": 2,
            "pyramiding_enabled": True,
        },
        logging.getLogger(__name__),
    )
    # entry slot
    assert (await queue.put(_make_payload("ETH", INTENT_ENTRY)))[0] is True
    # two scale-ins allowed
    assert (await queue.put(_make_payload("ETH", INTENT_SCALE_IN)))[0] is True
    assert (await queue.put(_make_payload("ETH", INTENT_SCALE_IN)))[0] is True
    # third scale-in rejected
    ok, reason, _ = await queue.put(_make_payload("ETH", INTENT_SCALE_IN))
    assert ok is False
    assert "scale" in (reason or "").lower()


@pytest.mark.asyncio
async def test_entry_stays_strict_even_with_scale_in_capacity():
    queue = PrioritySignalQueue(
        {
            "ttl_seconds": 60,
            "max_queue_depth": 5,
            "max_pending_per_symbol": 1,
            "max_pending_scale_in_per_symbol": 2,
            "pyramiding_enabled": True,
        },
        logging.getLogger(__name__),
    )
    assert (await queue.put(_make_payload("SOL", INTENT_ENTRY)))[0] is True
    ok, reason, _ = await queue.put(_make_payload("SOL", INTENT_ENTRY))
    assert ok is False
    assert "limit" in reason.lower()
