import json
import logging
import uuid
from datetime import datetime, timezone

import pytest

from src.core.strategy_coordinator import StrategyCoordinator


class DummyPortfolioManager:
    cfg = {}
    performance_monitor = None
    exchange_clients = {}

    def get_current_equity(self):
        return 1000.0

    def get_strategy_allocation(self, strategy_name: str):
        return 1.0

    def get_open_positions_for_symbol(self, symbol: str):
        return []


@pytest.mark.asyncio
async def test_incubate_signal_stores_json_serializable_item():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
    }

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    assert result["status"] == "incubated"
    dedupe_key = result["dedupe_key"]

    item = coordinator._incubator_items[dedupe_key]
    json.dumps(item)  # must not raise
    assert item["reason_code"] == "queue.capacity"
    assert item["payload"]["dedupe_key"] == dedupe_key


@pytest.mark.asyncio
async def test_waiting_room_events_never_emit_null_signal_id(caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    caplog.set_level(logging.INFO)

    # Trigger early incubation (volume_policy tight-stop path) before risk/queue stages.
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
        "stop": 99.9,  # 0.10% stop -> tight_stop
        "volume_bucket": "LOW",
    }

    result = await coordinator.process_strategy_signal("mean_reversion", signal)
    assert result["status"] == "incubated"

    events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_")]
    assert events, "Expected at least one waiting_room_* event"

    for msg in events:
        _, json_blob = msg.split(" ", 1)
        data = json.loads(json_blob)
        assert data.get("signal_id"), f"Missing signal_id in event: {data}"
        parsed = uuid.UUID(str(data["signal_id"]))
        assert parsed.hex == str(data["signal_id"])


@pytest.mark.asyncio
async def test_incubator_tick_replays_and_removes_on_accept(monkeypatch):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
    }
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    dedupe_key = result["dedupe_key"]
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    seen = {"called": 0, "incubator_replay": None}

    async def fake_process_strategy_signal(*, strategy_name: str, signal: dict):
        seen["called"] += 1
        seen["incubator_replay"] = bool(signal.get("incubator_replay"))
        return {"status": "accepted", "signal_id": "sig_1"}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    processed = await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    assert processed >= 1
    assert seen["called"] == 1
    assert seen["incubator_replay"] is True
    assert dedupe_key not in coordinator._incubator_items


@pytest.mark.asyncio
async def test_process_strategy_signal_refreshes_existing_waiting_item():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    base_signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
    }
    incubated = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=base_signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    dedupe_key = incubated["dedupe_key"]

    updated_signal = dict(base_signal)
    updated_signal["entry"] = 101.0
    result = await coordinator.process_strategy_signal("mean_reversion", updated_signal)
    assert result["status"] == "incubated"
    assert result["dedupe_key"] == dedupe_key
    assert coordinator._incubator_items[dedupe_key]["payload"]["entry"] == 101.0


@pytest.mark.asyncio
async def test_ingress_normalizes_side_and_refreshes_matching_waiting_item():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    waiting_signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
    }
    incubated = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=waiting_signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    dedupe_key = incubated["dedupe_key"]

    incoming = {
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",  # should normalize to long before dedupe check
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 101.0,
    }
    result = await coordinator.process_strategy_signal("mean_reversion", incoming)
    assert incoming["side"] == "long"
    assert result["status"] == "incubated"
    assert result["dedupe_key"] == dedupe_key
    assert coordinator._incubator_items[dedupe_key]["payload"]["side"] == "long"


@pytest.mark.asyncio
async def test_quality_below_threshold_emits_waiting_room_drop(caplog, monkeypatch):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)

    async def fake_assess(_signal: dict, _strategy_name: str):
        return {
            "acceptable": False,
            "reason": "scale_in_quality_below_threshold",
            "reason_code": "risk.scale_in.quality_below_threshold",
            "metrics": {"blocked_by": "RiskManager._can_dynamic_scale"},
        }

    async def fake_conflicts(_signal: dict):
        return {"has_conflict": False, "conflicts": [], "conflicting_signals": []}

    monkeypatch.setattr(coordinator, "_assess_signal_risk", fake_assess)
    monkeypatch.setattr(coordinator, "_check_signal_conflicts", fake_conflicts)
    monkeypatch.setattr(coordinator, "validate_duplicate", lambda *_args, **_kwargs: (True, "ok"))

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
        "stop": 90.0,
        "target": 120.0,
        "volume_bucket": "NORMAL",
    }

    result = await coordinator.process_strategy_signal("mean_reversion", signal)
    assert result["status"] == "rejected"
    assert result.get("reason_code") == "risk.scale_in.quality_below_threshold"

    drop_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_drop ")]
    assert drop_events, "Expected a waiting_room_drop event"

    _, json_blob = drop_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("reason_code") == "incubator.blocked.risk.scale_in.quality_below_threshold"
    assert data.get("drop_kind") == "sanity_guard"
