import json
import logging
import uuid
from datetime import datetime, timezone

import pytest

from src.core.strategy_coordinator import StrategyCoordinator
from src.core.production_coordinator import ProductionCoordinator


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


def test_waiting_room_distinct_guard_does_not_mutate_payload_signal_id(caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    caplog.set_level(logging.INFO)
    shared_id = uuid.uuid4().hex
    item = {
        "pending_id": shared_id,
        "payload": {
            "strategy_name": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "signal_id": shared_id,
        },
        "reason_code": "queue.capacity",
        "attempts": 0,
        "ttl_seconds": 5,
        "first_seen_ts_ms": coordinator._now_ms(),
        "dedupe_key": "dummy",
    }

    coordinator._emit_waiting_room_event("waiting_room_retry", item)
    assert item["payload"]["signal_id"] == shared_id

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("pending_id") == shared_id
    assert data.get("signal_id") != shared_id


def test_soft_deferral_salvaged_is_idempotent(caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)

    parent_pending_id = "p" * 32
    signal_id = uuid.uuid4().hex
    payload = {
        "strategy_name": "mean_reversion",
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "reason_code": "strategy.mean_reversion.entry",
        "meta": {"pending_reason_code": "strategy.mean_reversion.near_miss"},
    }

    coordinator._emit_soft_deferral_salvaged(
        parent_pending_id=parent_pending_id,
        signal_id=signal_id,
        final_status="executing",
        signal_payload=payload,
    )
    coordinator._emit_soft_deferral_salvaged(
        parent_pending_id=parent_pending_id,
        signal_id=signal_id,
        final_status="accepted",
        signal_payload=payload,
    )

    events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_salvaged ")]
    assert len(events) == 1
    _, json_blob = events[0].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("parent_pending_id") == parent_pending_id
    assert data.get("signal_id") == signal_id
    assert data.get("final_status") == "accepted"
    assert data.get("pending_reason_code") == "strategy.mean_reversion.near_miss"


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
async def test_incubator_replay_emits_waiting_room_outcome_success(monkeypatch, caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    caplog.set_level(logging.INFO)

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
        "signal_id": uuid.uuid4().hex,
    }

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="risk.planner.heat_exhausted",
        refresh_policy="NONE",
        stage="risk",
    )
    dedupe_key = result["dedupe_key"]
    item = coordinator._incubator_items[dedupe_key]
    pending_id = item["pending_id"]
    signal_id = item["payload"]["signal_id"]

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    async def fake_process_strategy_signal(*, strategy_name: str, signal: dict):
        assert bool(signal.get("incubator_replay")) is True
        return {"status": "queued", "signal_id": signal.get("signal_id")}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    outcome_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_outcome ")]
    assert outcome_events, "Expected a waiting_room_outcome event"
    _, json_blob = outcome_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("outcome") == "success"
    assert data.get("success_kind") == "queued"
    assert data.get("pending_id") == pending_id
    assert data.get("signal_id") == signal_id
    assert data.get("final_reason") == "queued"


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


def test_low_vol_tight_stop_default_ttl_is_5_minutes():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    policy = coordinator._incubator_policies.get("volume.low_vol_tight_stop") or {}
    assert policy.get("ttl_seconds") == 300


@pytest.mark.asyncio
async def test_incubate_signal_dedupe_update_preserves_pending_id_and_timers():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    initial_signal_id = "0" * 32
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
        "signal_id": initial_signal_id,
    }

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    dedupe_key = result["dedupe_key"]
    item = coordinator._incubator_items[dedupe_key]

    assert item.get("pending_id")
    assert item.get("pending_id") != item["payload"]["signal_id"]
    assert item["payload"]["signal_id"] == initial_signal_id
    preserved = {k: item[k] for k in ("pending_id", "first_seen_ts_ms", "expires_at_ms", "attempts", "next_check_at_ms")}

    updated_signal_id = "1" * 32
    updated = dict(signal)
    updated["entry"] = 101.0
    updated["signal_id"] = updated_signal_id
    result2 = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=updated,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    assert result2["status"] == "incubated"
    assert result2["dedupe_key"] == dedupe_key

    item2 = coordinator._incubator_items[dedupe_key]
    assert item2.get("pending_id") == preserved["pending_id"]
    assert item2.get("first_seen_ts_ms") == preserved["first_seen_ts_ms"]
    assert item2.get("expires_at_ms") == preserved["expires_at_ms"]
    assert item2.get("attempts") == preserved["attempts"]
    assert item2.get("next_check_at_ms") == preserved["next_check_at_ms"]
    assert item2["payload"]["entry"] == 101.0
    assert item2["payload"]["signal_id"] == updated_signal_id
    assert item2.get("pending_id") != item2["payload"]["signal_id"]


@pytest.mark.asyncio
async def test_incubator_tick_does_not_increment_attempts_on_ctx_hash_unchanged_skip(monkeypatch, caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
        "signal_id": uuid.uuid4().hex,
    }
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="volume.low_vol_tight_stop",
        refresh_policy="REPRICE_AND_RESIZE",
        stage="volume_policy",
    )
    dedupe_key = result["dedupe_key"]
    item = coordinator._incubator_items[dedupe_key]
    item["next_check_at_ms"] = 0
    item["ctx_hash"] = "CTX"
    item["attempts"] = 0
    item["max_attempts"] = 1

    async def fake_compute_ctx_hash(_payload: dict, *, now_ms=None, price_cache=None, reason_code=None):
        return "CTX", {"fake": True}

    async def fake_check_volume_release_condition(_payload: dict):
        raise AssertionError("_check_volume_release_condition should not run when ctx_hash_unchanged")

    monkeypatch.setattr(coordinator, "_compute_ctx_hash", fake_compute_ctx_hash)
    monkeypatch.setattr(coordinator, "_check_volume_release_condition", fake_check_volume_release_condition)

    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    assert dedupe_key in coordinator._incubator_items
    assert coordinator._incubator_items[dedupe_key]["attempts"] == 0

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    assert dedupe_key in coordinator._incubator_items
    assert coordinator._incubator_items[dedupe_key]["attempts"] == 0

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("check_detail", {}).get("skip_reason") == "ctx_hash_unchanged"


@pytest.mark.asyncio
async def test_waiting_room_retry_emits_stable_pending_id_and_latest_signal_id(monkeypatch, caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)

    async def fake_can_accept(_payload):
        return False, None, None

    async def fake_compute_ctx_hash(_payload: dict, *, now_ms=None, price_cache=None, reason_code=None):
        return None, {}

    monkeypatch.setattr(coordinator.signal_queue, "can_accept", fake_can_accept)
    monkeypatch.setattr(coordinator, "_compute_ctx_hash", fake_compute_ctx_hash)

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    signal_id_1 = uuid.uuid4().hex
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
        "signal_id": signal_id_1,
    }
    incubated = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    dedupe_key = incubated["dedupe_key"]
    pending_id = coordinator._incubator_items[dedupe_key]["pending_id"]
    assert pending_id != signal_id_1

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("pending_id") == pending_id
    assert data.get("signal_id") == signal_id_1
    assert data.get("pending_id") != data.get("signal_id")

    caplog.clear()
    signal_id_2 = uuid.uuid4().hex
    updated = dict(signal)
    updated["entry"] = 101.0
    updated["signal_id"] = signal_id_2
    refreshed = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=updated,
        reason_code="queue.capacity",
        refresh_policy="NONE",
        stage="queue",
    )
    assert refreshed["status"] == "incubated"
    assert refreshed["dedupe_key"] == dedupe_key

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("pending_id") == pending_id
    assert data.get("signal_id") == signal_id_2
    assert data.get("pending_id") != data.get("signal_id")


@pytest.mark.asyncio
async def test_heat_ctx_hash_changes_when_price_changes(monkeypatch, caplog):
    class DummyRiskManager:
        def get_portfolio_summary(self, portfolio_manager=None):
            return {
                "active_positions": 1,
                "total_risk": 10.0,
                "portfolio_heat": 0.01,
            }

    class DummyMarketDataPipeline:
        def __init__(self):
            self._prices = [100.0, 101.0]
            self._i = 0

        async def get_latest_price(self, symbol: str, timeframe: str = "1m", exchange: str = None):
            price = self._prices[min(self._i, len(self._prices) - 1)]
            self._i += 1
            return price

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=DummyRiskManager(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    coordinator.market_data_pipeline = DummyMarketDataPipeline()
    caplog.set_level(logging.INFO)

    async def fake_process_strategy_signal(*, strategy_name: str, signal: dict):
        return {"status": "incubated", "dedupe_key": coordinator._derive_dedupe_key(strategy_name, signal)}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "entry": 100.0,
        "signal_id": "4" * 32,
    }

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="risk.planner.heat_exhausted",
        refresh_policy="NONE",
        stage="risk",
    )
    dedupe_key = result["dedupe_key"]

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data_1 = json.loads(json_blob)
    assert data_1.get("check_detail", {}).get("skip_reason") != "ctx_hash_unchanged"
    ctx_hash_1 = data_1.get("ctx_hash")

    caplog.clear()
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data_2 = json.loads(json_blob)
    assert data_2.get("check_detail", {}).get("skip_reason") != "ctx_hash_unchanged"
    ctx_hash_2 = data_2.get("ctx_hash")

    assert ctx_hash_1
    assert ctx_hash_2
    assert ctx_hash_1 != ctx_hash_2


@pytest.mark.asyncio
async def test_production_coordinator_dispatches_strategy_recheck_requests(monkeypatch):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator

    calls = []

    async def fake_dispatch(symbol: str, strategy: str, *, parent_pending_id: str | None = None, **_kwargs):
        calls.append((symbol, strategy))
        return True

    monkeypatch.setattr(prod, "dispatch_strategy", fake_dispatch)

    result = await coordinator.handle_soft_deferral(
        {
            "event_type": "soft_deferral_event",
            "strategy": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "timeframe": "5m",
            "setup_anchor_ts_ms": int(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000),
            "reason_code": "strategy.mean_reversion.near_miss",
            "reason": "budget_unavailable",
            "condition_data": {"timeframe": "5m"},
        }
    )
    dedupe_key = result["dedupe_key"]
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    await prod._drain_strategy_recheck_requests()

    assert calls == [("BTC/USDT:USDT", "mean_reversion")]


@pytest.mark.asyncio
async def test_handle_soft_deferral_canonicalizes_buy_sell_side():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    anchor_ms = int(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    result_buy = await coordinator.handle_soft_deferral(
        {
            "event_type": "soft_deferral_event",
            "strategy": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "BUY",
            "timeframe": "5m",
            "setup_anchor_ts_ms": anchor_ms,
            "reason_code": "strategy.mean_reversion.near_miss",
            "condition_data": {"timeframe": "5m"},
        }
    )
    assert result_buy["status"] == "incubated"
    buy_key = result_buy["dedupe_key"]
    assert coordinator._incubator_items[buy_key]["payload"]["side"] == "long"

    result_sell = await coordinator.handle_soft_deferral(
        {
            "event_type": "soft_deferral_event",
            "strategy": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "SELL",
            "timeframe": "5m",
            "setup_anchor_ts_ms": anchor_ms,
            "reason_code": "strategy.mean_reversion.near_miss",
            "condition_data": {"timeframe": "5m"},
        }
    )
    assert result_sell["status"] == "incubated"
    sell_key = result_sell["dedupe_key"]
    assert coordinator._incubator_items[sell_key]["payload"]["side"] == "short"


@pytest.mark.asyncio
async def test_handle_soft_deferral_invalid_schema_emits_waiting_room_drop(caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)
    caplog.clear()

    result = await coordinator.handle_soft_deferral(
        {
            "event_type": "soft_deferral_event",
            "strategy": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "BUY",
            "reason_code": "strategy.mean_reversion.near_miss",
            # missing timeframe + setup_anchor_ts_ms
        }
    )
    assert result["status"] == "rejected"

    drop_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_drop ")]
    assert drop_events, "Expected waiting_room_drop telemetry for soft deferral reject"
    _, json_blob = drop_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert str(data.get("reason_code", "")).startswith("soft_deferral.")
    assert data.get("drop_kind") == "soft_deferral_reject"
    assert data.get("error")


@pytest.mark.asyncio
async def test_risk_planner_heat_exhausted_is_incubated_with_distinct_pending_id(monkeypatch, caplog):
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    caplog.set_level(logging.INFO)

    async def fake_assess(_signal: dict, _strategy_name: str):
        return {
            "acceptable": False,
            "reason": "portfolio_heat_exhausted",
            "reason_code": "risk.planner.heat_exhausted",
            "metrics": {},
        }

    async def fake_conflicts(_signal: dict):
        return {"has_conflict": False, "conflicts": [], "conflicting_signals": []}

    monkeypatch.setattr(coordinator, "_assess_signal_risk", fake_assess)
    monkeypatch.setattr(coordinator, "_check_signal_conflicts", fake_conflicts)
    monkeypatch.setattr(coordinator, "validate_duplicate", lambda *_args, **_kwargs: (True, "ok"))

    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    signal_id_1 = "2" * 32
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "5m",
        "intent": "entry",
        "timestamp": base_ts,
        "entry": 100.0,
        "stop": 90.0,
        "target": 120.0,
        "volume_bucket": "NORMAL",
        "signal_id": signal_id_1,
    }

    result = await coordinator.process_strategy_signal("mean_reversion", signal)
    assert result["status"] == "incubated"
    assert result["reason_code"] == "risk.planner.heat_exhausted"

    dedupe_key = result["dedupe_key"]
    item = coordinator._incubator_items[dedupe_key]
    pending_id = item["pending_id"]
    assert pending_id
    assert pending_id != item["payload"]["signal_id"]
    assert item["payload"]["signal_id"] == signal_id_1

    signal_id_2 = "3" * 32
    updated = dict(signal)
    updated["entry"] = 101.0
    updated["signal_id"] = signal_id_2
    refreshed = await coordinator.process_strategy_signal("mean_reversion", updated)
    assert refreshed["status"] == "incubated"
    assert refreshed["dedupe_key"] == dedupe_key

    item2 = coordinator._incubator_items[dedupe_key]
    assert item2["pending_id"] == pending_id
    assert item2["payload"]["signal_id"] == signal_id_2
    assert item2["pending_id"] != item2["payload"]["signal_id"]

    item2["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    retry_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_retry ")]
    assert retry_events
    _, json_blob = retry_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("pending_id") == pending_id
    assert data.get("signal_id") == signal_id_2
    assert data.get("pending_id") != data.get("signal_id")
