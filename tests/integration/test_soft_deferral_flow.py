import json
import logging
from collections import OrderedDict

import pandas as pd
import pytest

from src.core.production_coordinator import ProductionCoordinator
from src.core.strategy_coordinator import StrategyCoordinator
from src.strategies.mean_reversion import VWAPMeanReversion


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
@pytest.mark.integration
async def test_soft_deferral_flow_routes_and_dedupes_long_short(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

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

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )

    symbol = "BTC/USDT:USDT"
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )

    df_sig_long = pd.DataFrame({"close": [99.2, 99.2], "adx": [10.0, 10.0]}, index=idx)
    df_sig_short = pd.DataFrame({"close": [100.8, 100.8], "adx": [10.0, 10.0]}, index=idx)

    event_long = await strategy.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig_long)
    event_short = await strategy.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig_short)
    assert isinstance(event_long, dict)
    assert isinstance(event_short, dict)
    assert event_long.get("event_type") == "soft_deferral_event"
    assert event_short.get("event_type") == "soft_deferral_event"
    assert event_long.get("side") != event_short.get("side")

    caplog.clear()

    result_long = await prod._route_strategy_output("mean_reversion", event_long)
    result_short = await prod._route_strategy_output("mean_reversion", event_short)
    assert result_long and result_long.get("status") == "incubated"
    assert result_short and result_short.get("status") == "incubated"

    add_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_add ")]
    assert len(add_events) == 2
    for msg in add_events:
        _, json_blob = msg.split(" ", 1)
        data = json.loads(json_blob)
        assert data.get("reason_code") == "strategy.mean_reversion.near_miss"

    coordinator._incubator_items[result_long["dedupe_key"]]["next_check_at_ms"] = 0
    coordinator._incubator_items[result_short["dedupe_key"]]["next_check_at_ms"] = 0

    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    await prod._drain_strategy_recheck_requests()

    assert calls == [(symbol, "mean_reversion"), (symbol, "mean_reversion")]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_recheck_dedupe_mode_timeframe_toggle(monkeypatch):
    symbol = "BTC/USDT:USDT"
    anchor_ms = int(pd.Timestamp("2026-01-01T00:05:00Z").timestamp() * 1000)

    base_event = {
        "event_type": "soft_deferral_event",
        "strategy": "mean_reversion",
        "symbol": symbol,
        "side": "BUY",
        "setup_anchor_ts_ms": anchor_ms,
        "reason_code": "strategy.mean_reversion.near_miss",
        "condition_data": {"source": "test"},
    }

    event_tf_5m = {**base_event, "timeframe": "5m"}
    event_tf_15m = {**base_event, "timeframe": "15m"}

    async def _run(mode: str) -> int:
        coordinator = StrategyCoordinator(
            DummyPortfolioManager(),
            risk_manager=object(),
            config={"risk": {"queue": {"ttl_seconds": 5}}},
        )
        prod = ProductionCoordinator.__new__(ProductionCoordinator)
        prod.strategy_coordinator = coordinator
        prod.config = {"incubator": {"recheck_dedupe_mode": mode}}

        calls = []

        async def fake_dispatch(_symbol: str, _strategy: str, *, parent_pending_id: str | None = None, **_kwargs):
            calls.append((_symbol, _strategy))
            return True

        monkeypatch.setattr(prod, "dispatch_strategy", fake_dispatch)

        result_5m = await prod._route_strategy_output("mean_reversion", event_tf_5m)
        result_15m = await prod._route_strategy_output("mean_reversion", event_tf_15m)
        assert result_5m and result_5m.get("status") == "incubated"
        assert result_15m and result_15m.get("status") == "incubated"

        coordinator._incubator_items[result_5m["dedupe_key"]]["next_check_at_ms"] = 0
        coordinator._incubator_items[result_15m["dedupe_key"]]["next_check_at_ms"] = 0
        await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
        await prod._drain_strategy_recheck_requests()
        return len(calls)

    assert await _run("strategy_symbol_side") == 1
    assert await _run("strategy_symbol_side_timeframe") == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_invalid_schema_emits_waiting_room_drop(caplog):
    caplog.set_level(logging.INFO)

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator

    caplog.clear()

    invalid_event = {
        "event_type": "soft_deferral_event",
        "strategy": "mean_reversion",
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "setup_anchor_ts_ms": int(pd.Timestamp("2026-01-01T00:05:00Z").timestamp() * 1000),
        "reason_code": "strategy.mean_reversion.near_miss",
        # missing timeframe
    }

    result = await prod._route_strategy_output("mean_reversion", invalid_event)
    assert result and result.get("status") == "rejected"

    drop_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_drop ")]
    assert drop_events, "Expected waiting_room_drop telemetry for soft deferral reject"
    _, json_blob = drop_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("reason_code") == "soft_deferral.missing_timeframe"
    assert data.get("drop_kind") == "soft_deferral_reject"
    assert data.get("error")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mean_reversion_soft_deferral_rate_limited_per_candle():
    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )
    symbol = "BTC/USDT:USDT"
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [99.2, 99.2], "adx": [10.0, 10.0]}, index=idx)

    first = await strategy.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)
    second = await strategy.generate_signal(symbol=symbol, df_vwap=df_vwap, df_sig=df_sig)

    assert isinstance(first, dict)
    assert first.get("event_type") == "soft_deferral_event"
    assert second is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_parent_pending_id_propagates_to_child_signal(monkeypatch):
    symbol = "BTC/USDT:USDT"
    pending_id = "p" * 32

    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [98.5, 98.5], "adx": [10.0, 10.0]}, index=idx)

    class DummyPipeline:
        async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
            if timeframe == "1m":
                return df_vwap
            if timeframe == "5m":
                return df_sig
            return df_sig

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )

    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    coordinator = StrategyCoordinator(
        portfolio,
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    captured = {}

    async def fake_process_strategy_signal(strategy_name: str, signal: dict):
        captured["strategy_name"] = strategy_name
        captured["signal"] = signal
        return {"status": "accepted", "signal_id": "child_sig_1"}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}

    dispatched = await prod._handle_strategy_recheck_request(
        {
            "event": "strategy_recheck_request",
            "strategy": "mean_reversion",
            "symbol": symbol,
            "pending_id": pending_id,
        }
    )
    assert dispatched is True
    assert captured.get("strategy_name") == "mean_reversion"
    meta = captured.get("signal", {}).get("meta")
    assert isinstance(meta, dict)
    assert meta.get("parent_pending_id") == pending_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_recheck_near_miss_loop_prevented_emits_no_signal_outcome(caplog):
    caplog.set_level(logging.INFO)

    symbol = "BTC/USDT:USDT"
    pending_id = "p" * 32

    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [99.2, 99.2], "adx": [10.0, 10.0]}, index=idx)

    class DummyPipeline:
        async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
            if timeframe == "1m":
                return df_vwap
            if timeframe == "5m":
                return df_sig
            return df_sig

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )
    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    coordinator = StrategyCoordinator(
        portfolio,
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}

    caplog.clear()
    dispatched = await prod._handle_strategy_recheck_request(
        {
            "event": "strategy_recheck_request",
            "strategy": "mean_reversion",
            "symbol": symbol,
            "parent_pending_id": pending_id,
            "side": "BUY",
            "timeframe": "5m",
            "reason_code": "strategy.mean_reversion.near_miss",
        }
    )
    assert dispatched is False

    outcome_events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_recheck_outcome ")]
    assert outcome_events, "Expected soft_deferral_recheck_outcome telemetry for recheck"
    _, json_blob = outcome_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("parent_pending_id") == pending_id
    assert data.get("outcome") == "no_signal"
    assert data.get("attempt") == 1
    assert data.get("side") == "long"
    assert data.get("timeframe") == "5m"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_recheck_outcome_includes_signal_id_on_signal_emitted(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    symbol = "BTC/USDT:USDT"
    pending_id = "p" * 32

    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [98.5, 98.5], "adx": [10.0, 10.0]}, index=idx)

    class DummyPipeline:
        async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
            if timeframe == "1m":
                return df_vwap
            if timeframe == "5m":
                return df_sig
            return df_sig

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "soft_deferral_threshold": 0.005,
            "dynamic_controller": {"enabled": False},
        }
    )
    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    coordinator = StrategyCoordinator(
        portfolio,
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    async def fake_process_strategy_signal(strategy_name: str, signal: dict):
        return {"status": "accepted", "signal_id": "child_sig_1"}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}

    caplog.clear()
    dispatched = await prod._handle_strategy_recheck_request(
        {
            "event": "strategy_recheck_request",
            "strategy": "mean_reversion",
            "symbol": symbol,
            "parent_pending_id": pending_id,
            "side": "SELL",
            "timeframe": "5m",
            "reason_code": "strategy.mean_reversion.near_miss",
        }
    )
    assert dispatched is True

    outcome_events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_recheck_outcome ")]
    assert outcome_events, "Expected soft_deferral_recheck_outcome telemetry for recheck"
    _, json_blob = outcome_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("parent_pending_id") == pending_id
    assert data.get("outcome") == "signal_emitted"
    assert data.get("signal_id") == "child_sig_1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_salvaged_emitted_on_queued_replay(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    pending_id = "p" * 32
    anchor_ms = int(pd.Timestamp("2026-01-01T00:05:00Z").timestamp() * 1000)

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
        "entry": 100.0,
        "setup_anchor_ts_ms": anchor_ms,
        "timestamp": anchor_ms,
        "meta": {
            "parent_pending_id": pending_id,
            "pending_reason_code": "strategy.mean_reversion.near_miss",
        },
    }

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="risk.planner.heat_exhausted",
        refresh_policy="NONE",
        stage="risk",
    )
    assert result["status"] == "incubated"
    dedupe_key = result["dedupe_key"]

    async def fake_process_strategy_signal(*, strategy_name: str, signal: dict):
        return {"status": "queued", "signal_id": "child_sig_queued"}

    monkeypatch.setattr(coordinator, "process_strategy_signal", fake_process_strategy_signal)

    caplog.clear()
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)

    salvage_events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_salvaged ")]
    assert salvage_events, "Expected soft_deferral_salvaged for queued outcome"
    _, json_blob = salvage_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("parent_pending_id") == pending_id
    assert data.get("signal_id") == "child_sig_queued"
    assert data.get("final_status") == "queued"
    assert data.get("pending_reason_code") == "strategy.mean_reversion.near_miss"


def test_soft_deferral_salvaged_idempotency_cache_is_bounded():
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}, "soft_deferral": {"salvage_idempotency_max_items": 5}},
    )
    max_items = coordinator._soft_deferral_salvage_cache_max()
    assert max_items == 5

    for i in range(max_items + 5):
        coordinator._mark_soft_deferral_salvaged(f"{i:032d}")

    cache = getattr(coordinator, "_salvaged_parent_ids", None)
    assert isinstance(cache, OrderedDict)
    assert len(cache) == max_items
    assert f"{0:032d}" not in cache


def test_soft_deferral_salvaged_pending_reason_falls_back_when_parent_cache_evicted(caplog):
    caplog.set_level(logging.INFO)

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}, "soft_deferral": {"salvage_idempotency_max_items": 3}},
    )
    max_items = coordinator._soft_deferral_salvage_cache_max()
    assert max_items == 3

    parent_pending_id = "p" * 32
    near_miss = "strategy.mean_reversion.near_miss"

    coordinator._remember_soft_deferral_pending_reason(parent_pending_id, near_miss)
    for i in range(max_items):
        coordinator._remember_soft_deferral_pending_reason(f"{i:032d}", f"other_reason_{i}")

    cache = getattr(coordinator, "_soft_deferral_pending_reason_by_parent_id", None)
    assert isinstance(cache, OrderedDict)
    assert parent_pending_id not in cache, "Expected parent pending_reason_code to be evicted from LRU"

    caplog.clear()
    coordinator._emit_soft_deferral_salvaged(
        parent_pending_id=parent_pending_id,
        signal_id="child_sig_1",
        final_status="queued",
        signal_payload={
            "strategy_name": "mean_reversion",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "timeframe": "5m",
            "pending_reason_code": near_miss,
            "meta": {
                "parent_pending_id": parent_pending_id,
                "pending_reason_code": near_miss,
            },
        },
    )

    salvage_events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_salvaged ")]
    assert salvage_events, "Expected soft_deferral_salvaged"
    _, json_blob = salvage_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("parent_pending_id") == parent_pending_id
    assert data.get("pending_reason_code") == near_miss


@pytest.mark.asyncio
@pytest.mark.integration
async def test_soft_deferral_recheck_dropped_deduped_outcomes_emitted(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.config = {"incubator": {"max_rechecks_per_loop": 1}}

    async def fake_dispatch(_symbol: str, _strategy: str, *, parent_pending_id: str | None = None, **_kwargs):
        return True

    monkeypatch.setattr(prod, "dispatch_strategy", fake_dispatch)

    pending_id = "p" * 32
    base = {
        "event": "strategy_recheck_request",
        "strategy": "mean_reversion",
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "timeframe": "5m",
        "parent_pending_id": pending_id,
        "pending_reason_code": "strategy.mean_reversion.near_miss",
        "reason_code": "strategy.mean_reversion.near_miss",
    }

    coordinator.strategy_recheck_queue.put_nowait({**base, "ts_ms": 1})
    coordinator.strategy_recheck_queue.put_nowait({**base, "ts_ms": 2})
    coordinator.strategy_recheck_queue.put_nowait({**base, "ts_ms": 3, "side": "SELL"})

    caplog.clear()
    await prod._drain_strategy_recheck_requests()

    events = [r.message for r in caplog.records if str(r.message).startswith("soft_deferral_recheck_outcome ")]
    assert events, "Expected soft_deferral_recheck_outcome telemetry"

    parsed = []
    for msg in events:
        _, json_blob = msg.split(" ", 1)
        data = json.loads(json_blob)
        if data.get("outcome") == "dropped_deduped":
            parsed.append(data)

    assert parsed, "Expected dropped_deduped outcomes"
    reasons = {d.get("dropped_reason") for d in parsed}
    assert "deduped_older_ts" in reasons
    assert "over_capacity" in reasons
