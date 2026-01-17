import json
import logging

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


class DummyCacheOnlyPipeline:
    def __init__(self, prices):
        self._prices = list(prices)
        self.cache_calls = 0
        self.rest_calls = 0

    async def get_latest_price_cache_only(self, _symbol: str, timeframe: str = "1m", exchange=None):
        self.cache_calls += 1
        if self._prices:
            return self._prices.pop(0)
        return None

    async def get_latest_price(self, *args, **kwargs):
        self.rest_calls += 1
        raise AssertionError("REST fallback should not be called")


def _build_fast_watch_signal(
    anchor_ms: int,
    trigger_price: float,
    max_checks: int = 3,
    *,
    near: str | None = "lower",
    eps_bps: float = 0,
) -> dict:
    condition_data = {
        "trigger_price": trigger_price,
        "eps_bps": eps_bps,
        "trigger_kind": "geq",
        "watch_interval_ms": 1,
        "max_checks": max_checks,
        "ttl_ms": 10_000,
    }
    if near is not None:
        condition_data["near"] = near
    return {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "timeframe": "1m",
        "intent": "soft_deferral",
        "setup_anchor_ts_ms": anchor_ms,
        "timestamp": anchor_ms,
        "condition_data": condition_data,
    }


def _build_mr_frames(price: float, *, lower: float = 99.0, upper: float = 101.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "close": [price, price],
            "volume": [1.0, 1.0],
            "vwap": [100.0, 100.0],
            "vwap_lower": [lower, lower],
            "vwap_upper": [upper, upper],
            "vwap_std": [1.0, 1.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [price, price], "adx": [10.0, 10.0]}, index=idx)
    return df_vwap, df_sig


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_triggers_recheck_and_wakeup(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    wakeups = []

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=DummyCacheOnlyPipeline([99.0, 100.0]),
        config={"fast_watch": {"enabled": True, "interval_ms": 1, "time_budget_ms": 500, "max_items_per_tick": 10}},
        recheck_ready_callback=lambda: wakeups.append("wakeup"),
    )

    anchor_ms = coordinator._now_ms()
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=5, near="upper")
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    caplog.clear()
    await coordinator._fast_watch_tick()
    assert coordinator.strategy_recheck_queue.empty()

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    await coordinator._fast_watch_tick()
    assert not coordinator.strategy_recheck_queue.empty()
    recheck = coordinator.strategy_recheck_queue.get_nowait()
    assert recheck.get("event") == "strategy_recheck_request"
    assert wakeups, "Expected recheck wakeup callback to be invoked"
    assert dedupe_key in coordinator._incubator_items
    assert coordinator._incubator_items[dedupe_key].get("state") == "awaiting_recheck"

    outcome_events = [r.message for r in caplog.records if str(r.message).startswith("fast_watch_outcome ")]
    assert outcome_events, "Expected fast_watch_outcome telemetry"
    _, json_blob = outcome_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("outcome") == "triggered"
    assert data.get("symbol") == "BTC/USDT:USDT"
    assert data.get("near") == "upper"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_trigger_recheck_emits_mr_eval(caplog):
    caplog.set_level(logging.INFO)

    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=DummyCacheOnlyPipeline([99.0, 100.0]),
        config={"fast_watch": {"enabled": True, "interval_ms": 1, "time_budget_ms": 500, "max_items_per_tick": 10}},
    )

    anchor_ms = coordinator._now_ms()
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=5, near="upper")
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator._fast_watch_tick()
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator._fast_watch_tick()

    assert not coordinator.strategy_recheck_queue.empty()
    recheck = coordinator.strategy_recheck_queue.get_nowait()
    assert recheck.get("event") == "strategy_recheck_request"

    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"])
    df_vwap = pd.DataFrame(
        {
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "vwap": [100.0, 100.0],
            "vwap_lower": [99.0, 99.0],
            "vwap_upper": [101.0, 101.0],
        },
        index=idx,
    )
    df_sig = pd.DataFrame({"close": [100.0, 100.0], "adx": [10.0, 10.0]}, index=idx)

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
            "dynamic_controller": {"enabled": False},
        }
    )
    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}
    async def _route_override(_strategy, _signal):
        return {"signal_id": "test-signal"}
    prod._route_strategy_output = _route_override

    caplog.clear()
    dispatched = await prod._handle_strategy_recheck_request(recheck)
    assert dispatched is False

    eval_events = [r.message for r in caplog.records if str(r.message).startswith("mr_recheck_eval ")]
    assert eval_events, "Expected mr_recheck_eval telemetry for recheck"
    _, json_blob = eval_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("near") == "upper"
    assert data.get("px_source") == "market_price"
    assert data.get("trigger_price") is not None
    assert data.get("fast_watch_price") is not None
    assert data.get("gate_reasons")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_past_expiry_does_not_expire_immediately():
    pipeline = DummyCacheOnlyPipeline([99.0])
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=pipeline,
        config={"fast_watch": {"enabled": True, "interval_ms": 1, "time_budget_ms": 500, "max_items_per_tick": 10}},
    )

    anchor_ms = coordinator._now_ms() - 120_000
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=5, near="upper")
    signal["condition_data"]["ttl_ms"] = 30_000
    signal["condition_data"]["expires_at_ms"] = coordinator._now_ms() - 1_000

    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    await coordinator._fast_watch_tick()

    assert dedupe_key in coordinator._incubator_items
    item = coordinator._incubator_items[dedupe_key]
    now_ms = coordinator._now_ms()
    assert item.get("expires_at_ms") is not None
    assert item.get("expires_at_ms") > now_ms
    assert item.get("next_check_at_ms") > now_ms


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_cache_miss_never_calls_rest(caplog):
    caplog.set_level(logging.INFO)

    pipeline = DummyCacheOnlyPipeline([None])
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=pipeline,
        config={"fast_watch": {"enabled": True, "interval_ms": 1, "time_budget_ms": 500, "max_items_per_tick": 10}},
    )

    anchor_ms = coordinator._now_ms()
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=1, near=None)
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    caplog.clear()
    await coordinator._fast_watch_tick()
    assert pipeline.rest_calls == 0
    assert dedupe_key in coordinator._incubator_items

    outcome_events = [r.message for r in caplog.records if str(r.message).startswith("fast_watch_outcome ")]
    assert outcome_events, "Expected fast_watch_outcome telemetry"
    _, json_blob = outcome_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("outcome") == "cache_miss"
    assert data.get("cache_hit") is False
    assert data.get("near") == "unknown"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_rearm_and_finalize(caplog):
    caplog.set_level(logging.INFO)

    pipeline = DummyCacheOnlyPipeline([100.0, 100.0])
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=pipeline,
        config={
            "fast_watch": {
                "enabled": True,
                "interval_ms": 1,
                "time_budget_ms": 500,
                "max_items_per_tick": 10,
                "max_rearms": 1,
                "rearm_backoff_mult": 2.0,
                "rearm_max_interval_ms": 10,
            }
        },
    )

    anchor_ms = coordinator._now_ms()
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=5, near="upper", eps_bps=10)
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    item = coordinator._incubator_items[dedupe_key]
    original_expires_at = item.get("expires_at_ms")
    original_interval = item.get("fast_watch_interval_ms") or coordinator._fast_watch_interval_ms
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    await coordinator._fast_watch_tick()
    assert not coordinator.strategy_recheck_queue.empty()
    recheck = coordinator.strategy_recheck_queue.get_nowait()
    assert coordinator._incubator_items[dedupe_key].get("state") == "awaiting_recheck"

    df_vwap_hold, df_sig_hold = _build_mr_frames(100.0, lower=99.0, upper=101.0)
    df_vwap_signal, df_sig_signal = _build_mr_frames(98.5, lower=99.0, upper=101.0)

    class DummyPipeline:
        def __init__(self):
            self.mode = "hold"

        async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
            if self.mode == "signal":
                return df_vwap_signal if timeframe == "1m" else df_sig_signal
            return df_vwap_hold if timeframe == "1m" else df_sig_hold

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "dynamic_controller": {"enabled": False},
            "soft_deferral_threshold": 0.0,
            "rsi_rebound_guard": {"enabled": False},
        }
    )
    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}

    caplog.clear()
    dispatched = await prod._handle_strategy_recheck_request(recheck)
    assert dispatched is False
    item = coordinator._incubator_items[dedupe_key]
    assert item.get("state") == "watching"
    assert item.get("rearm_count") == 1
    assert item.get("expires_at_ms") == original_expires_at
    expected_interval = int(original_interval * coordinator._fast_watch_rearm_backoff_mult)
    expected_interval = max(1, expected_interval)
    expected_interval = min(expected_interval, coordinator._fast_watch_rearm_max_interval_ms)
    assert item.get("fast_watch_interval_ms") == expected_interval
    assert any(str(r.message).startswith("soft_deferral_rearm ") for r in caplog.records)

    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator._fast_watch_tick()
    recheck = coordinator.strategy_recheck_queue.get_nowait()

    prod.market_data_pipeline.mode = "signal"
    caplog.clear()
    dispatched = await prod._handle_strategy_recheck_request(recheck)
    assert dispatched is True
    assert dedupe_key not in coordinator._incubator_items
    drop_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_drop ")]
    assert drop_events, "Expected final waiting_room_drop after signal"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fast_watch_rearm_limit_finalize(caplog):
    caplog.set_level(logging.INFO)

    pipeline = DummyCacheOnlyPipeline([100.0, 100.0])
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        market_data_pipeline=pipeline,
        config={
            "fast_watch": {
                "enabled": True,
                "interval_ms": 1,
                "time_budget_ms": 500,
                "max_items_per_tick": 10,
                "max_rearms": 1,
                "rearm_backoff_mult": 2.0,
                "rearm_max_interval_ms": 10,
            }
        },
    )

    anchor_ms = coordinator._now_ms()
    signal = _build_fast_watch_signal(anchor_ms, trigger_price=100.0, max_checks=5, near="upper", eps_bps=10)
    result = await coordinator.incubate_signal(
        strategy_name="mean_reversion",
        signal=signal,
        reason_code="strategy.fast_watch",
        refresh_policy="FAST_PRICE_WATCH",
        stage="soft_deferral",
    )
    dedupe_key = result.get("dedupe_key")
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator._fast_watch_tick()
    recheck = coordinator.strategy_recheck_queue.get_nowait()

    df_vwap_hold, df_sig_hold = _build_mr_frames(100.0, lower=99.0, upper=101.0)

    class DummyPipeline:
        async def get_latest_ohlcv(self, _symbol: str, timeframe: str, limit=None, include_forming=True):
            return df_vwap_hold if timeframe == "1m" else df_sig_hold

    strategy = VWAPMeanReversion(
        {
            "timeframe": "1m",
            "signal_timeframe": "5m",
            "min_rows": 2,
            "min_signal_rows": 2,
            "dynamic_controller": {"enabled": False},
            "soft_deferral_threshold": 0.0,
            "rsi_rebound_guard": {"enabled": False},
        }
    )
    portfolio = DummyPortfolioManager()
    portfolio.strategies = {"mean_reversion": strategy}

    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.strategy_coordinator = coordinator
    prod.portfolio_manager = portfolio
    prod.market_data_pipeline = DummyPipeline()
    prod.ml_integration = None
    prod.config = {}

    await prod._handle_strategy_recheck_request(recheck)
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0
    await coordinator._fast_watch_tick()
    recheck = coordinator.strategy_recheck_queue.get_nowait()

    caplog.clear()
    await prod._handle_strategy_recheck_request(recheck)
    assert dedupe_key not in coordinator._incubator_items
    drop_events = [r.message for r in caplog.records if str(r.message).startswith("waiting_room_drop ")]
    assert drop_events, "Expected waiting_room_drop on max_rearms"
    _, json_blob = drop_events[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("drop_reason") == "max_rearms"
