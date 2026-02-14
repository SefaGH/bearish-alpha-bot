import json
import logging

import pandas as pd
import pytest

from src.core.production_coordinator import ProductionCoordinator
from src.core.strategy_coordinator import StrategyCoordinator


class DummyPortfolioManager:
    cfg = {}
    performance_monitor = None
    exchange_clients = {}

    def get_current_equity(self):
        return 1000.0

    def get_strategy_allocation(self, strategy_name: str):
        del strategy_name
        return 1.0

    def get_open_positions_for_symbol(self, symbol: str):
        del symbol
        return []


class _CaptureSignalStrategy:
    strategy_name = "adaptive_ob"

    def __init__(self):
        self.last_kwargs = None

    async def signal(self, market_data=None, **kwargs):
        data = dict(kwargs)
        data["market_data"] = market_data
        self.last_kwargs = data
        return {
            "symbol": kwargs.get("symbol"),
            "side": "buy",
            "entry": 100.0,
            "stop": 99.0,
            "target": 101.0,
        }


class _PortfolioDispatchStub:
    def __init__(self, strategy):
        self.strategies = {"adaptive_ob": strategy}
        self.strategy_metadata = {"adaptive_ob": {"active": True}}


class _PipelineStub:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def get_latest_ohlcv(self, _symbol: str, _timeframe: str, limit=None, include_forming=True):
        del limit, include_forming
        return self._df


def _build_ohlcv_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"])
    return pd.DataFrame(
        {
            "open": [100.0, 100.2],
            "high": [100.5, 100.7],
            "low": [99.5, 99.8],
            "close": [100.1, 100.3],
            "volume": [10.0, 12.0],
        },
        index=idx,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dispatch_strategy_propagates_level_snapshot_to_signal_and_market_data():
    strategy = _CaptureSignalStrategy()
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.portfolio_manager = _PortfolioDispatchStub(strategy)
    prod.market_data_pipeline = _PipelineStub(_build_ohlcv_df())
    prod.strategy_coordinator = object()
    prod.ml_integration = None
    prod.market_regime_analyzer = None
    prod.strategies = {}
    prod.config = {
        "strategies": {
            "level_zone_router": {
                "enabled": True,
                "source": {"timeframes": ["15m"]},
                "zones": {"no_trade_new_entry": True, "near_level_bps": 50.0},
            }
        }
    }

    # Deterministic snapshot for integration assertion.
    prod._build_symbol_level_zone_snapshot = lambda **kwargs: {
        "symbol": kwargs.get("symbol"),
        "zone": "IN_RANGE",
        "price": 100.3,
        "ts_ms": 1735689600000,
        "primary_timeframe": "15m",
    }

    captured = {}

    async def _capture_route(strategy_name: str, payload: dict):
        captured["strategy_name"] = strategy_name
        captured["payload"] = payload
        return {"status": "accepted", "signal_id": "sig_level_1"}

    prod._route_strategy_output = _capture_route

    detail = await prod.dispatch_strategy("BTC/USDT:USDT", "adaptive_ob", return_detail=True)
    assert isinstance(detail, dict)
    assert detail.get("dispatched") is True

    assert isinstance(strategy.last_kwargs, dict)
    assert strategy.last_kwargs.get("level_zone_snapshot", {}).get("zone") == "IN_RANGE"
    assert strategy.last_kwargs.get("market_data", {}).get("level_zone_snapshot", {}).get("zone") == "IN_RANGE"

    routed_signal = captured.get("payload")
    assert isinstance(routed_signal, dict)
    assert routed_signal.get("level_zone_snapshot", {}).get("zone") == "IN_RANGE"
    assert routed_signal.get("meta", {}).get("level_zone_snapshot", {}).get("zone") == "IN_RANGE"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_strategy_recheck_request_emits_level_reason_telemetry(caplog):
    caplog.set_level(logging.INFO)
    coordinator = StrategyCoordinator(
        DummyPortfolioManager(),
        risk_manager=object(),
        config={"risk": {"queue": {"ttl_seconds": 5}}},
    )

    event = {
        "event_type": "soft_deferral_event",
        "strategy": "adaptive_ob",
        "symbol": "BTC/USDT:USDT",
        "side": "BUY",
        "timeframe": "15m",
        "setup_anchor_ts_ms": int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000),
        "reason_code": "level_router.at_level",
        "reason": "level_router.at_level",
        "refresh_policy": "STRATEGY_RECHECK",
        "condition_data": {
            "near": "level",
            "trigger_price": 100.3,
            "eps_bps": 50,
        },
    }

    result = await coordinator.handle_soft_deferral(event)
    assert result.get("status") == "incubated"

    dedupe_key = result.get("dedupe_key")
    assert dedupe_key in coordinator._incubator_items
    coordinator._incubator_items[dedupe_key]["next_check_at_ms"] = 0

    caplog.clear()
    processed = await coordinator.incubator_tick(max_items=10, time_budget_ms=1000)
    assert processed >= 1
    assert not coordinator.strategy_recheck_queue.empty()

    req = coordinator.strategy_recheck_queue.get_nowait()
    assert req.get("event") == "strategy_recheck_request"
    assert req.get("pending_reason_code") == "level_router.at_level"
    assert req.get("reason_code") == "level_router.at_level"
    assert req.get("refresh_policy") == "STRATEGY_RECHECK"

    telemetry = [r.message for r in caplog.records if str(r.message).startswith("strategy_recheck_request ")]
    assert telemetry, "Expected strategy_recheck_request telemetry"
    _, json_blob = telemetry[-1].split(" ", 1)
    data = json.loads(json_blob)
    assert data.get("pending_reason_code") == "level_router.at_level"
    assert data.get("reason_code") == "level_router.at_level"
