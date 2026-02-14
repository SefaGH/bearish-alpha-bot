import json

import pandas as pd
import pytest

from src.core.production_coordinator import ProductionCoordinator


class _GateStrategy:
    strategy_name = "adaptive_ob"

    def __init__(self) -> None:
        self.call_count = 0

    async def signal(self, **kwargs):
        self.call_count += 1
        return {"symbol": kwargs.get("symbol"), "side": "buy"}


class _PortfolioStub:
    def __init__(self, strategy):
        self.strategies = {"adaptive_ob": strategy}
        self.strategy_metadata = {"adaptive_ob": {"active": True}}


class _PipelineStub:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def get_latest_ohlcv(self, _symbol: str, _timeframe: str, limit=None, include_forming=True):
        del limit, include_forming
        return self._df


def _build_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"])
    close = [100.0, 100.2]
    return pd.DataFrame(
        {
            "open": close,
            "high": [100.5, 100.7],
            "low": [99.5, 99.7],
            "close": close,
            "volume": [10.0, 11.0],
        },
        index=idx,
    )


def _make_prod(strategy: _GateStrategy, df: pd.DataFrame) -> ProductionCoordinator:
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.portfolio_manager = _PortfolioStub(strategy)
    prod.market_data_pipeline = _PipelineStub(df)
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
                "soft_deferral": {"enabled": True, "mode": "fast_watch_then_recheck"},
            }
        }
    }
    return prod


def test_build_level_router_soft_deferral_event_for_adaptive_ob():
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    strategy = _GateStrategy()
    router_cfg = {
        "soft_deferral": {"enabled": True, "mode": "fast_watch_then_recheck"},
        "zones": {"near_level_bps": 50.0},
    }
    snapshot = {
        "symbol": "BTC/USDT:USDT",
        "zone": "AT_LEVEL",
        "price": 100.0,
        "ts_ms": 1735689600000,
        "primary_timeframe": "15m",
    }

    event = prod._build_level_router_soft_deferral_event(
        strategy_name="adaptive_ob",
        strategy_instance=strategy,
        symbol="BTC/USDT:USDT",
        reason_code="level_router.at_level",
        level_zone_snapshot=snapshot,
        level_router_cfg=router_cfg,
    )

    assert isinstance(event, dict)
    assert event.get("event_type") == "soft_deferral_event"
    assert event.get("side") == "buy"
    assert event.get("refresh_policy") == "FAST_PRICE_WATCH"
    assert event.get("reason_code") == "level_router.at_level"
    assert event.get("condition_data", {}).get("trigger_price") == pytest.approx(100.0)


def test_emit_level_router_decision_logs_structured_payload(caplog):
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    snapshot = {
        "symbol": "BTC/USDT:USDT",
        "zone": "AT_LEVEL",
        "mode": "consensus",
        "primary_timeframe": "15m",
    }
    router_cfg = {"rollout": {"mode": "observe"}}

    with caplog.at_level("INFO"):
        prod._emit_level_router_decision(
            scope="main_loop",
            symbol="BTC/USDT:USDT",
            strategy_name="adaptive_ob",
            side=None,
            allowed=True,
            reason_code="level_router.observe_would_block",
            level_zone_snapshot=snapshot,
            level_router_cfg=router_cfg,
        )

    events = [r.message for r in caplog.records if str(r.message).startswith("level_router_decision ")]
    assert events, "Expected level_router_decision telemetry"
    _, blob = events[-1].split(" ", 1)
    data = json.loads(blob)
    assert data.get("event") == "level_router_decision"
    assert data.get("scope") == "main_loop"
    assert data.get("symbol") == "BTC/USDT:USDT"
    assert data.get("strategy") == "adaptive_ob"
    assert data.get("allowed") is True
    assert data.get("reason_code") == "level_router.observe_would_block"
    assert data.get("zone") == "AT_LEVEL"
    assert data.get("rollout_mode") == "observe"


@pytest.mark.asyncio
async def test_dispatch_strategy_level_router_at_level_rearms_fast_watch_on_recheck():
    strategy = _GateStrategy()
    prod = _make_prod(strategy, _build_df())
    prod._build_symbol_level_zone_snapshot = lambda **kwargs: {
        "symbol": kwargs.get("symbol"),
        "zone": "AT_LEVEL",
        "price": 100.0,
        "ts_ms": 1735689600000,
        "primary_timeframe": "15m",
    }

    detail = await prod.dispatch_strategy(
        "BTC/USDT:USDT",
        "adaptive_ob",
        parent_pending_id="pending-parent-1",
        pending_id="pending-1",
        side="buy",
        timeframe="15m",
        refresh_policy="FAST_PRICE_WATCH",
        pending_reason_code="level_router.at_level",
        return_detail=True,
    )

    assert isinstance(detail, dict)
    assert detail.get("dispatched") is False
    assert detail.get("rearm_fast_watch") is True
    assert detail.get("final_reason") == "level_router.breakout_unconfirmed"
    assert detail.get("decision_meta", {}).get("router_reason") == "level_router.at_level"
    assert strategy.call_count == 0


@pytest.mark.asyncio
async def test_dispatch_strategy_level_router_zone_mismatch_marks_recheck_cancelled():
    strategy = _GateStrategy()
    prod = _make_prod(strategy, _build_df())
    prod._build_symbol_level_zone_snapshot = lambda **kwargs: {
        "symbol": kwargs.get("symbol"),
        "zone": "BREAKOUT_DOWN_CONFIRMED",
        "price": 100.0,
        "ts_ms": 1735689600000,
        "primary_timeframe": "15m",
    }

    detail = await prod.dispatch_strategy(
        "BTC/USDT:USDT",
        "adaptive_ob",
        parent_pending_id="pending-parent-1",
        pending_id="pending-1",
        side="buy",
        timeframe="15m",
        refresh_policy="FAST_PRICE_WATCH",
        pending_reason_code="level_router.at_level",
        return_detail=True,
    )

    assert isinstance(detail, dict)
    assert detail.get("dispatched") is False
    assert detail.get("rearm_fast_watch") is False
    assert detail.get("final_reason") == "level_router.recheck_cancelled"
    assert detail.get("decision_meta", {}).get("router_reason") == "level_router.zone_mismatch"
    assert strategy.call_count == 0
