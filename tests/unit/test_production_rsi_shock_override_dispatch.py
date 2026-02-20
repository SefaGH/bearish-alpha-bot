import pandas as pd
import pytest

from src.core.production_coordinator import ProductionCoordinator


class _AdaptiveStrStrategy:
    strategy_name = "adaptive_str"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_kwargs = None

    async def signal(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return {
            "event_type": "strategy_recheck_decision",
            "decision_meta": {"rearm_fast_watch": False},
        }


class _PortfolioStub:
    def __init__(self, strategy: _AdaptiveStrStrategy) -> None:
        self.strategies = {"adaptive_str": strategy}
        self.strategy_metadata = {"adaptive_str": {"active": True}}


class _PipelineStub:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    async def get_latest_ohlcv(self, _symbol: str, _timeframe: str, limit=None, include_forming=True):
        del limit, include_forming
        return self._df


class _RegimeAnalyzerStub:
    def analyze_market_regime(self, df_30m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame):
        del df_30m, df_1h, df_4h
        return {
            "trend": "bearish",
            "momentum": "strong",
            "volatility": "normal",
            "trend_strength": 33.0,
        }


class _ShockSource:
    strategy_name = "adaptive_ob"

    def update_dyn_gate(self, symbol: str, df_30m: pd.DataFrame, now_ms: int, slow_fallback_reason=None):
        del symbol, df_30m, now_ms, slow_fallback_reason

    def get_dyn_gate_snapshot(self, symbol: str):
        del symbol
        return {"state": "ARMED", "shock_score": 0.87}


def _build_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T01:00:00Z",
        ]
    )
    close = [100.0, 99.8, 99.7]
    return pd.DataFrame(
        {
            "open": close,
            "high": [100.3, 100.1, 100.0],
            "low": [99.7, 99.6, 99.5],
            "close": close,
            "volume": [10.0, 11.0, 12.0],
        },
        index=idx,
    )


def _make_prod(strategy: _AdaptiveStrStrategy, *, shock_override_enabled: bool) -> ProductionCoordinator:
    prod = ProductionCoordinator.__new__(ProductionCoordinator)
    prod.portfolio_manager = _PortfolioStub(strategy)
    prod.market_data_pipeline = _PipelineStub(_build_df())
    prod.strategy_coordinator = object()
    prod.ml_integration = None
    prod.market_regime_analyzer = _RegimeAnalyzerStub()
    prod.strategies = {"adaptive_ob": _ShockSource()}
    prod.config = {
        "strategies": {
            "rsi_zone_router": {
                "enabled": True,
                "source": {"mode": "consensus"},
                "transition": {
                    "no_trade_new_entry": True,
                    "shock_override": {
                        "enabled": bool(shock_override_enabled),
                        "mode": "enforce",
                        "canary_symbols": ["*"],
                        "allow_strategies": ["adaptive_str"],
                        "state": "ARMED",
                        "min_score": 0.60,
                        "min_adx": 25.0,
                    },
                },
            },
            "level_zone_router": {"enabled": False},
        }
    }
    return prod


@pytest.mark.asyncio
async def test_dispatch_strategy_rsi_shock_override_enforce_allows_transition_entry():
    strategy = _AdaptiveStrStrategy()
    prod = _make_prod(strategy, shock_override_enabled=True)
    capture = {}

    def _fake_build_symbol_rsi_zone_snapshot(**kwargs):
        capture.update(kwargs)
        return {
            "symbol": kwargs.get("symbol"),
            "zone": "TRANSITION_HIGH",
            "rsi_slow": 62.0,
            "ob_threshold": 35.0,
            "str_threshold": 65.0,
            "shock_state": kwargs.get("shock_state"),
            "shock_score": kwargs.get("shock_score"),
            "regime_adx": ((kwargs.get("regime_data") or {}).get("trend_strength")),
        }

    prod._build_symbol_rsi_zone_snapshot = _fake_build_symbol_rsi_zone_snapshot

    detail = await prod.dispatch_strategy(
        "BTC/USDT:USDT",
        "adaptive_str",
        side="sell",
        return_detail=True,
    )

    assert isinstance(detail, dict)
    assert detail.get("final_reason") == "recheck_hold"
    assert strategy.call_count == 1
    assert capture.get("shock_state") == "ARMED"
    assert capture.get("shock_score") == pytest.approx(0.87)
    assert strategy.last_kwargs is not None
    assert strategy.last_kwargs.get("rsi_zone_snapshot", {}).get("shock_state") == "ARMED"
    assert strategy.last_kwargs.get("rsi_zone_snapshot", {}).get("shock_score") == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_dispatch_strategy_rsi_shock_override_disabled_stays_transition_blocked():
    strategy = _AdaptiveStrStrategy()
    prod = _make_prod(strategy, shock_override_enabled=False)

    def _fake_build_symbol_rsi_zone_snapshot(**kwargs):
        del kwargs
        return {
            "symbol": "BTC/USDT:USDT",
            "zone": "TRANSITION_HIGH",
            "rsi_slow": 62.0,
            "ob_threshold": 35.0,
            "str_threshold": 65.0,
            "shock_state": "ARMED",
            "shock_score": 0.90,
            "regime_adx": 30.0,
        }

    prod._build_symbol_rsi_zone_snapshot = _fake_build_symbol_rsi_zone_snapshot

    detail = await prod.dispatch_strategy(
        "BTC/USDT:USDT",
        "adaptive_str",
        side="sell",
        return_detail=True,
    )

    assert isinstance(detail, dict)
    assert detail.get("dispatched") is False
    assert detail.get("final_reason") == "rsi_router.transition_no_trade"
    assert strategy.call_count == 0
