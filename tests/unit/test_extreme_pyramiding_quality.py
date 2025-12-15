import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.signal_intents import INTENT_SCALE_IN
from src.core.strategy_coordinator import StrategyCoordinator
from src.core.risk_manager import RiskManager


def _make_coordinator(config=None):
    pm = MagicMock()
    pm.cfg = config or {}
    pm.get_open_positions_for_symbol.return_value = []
    rm = MagicMock()
    cfg = config or {}
    coord = StrategyCoordinator(pm, rm, config=cfg)
    coord.ml_integration = None
    return coord


def _make_portfolio_manager(positions, cfg=None):
    pm = MagicMock()
    pm._positions = positions or []
    pm.cfg = cfg or {}

    def count_open_positions(symbol=None):
        if symbol:
            return sum(1 for p in pm._positions if p.get("symbol") == symbol)
        return len(pm._positions)

    def get_open_positions_for_symbol(symbol):
        return [p for p in pm._positions if p.get("symbol") == symbol]

    pm.count_open_positions.side_effect = count_open_positions
    pm.get_open_positions_for_symbol.side_effect = get_open_positions_for_symbol
    return pm


def _make_risk_manager(dynamic_cfg):
    rm = RiskManager.__new__(RiskManager)
    rm.config = {"concurrent_limits": {"dynamic_scaling": dynamic_cfg}}
    rm.concurrent_limits = SimpleNamespace(
        max_open_positions=5,
        max_positions_per_symbol=1,
        max_total_risk_pct=1.0,
        correlation_bucket_threshold=0.8,
    )
    return rm


@pytest.mark.asyncio
async def test_extreme_bypass_quality_not_overwritten_by_dynamic_rr():
    coord = _make_coordinator(
        config={
            "signals": {"signal_scoring": {"extreme_min_quality": 0.6}},
            "volume_analyzer": {"enabled": False},
        }
    )
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "extreme_bypass": True,
        "volume_strength": 1.0,
        "momentum_strength": 0.8,
        "regime_confidence": 0.9,
        "intent": INTENT_SCALE_IN,
    }

    quality = coord._compute_signal_quality(signal)
    assert signal["quality_score"] >= 0.6
    enriched = await coord._enrich_signal_for_dynamic_rr(signal)
    assert enriched["quality_score"] == pytest.approx(quality["value"])


def test_extreme_bypass_scale_in_allows_when_quality_high():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.8,
        "min_unrealized_pnl_pct": 0.005,
        "min_distance_pct": 0.005,
        "max_additional_positions": 2,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.01, "entry_time": 1}
    ]
    pm = _make_portfolio_manager(
        positions,
        cfg={
            "pyramiding": {
                "enabled": True,
                "min_scale_in_quality": 0.8,
                "min_scale_in_unrealized_pnl_pct": 0.005,
                "min_scale_in_distance_pct": 0.005,
                "max_layers_per_symbol": 2,
            }
        },
    )

    # Build an extreme-bypass quality profile manually
    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry": 101.0,
        "quality_score": 0.9,
        "extreme_bypass": True,
        "intent": INTENT_SCALE_IN,
    }
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert reason.startswith("OK")
