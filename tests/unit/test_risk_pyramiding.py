from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.risk_manager import RiskManager
from src.core.signal_intents import INTENT_SCALE_IN, INTENT_ENTRY


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


def test_scale_in_rejected_when_pyramiding_disabled_and_scaling_off():
    dynamic_cfg = {"enabled": False, "max_additional_positions": 0}
    rm = _make_risk_manager(dynamic_cfg)
    positions = [{"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.01}]
    pm = _make_portfolio_manager(positions, cfg={"pyramiding": {"enabled": False}})

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False
    assert "Max positions" in reason or "max positions" in reason.lower()


def test_scale_in_allowed_when_pyramiding_enabled_and_thresholds_met():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.8,
        "min_unrealized_pnl_pct": 0.001,
        "min_distance_pct": 0.001,
        "max_additional_positions": 2,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [
        {"symbol": "ETH/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.01, "entry_time": 1}
    ]
    pm = _make_portfolio_manager(
        positions,
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.8, "min_scale_in_unrealized_pnl_pct": 0.001}},
    )

    signal = {"symbol": "ETH/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert "OK" in reason or "Allowed" in reason


def test_scale_in_rejected_when_quality_below_threshold():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.8,
        "min_unrealized_pnl_pct": 0.0,
        "min_distance_pct": 0.0,
        "max_additional_positions": 2,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [
        {"symbol": "ETH/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.02, "entry_time": 1}
    ]
    pm = _make_portfolio_manager(positions, cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.85}})

    signal = {"symbol": "ETH/USDT", "side": "long", "entry": 110, "quality_score": 0.8, "intent": INTENT_SCALE_IN}
    allowed, _ = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False


def test_scale_in_rejected_when_max_layers_reached():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.1,
        "min_unrealized_pnl_pct": 0.0,
        "min_distance_pct": 0.0,
        "max_additional_positions": 5,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [
        {"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.02, "entry_time": 1},
        {"symbol": "BTC/USDT", "side": "long", "entry_price": 105, "unrealized_pnl_pct": 0.01, "entry_time": 2},
    ]
    pm = _make_portfolio_manager(positions, cfg={"pyramiding": {"enabled": True, "max_layers_per_symbol": 2}})

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 110, "quality_score": 1.0, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False
    assert "max layers" in reason.lower() or "pyramiding" in reason.lower() or "max positions" in reason.lower()


def test_scale_in_allowed_when_quality_pnl_distance_meet_thresholds():
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

    signal = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry": 101.0,
        "quality_score": 0.9,
        "intent": INTENT_SCALE_IN,
    }
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert reason.startswith("OK")
