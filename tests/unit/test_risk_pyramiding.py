from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.risk_manager import RiskManager
from src.core.position_manager import PositionManagerPnlProvider
from src.core.signal_intents import INTENT_SCALE_IN, INTENT_ENTRY


class _StubPnlProvider:
    def __init__(self, positions):
        self.positions = positions

    def get_positions_for_symbol(self, symbol, strategy_name=None, side=None):
        results = []
        for p in self.positions:
            if p.get("symbol") != symbol:
                continue
            snapshot = dict(p)
            if "unrealized_pnl_pct" not in snapshot and "pnl_pct" in snapshot:
                snapshot["unrealized_pnl_pct"] = snapshot.get("pnl_pct")
            results.append(snapshot)
        return results


class _DummyPositionManager:
    def __init__(self, positions):
        self.positions = positions


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
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

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
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

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
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

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
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

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
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

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


def test_scale_in_allowed_with_positive_pnl_from_provider():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.003,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [{"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": 0.01, "entry_time": 1}]
    pm = _make_portfolio_manager(
        positions,
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.003}},
    )
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert reason.startswith("OK")


def test_scale_in_rejected_with_negative_pnl_from_provider():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.001,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [{"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl_pct": -0.01, "entry_time": 1}]
    pm = _make_portfolio_manager(
        positions,
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.001}},
    )
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 99, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False
    assert reason == "scale_in_pnl_below_threshold"


def test_scale_in_rejected_when_pnl_data_unavailable():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.001,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [{"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "entry_time": 1}]  # missing PnL fields
    pm = _make_portfolio_manager(
        positions,
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.001}},
    )
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False
    assert reason == "scale_in_pnl_data_unavailable"


def test_scale_in_uses_pnl_pct_fallback():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.002,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    positions = [
        {"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "pnl_pct": 0.5, "entry_time": 1}
    ]  # pnl_pct only
    pm = _make_portfolio_manager(
        positions,
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.002}},
    )
    rm.set_pnl_provider(_StubPnlProvider(pm._positions))

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert reason.startswith("OK")


def test_scale_in_reads_pnl_pct_from_position_manager_provider():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.002,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    pm_positions = {
        "pos_btc": {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "amount": 1.0,
            "pnl_pct": 0.44,  # PositionManager stores percent-style pct for logging
            "unrealized_pnl": 0.44,
            "entry_time": 1,
        }
    }
    pm = _make_portfolio_manager(
        list(pm_positions.values()),
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.002}},
    )
    provider = PositionManagerPnlProvider(_DummyPositionManager(pm_positions))
    rm.set_pnl_provider(provider)

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is True
    assert reason.startswith("OK")


def test_scale_in_rejected_when_position_manager_provider_has_no_pnl():
    dynamic_cfg = {
        "enabled": True,
        "quality_threshold": 0.5,
        "min_unrealized_pnl_pct": 0.002,
        "min_distance_pct": 0.0,
        "max_additional_positions": 1,
    }
    rm = _make_risk_manager(dynamic_cfg)
    pm_positions = {
        "pos_btc": {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "amount": 1.0,
            "entry_time": 1,
        }
    }
    pm = _make_portfolio_manager(
        list(pm_positions.values()),
        cfg={"pyramiding": {"enabled": True, "min_scale_in_quality": 0.5, "min_scale_in_unrealized_pnl_pct": 0.002}},
    )
    provider = PositionManagerPnlProvider(_DummyPositionManager(pm_positions))
    rm.set_pnl_provider(provider)

    signal = {"symbol": "BTC/USDT", "side": "long", "entry": 101, "quality_score": 0.9, "intent": INTENT_SCALE_IN}
    allowed, reason = rm._check_concurrent_limits(signal, {}, pm)
    assert allowed is False
    assert reason == "scale_in_pnl_data_unavailable"
