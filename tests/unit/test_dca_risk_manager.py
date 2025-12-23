import time

import pytest

from src.config.risk_config import RiskConfiguration
from src.core.risk_manager import RiskManager


class DummyPortfolio:
    def __init__(self, cfg, equity=1000.0):
        self.cfg = cfg
        self.active_positions = {}
        self._equity = equity

    def count_open_positions(self, symbol=None):
        if symbol:
            return sum(1 for p in self.active_positions.values() if p.get("symbol") == symbol)
        return len(self.active_positions)

    def get_open_positions(self):
        return self.active_positions

    def get_open_positions_for_symbol(self, symbol):
        return [dict(v, position_id=k) for k, v in self.active_positions.items() if v.get("symbol") == symbol]

    def get_total_equity(self):
        return self._equity

    def get_total_exposure(self):
        return 0.0


def _make_risk_manager(equity=1000.0):
    risk_cfg = RiskConfiguration(custom_limits={"equity_usd": equity})
    return RiskManager(portfolio_value=equity, risk_config=risk_cfg)


@pytest.fixture
def base_signal():
    return {
        "symbol": "BTC/USDT:USDT",
        "intent": "scale_in",
        "scale_profile": "dca",
        "dca_metadata": {"profile": "dca", "layer_index": 1, "price_drop_pct": 0.02, "anchor_price": 100},
        "side": "long",
        "entry": 98,
    }


def test_dca_disabled_rejected(base_signal):
    cfg = {"dca": {"enabled": False}}
    pm = DummyPortfolio(cfg)
    rm = _make_risk_manager()

    ok, reason = rm._check_concurrent_limits(base_signal, {"portfolio_heat": 0.0}, pm)
    assert not ok
    assert "dca_not_enabled" in reason


def test_dca_mutual_exclusive_with_trend(base_signal):
    cfg = {
        "dca": {
            "enabled": True,
            "risk_limits": {"allow_concurrent_with_trend": False},
            "strategy": {"max_layers": 3},
        }
    }
    pm = DummyPortfolio(cfg)
    pm.active_positions["pos1"] = {"symbol": "BTC/USDT:USDT", "entry_price": 100, "amount": 1, "scale_profile": "trend"}
    rm = _make_risk_manager()

    ok, reason = rm._check_concurrent_limits(base_signal, {"portfolio_heat": 0.0}, pm)
    assert not ok
    assert "dca_trend_mutually_exclusive" in reason


def test_dca_layer_cap_enforced(base_signal):
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {"max_layers": 2, "cooldown_seconds": 0},
            "risk_limits": {"allow_concurrent_with_trend": True},
        }
    }
    pm = DummyPortfolio(cfg)
    # existing DCA layer already present
    pm.active_positions["pos1"] = {
        "symbol": "BTC/USDT:USDT",
        "entry_price": 100,
        "amount": 1,
        "scale_profile": "dca",
        "dca_metadata": {"profile": "dca", "layer_index": 1},
    }
    rm = _make_risk_manager()

    ok, reason = rm._check_concurrent_limits(base_signal, {"portfolio_heat": 0.0}, pm)
    assert not ok
    assert "dca_max_layers_reached" in reason


def test_dca_portfolio_heat_cap(base_signal):
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {"max_layers": 4},
            "risk_limits": {"max_dca_portfolio_pct": 0.1, "allow_concurrent_with_trend": True},
        }
    }
    pm = DummyPortfolio(cfg, equity=1000.0)
    pm.active_positions["pos1"] = {
        "symbol": "BTC/USDT:USDT",
        "entry_price": 100,
        "amount": 1.2,  # notional 120
        "scale_profile": "dca",
        "dca_metadata": {"profile": "dca", "layer_index": 1},
    }
    rm = _make_risk_manager()

    ok, reason = rm._check_concurrent_limits(base_signal, {"portfolio_heat": 0.0}, pm)
    assert not ok
    assert "dca_portfolio_heat_limit" in reason


def test_dca_panic_cutoff(base_signal):
    cfg = {
        "dca": {
            "enabled": True,
            "strategy": {"max_layers": 4},
            "risk_limits": {"panic_cutoff_pct": 0.08, "allow_concurrent_with_trend": True},
        }
    }
    pm = DummyPortfolio(cfg)
    rm = _make_risk_manager()

    signal = dict(base_signal)
    signal["dca_metadata"] = dict(base_signal["dca_metadata"], price_drop_pct=0.09)

    ok, reason = rm._check_concurrent_limits(signal, {"portfolio_heat": 0.0}, pm)
    assert not ok
    assert "dca_panic_cutoff_triggered" in reason
