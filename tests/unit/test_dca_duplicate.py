import time

import pytest

from src.core.strategy_coordinator import StrategyCoordinator


class DummyRisk:
    def __init__(self, equity=1000.0):
        self.portfolio_value = equity


class DummyPortfolio:
    def __init__(self, cfg, equity=1000.0):
        self.cfg = cfg
        self.active_positions = {}
        self._equity = equity

    def get_current_equity(self):
        return self._equity

    def get_open_positions_for_symbol(self, symbol):
        return [dict(v, position_id=k) for k, v in self.active_positions.items() if v.get("symbol") == symbol]


def _make_coordinator(cfg=None):
    cfg = cfg or {}
    pm = DummyPortfolio(cfg)
    risk = DummyRisk()
    return StrategyCoordinator(portfolio_manager=pm, risk_manager=risk, config=cfg), pm


@pytest.fixture
def base_signal():
    return {
        "symbol": "BTC/USDT:USDT",
        "intent": "scale_in",
        "scale_profile": "dca",
        "side": "long",
        "entry": 98,
        "dca_metadata": {"profile": "dca", "layer_index": 1, "anchor_price": 100, "price_drop_pct": 0.02},
    }


def test_dca_duplicate_same_layer_rejected(base_signal):
    cfg = {"dca": {"enabled": True, "strategy": {"cooldown_seconds": 0}}}
    coord, pm = _make_coordinator(cfg)
    pm.active_positions["pos1"] = {
        "symbol": "BTC/USDT:USDT",
        "scale_profile": "dca",
        "dca_metadata": {"profile": "dca", "layer_index": 1},
    }

    ok, reason = coord.validate_duplicate(base_signal, "dca_watcher")
    assert not ok
    assert reason == "dca_layer_duplicate"


def test_dca_not_adverse_rejected(base_signal):
    cfg = {"dca": {"enabled": True, "strategy": {"cooldown_seconds": 0}}}
    coord, _ = _make_coordinator(cfg)
    signal = dict(base_signal)
    signal["entry"] = 101  # not adverse for long vs anchor 100
    ok, reason = coord.validate_duplicate(signal, "dca_watcher")
    assert not ok
    assert reason == "dca_not_adverse_enough"


def test_dca_cooldown_blocks_repeats(base_signal):
    cfg = {"dca": {"enabled": True, "strategy": {"cooldown_seconds": 5}}}
    coord, _ = _make_coordinator(cfg)
    coord._dca_last_signal_time[base_signal["symbol"]] = time.time()
    ok, reason = coord.validate_duplicate(base_signal, "dca_watcher")
    assert not ok
    assert reason == "dca_cooldown_not_passed"


def test_dca_duplicate_accepts_when_valid(base_signal):
    cfg = {"dca": {"enabled": True, "strategy": {"cooldown_seconds": 0}}}
    coord, _ = _make_coordinator(cfg)
    ok, reason = coord.validate_duplicate(base_signal, "dca_watcher")
    assert ok
    assert reason == "OK"
