from unittest.mock import MagicMock

import pytest

from src.core.strategy_coordinator import StrategyCoordinator
from src.core.signal_intents import (
    INTENT_ENTRY,
    INTENT_SCALE_IN,
    INTENT_REVERSE,
    INTENT_FORCE_SWAP,
)


def _build_coordinator(cfg_override=None):
    portfolio_manager = MagicMock()
    risk_manager = MagicMock()
    base_cfg = {
        "signals": {
            "duplicate_prevention": {
                "enabled": True,
                "cooldown_seconds": 60,
                "min_price_change_pct": 0.0005,
                "price_delta_bypass_enabled": True,
                "price_delta_bypass_threshold": 0.0015,
                "scale_in_min_price_change_pct": 0.0005,
                "scale_in_cooldown_seconds": 60,
            }
        }
    }
    if cfg_override:
        base_cfg["signals"]["duplicate_prevention"].update(cfg_override)
    portfolio_manager.cfg = base_cfg
    return StrategyCoordinator(portfolio_manager, risk_manager)


def test_maintenance_intents_bypass_duplicate():
    coordinator = _build_coordinator()
    signal = {"symbol": "BTC/USDT", "entry": 50000, "intent": INTENT_REVERSE}
    ok, reason = coordinator.validate_duplicate(signal, "test_strategy")
    assert ok is True
    assert "maintenance" in reason

    signal["intent"] = INTENT_FORCE_SWAP
    ok, reason = coordinator.validate_duplicate(signal, "test_strategy")
    assert ok is True
    assert "maintenance" in reason


def test_scale_in_behaves_like_entry_conservatively():
    cfg_override = {
        "cooldown_seconds": 10,
        "price_delta_bypass_enabled": False,
    }
    coordinator = _build_coordinator(cfg_override)

    signal_entry = {"symbol": "BTC/USDT", "entry": 30000, "intent": INTENT_ENTRY}
    signal_scale = {"symbol": "ETH/USDT", "entry": 2000, "intent": INTENT_SCALE_IN}

    # First signals should pass
    assert coordinator.validate_duplicate(signal_entry, "strat")[0] is True
    assert coordinator.validate_duplicate(signal_scale, "strat")[0] is True

    # Immediate repeats should be rejected for both intents (same thresholds)
    assert coordinator.validate_duplicate(signal_entry, "strat")[0] is False
    assert coordinator.validate_duplicate(signal_scale, "strat")[0] is False


def test_bypass_threshold_used_instead_of_min_price_change():
    cfg_override = {
        "cooldown_seconds": 30,
        "min_price_change_pct": 0.0005,
        "price_delta_bypass_enabled": True,
        "price_delta_bypass_threshold": 0.02,  # 2%
    }
    coordinator = _build_coordinator(cfg_override)
    symbol = "SOL/USDT"

    # First signal seeds history
    first = {"symbol": symbol, "entry": 100}
    assert coordinator.validate_duplicate(first, "strat")[0] is True

    # Second signal within cooldown with 1% move (< bypass threshold but > min_price_change_pct) should be rejected
    second = {"symbol": symbol, "entry": 101}
    ok, reason = coordinator.validate_duplicate(second, "strat")
    assert ok is False
    assert "cooldown" in reason.lower()

    # Third signal within cooldown with 3% move (>= bypass threshold) should be accepted
    third = {"symbol": symbol, "entry": 103}
    ok, reason = coordinator.validate_duplicate(third, "strat")
    assert ok is True
    assert "bypass" in reason.lower()
