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


def test_better_price_bypasses_cooldown_for_long_with_noise_filter():
    cfg_override = {
        "cooldown_seconds": 60,
        "price_delta_bypass_enabled": True,
        "price_delta_bypass_threshold": 0.02,  # keep high so only better-price can accept
    }
    coordinator = _build_coordinator(cfg_override)
    symbol = "BTC/USDT"

    first = {"symbol": symbol, "entry": 100.0, "side": "long"}
    assert coordinator.validate_duplicate(first, "strat")[0] is True

    # Within cooldown, microscopic improvement should NOT bypass (noise filter).
    micro = {"symbol": symbol, "entry": 99.995, "side": "long"}
    ok, reason = coordinator.validate_duplicate(micro, "strat")
    assert ok is False
    assert "price change" in reason.lower()

    # Within cooldown, meaningful improvement should bypass.
    second = {"symbol": symbol, "entry": 99.98, "side": "long"}
    ok, reason = coordinator.validate_duplicate(second, "strat")
    assert ok is True
    assert "better price" in reason.lower()


def test_better_price_bypasses_cooldown_for_short_with_noise_filter():
    cfg_override = {
        "cooldown_seconds": 60,
        "price_delta_bypass_enabled": True,
        "price_delta_bypass_threshold": 0.02,  # keep high so only better-price can accept
    }
    coordinator = _build_coordinator(cfg_override)
    symbol = "ETH/USDT"

    first = {"symbol": symbol, "entry": 100.0, "side": "short"}
    assert coordinator.validate_duplicate(first, "strat")[0] is True

    # Within cooldown, microscopic improvement should NOT bypass (noise filter).
    micro = {"symbol": symbol, "entry": 100.005, "side": "short"}
    ok, reason = coordinator.validate_duplicate(micro, "strat")
    assert ok is False
    assert "price change" in reason.lower()

    # Within cooldown, meaningful improvement should bypass.
    second = {"symbol": symbol, "entry": 100.02, "side": "short"}
    ok, reason = coordinator.validate_duplicate(second, "strat")
    assert ok is True
    assert "better price" in reason.lower()


def test_rsi_session_anchor_used_for_price_delta_reference():
    cfg_override = {
        "cooldown_seconds": 60,
        "price_delta_bypass_enabled": True,
        "price_delta_bypass_threshold": 0.00062,
        "rsi_session_oversold_threshold": 30.0,
    }
    coordinator = _build_coordinator(cfg_override)
    symbol = "SOL/USDT"

    # First signal seeds history
    first = {"symbol": symbol, "entry": 100.0, "side": "long", "rsi": 25.0}
    assert coordinator.validate_duplicate(first, "strat")[0] is True

    # Small dip is below better-price threshold, so it is rejected,
    # but it should start/update the RSI session anchor.
    dip = {"symbol": symbol, "entry": 99.995, "side": "long", "rsi": 25.0}
    ok, reason = coordinator.validate_duplicate(dip, "strat")
    assert ok is False
    assert symbol in coordinator.rsi_session_state
    assert coordinator.rsi_session_state[symbol]["anchor_price"] == pytest.approx(99.995)

    # Bounce is accepted because price_delta is computed vs the anchor (99.995),
    # which makes the delta slightly larger than vs the last accepted price (100.0).
    bounce = {"symbol": symbol, "entry": 100.06, "side": "long", "rsi": 25.0}
    ok, reason = coordinator.validate_duplicate(bounce, "strat")
    assert ok is True
    assert "bypass" in reason.lower()


def test_rsi_session_resets_when_rsi_recovers():
    cfg_override = {
        "cooldown_seconds": 60,
        "price_delta_bypass_enabled": True,
        "price_delta_bypass_threshold": 0.5,  # prevent bypass; we only care about state reset
        "rsi_session_oversold_threshold": 30.0,
    }
    coordinator = _build_coordinator(cfg_override)
    symbol = "ADA/USDT"

    # Seed history
    assert coordinator.validate_duplicate({"symbol": symbol, "entry": 10.0, "side": "long", "rsi": 25.0}, "strat")[0] is True
    # Create session during cooldown
    coordinator.validate_duplicate({"symbol": symbol, "entry": 9.9995, "side": "long", "rsi": 25.0}, "strat")
    assert symbol in coordinator.rsi_session_state

    # RSI recovers; session should be cleared.
    coordinator.validate_duplicate({"symbol": symbol, "entry": 10.0, "side": "long", "rsi": 31.0}, "strat")
    assert symbol not in coordinator.rsi_session_state
