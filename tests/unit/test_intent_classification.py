from unittest.mock import MagicMock

import pytest

from src.core.strategy_coordinator import StrategyCoordinator
from src.core.signal_intents import INTENT_ENTRY, INTENT_SCALE_IN


def _make_coordinator(pyramiding_enabled=False, positions=None):
    portfolio_manager = MagicMock()
    risk_manager = MagicMock()

    cfg = {
        "pyramiding": {
            "enabled": pyramiding_enabled,
            "max_layers_per_symbol": 3,
        },
        # Minimal duplicate config to satisfy StrategyCoordinator init paths
        "signals": {
            "duplicate_prevention": {
                "enabled": True,
                "cooldown_seconds": 60,
                "min_price_change_pct": 0.0005,
                "price_delta_bypass_threshold": 0.0015,
            }
        },
    }
    portfolio_manager.cfg = cfg
    positions = positions or []
    portfolio_manager.get_open_positions_for_symbol = MagicMock(return_value=positions)

    coordinator = StrategyCoordinator(portfolio_manager, risk_manager, config=cfg)
    return coordinator


def test_pyramiding_disabled_always_entry():
    positions = [
        {"symbol": "BTC/USDT", "side": "long", "strategy_name": "alpha"},
    ]
    coordinator = _make_coordinator(pyramiding_enabled=False, positions=positions)
    signal = {"symbol": "BTC/USDT", "side": "long"}
    intent = coordinator._determine_intent(signal, "alpha")
    assert intent == INTENT_ENTRY


def test_pyramiding_enabled_no_open_positions_returns_entry():
    coordinator = _make_coordinator(pyramiding_enabled=True, positions=[])
    signal = {"symbol": "ETH/USDT", "side": "long"}
    intent = coordinator._determine_intent(signal, "alpha")
    assert intent == INTENT_ENTRY


def test_pyramiding_enabled_with_matching_position_returns_scale_in():
    positions = [
        {"symbol": "ETH/USDT", "side": "long", "strategy_name": "alpha"},
        {"symbol": "ETH/USDT", "side": "long", "strategy_name": "beta"},
    ]
    coordinator = _make_coordinator(pyramiding_enabled=True, positions=positions)
    signal = {"symbol": "ETH/USDT", "side": "long"}
    intent = coordinator._determine_intent(signal, "alpha")
    assert intent == INTENT_SCALE_IN


def test_pyramiding_enabled_side_mismatch_stays_entry():
    positions = [
        {"symbol": "ETH/USDT", "side": "short", "strategy_name": "alpha"},
    ]
    coordinator = _make_coordinator(pyramiding_enabled=True, positions=positions)
    signal = {"symbol": "ETH/USDT", "side": "long"}
    intent = coordinator._determine_intent(signal, "alpha")
    assert intent == INTENT_ENTRY
