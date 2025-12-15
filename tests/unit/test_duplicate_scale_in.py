import time
from unittest.mock import MagicMock

from src.core.signal_intents import INTENT_SCALE_IN
from src.core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(pyramiding_enabled: bool):
    portfolio_manager = MagicMock()
    risk_manager = MagicMock()
    cfg = {
        "signals": {
            "duplicate_prevention": {
                "enabled": True,
                "cooldown_seconds": 20,
                "min_price_change_pct": 0.0005,
                "price_delta_bypass_enabled": True,
                "price_delta_bypass_threshold": 0.0015,
                "scale_in_min_price_change_pct": 0.0005,
                "scale_in_cooldown_seconds": 20,
            }
        },
        "pyramiding": {
            "enabled": pyramiding_enabled,
            "max_layers_per_symbol": 3,
        },
    }
    portfolio_manager.cfg = cfg
    return StrategyCoordinator(portfolio_manager, risk_manager, config=cfg)


def test_scale_in_behaves_like_entry_when_pyramiding_disabled():
    coordinator = _make_coordinator(pyramiding_enabled=False)
    symbol = "BTC/USDT"
    first = {"symbol": symbol, "entry": 100, "intent": INTENT_SCALE_IN}
    ok, _ = coordinator.validate_duplicate(first, "strat")
    assert ok is True

    # Immediate repeat should be rejected due to cooldown when pyramiding disabled
    second = {"symbol": symbol, "entry": 100, "intent": INTENT_SCALE_IN}
    ok, reason = coordinator.validate_duplicate(second, "strat")
    assert ok is False
    assert "cooldown" in reason.lower()


def test_scale_in_spam_window_rejected_when_pyramiding_enabled():
    coordinator = _make_coordinator(pyramiding_enabled=True)
    symbol = "ETH/USDT"
    # Seed history
    coordinator.validate_duplicate({"symbol": symbol, "entry": 50, "intent": INTENT_SCALE_IN}, "strat")
    # Simulate last signal very recent with tiny delta
    coordinator.last_signal_time[f"{symbol}:strat"] = time.time() - 1.0
    coordinator.signal_price_history[symbol][-1] = (time.time() - 1.0, 50.0)

    spam_candidate = {"symbol": symbol, "entry": 50.0, "intent": INTENT_SCALE_IN}
    ok, reason = coordinator.validate_duplicate(spam_candidate, "strat")
    assert ok is False
    assert "spam" in reason.lower()


def test_scale_in_allows_within_cooldown_when_not_spam_and_pyramiding_enabled():
    coordinator = _make_coordinator(pyramiding_enabled=True)
    symbol = "SOL/USDT"
    # First signal
    coordinator.validate_duplicate({"symbol": symbol, "entry": 20, "intent": INTENT_SCALE_IN}, "strat")
    # Set last signal to 5s ago (within cooldown 20s but outside spam window 3s)
    coordinator.last_signal_time[f"{symbol}:strat"] = time.time() - 5.0
    coordinator.signal_price_history[symbol][-1] = (time.time() - 5.0, 20.0)

    follow_up = {"symbol": symbol, "entry": 20.5, "intent": INTENT_SCALE_IN}
    ok, reason = coordinator.validate_duplicate(follow_up, "strat")
    assert ok is True
    assert "ok" in reason.lower()
