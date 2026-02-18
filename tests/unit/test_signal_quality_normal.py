import pytest
from unittest.mock import MagicMock

from src.core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(cfg=None) -> StrategyCoordinator:
    pm = MagicMock()
    pm.cfg = cfg or {}
    pm.performance_monitor = None
    pm.exchange_clients = {}
    pm.get_strategy_allocation.return_value = 0.1

    rm = MagicMock()
    return StrategyCoordinator(pm, rm, market_data_pipeline=None, config=cfg or {})


def test_quality_neutral_defaults():
    coord = _make_coordinator()
    signal = {"symbol": "BTC/USDT"}

    result = coord._compute_signal_quality(signal)

    assert result["value"] == pytest.approx(0.5, rel=1e-3)
    assert signal["quality_score"] == result["value"]


def test_quality_weak_setup_has_low_score():
    coord = _make_coordinator()
    signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "ml_confidence": 0.2,
        "volume_strength": 0.2,
        "momentum_strength": 0.25,
        "regime_confidence": 0.2,
        "ppo_long_score": 0.2,
        "rl_is_agree": False,
    }

    result = coord._compute_signal_quality(signal)

    # Roughly 0.21 with default weights; ensure it is clearly below neutral.
    assert result["value"] == pytest.approx(0.21, rel=1e-2)
    assert result["value"] < 0.3


def test_quality_strong_setup_high_score_with_rr_bonus():
    coord = _make_coordinator()
    signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "ml_confidence": 0.9,
        "volume_strength": 1.0,
        "momentum_strength": 0.9,
        "regime_confidence": 0.8,
        "ppo_long_score": 1.0,
        "rl_is_agree": True,
        "spread_component": 0.6,
        "rr_ratio": 3.0,  # triggers a small positive adjustment
    }

    result = coord._compute_signal_quality(signal)

    assert result["value"] >= 0.9
    assert result["value"] <= 1.0
    assert result["rr_adjustment"] > 1.0


def test_quality_excludes_unhealthy_ppo_component():
    coord = _make_coordinator()
    signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "ppo_long_score": 0.0,
        "ppo_meta": {"reason": "health_guard_fast", "guard_active": True, "health_ok": False},
    }

    result = coord._compute_signal_quality(signal)

    assert result["value"] == pytest.approx(0.5, rel=1e-3)
    assert "ppo_rl" in result.get("excluded_components", [])
    assert result.get("component_health", {}).get("ppo_rl", {}).get("healthy") is False
    assert result.get("component_health", {}).get("ppo_rl", {}).get("included") is False


def test_quality_excludes_unhealthy_ml_component():
    coord = _make_coordinator()
    signal = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "ml_confidence": 0.1,
        "ml_context_is_healthy": False,
        "ml_context_reason": "ml_backend_unhealthy",
        "ppo_long_score": 0.5,
        "ppo_meta": {"reason": "ok", "health_ok": True},
    }

    result = coord._compute_signal_quality(signal)

    assert result["value"] == pytest.approx(0.5, rel=1e-3)
    assert "ml" in result.get("excluded_components", [])
    assert result.get("component_health", {}).get("ml", {}).get("healthy") is False
    assert result.get("component_health", {}).get("ml", {}).get("included") is False
