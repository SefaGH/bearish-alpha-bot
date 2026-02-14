import copy
from unittest.mock import MagicMock

import pytest

from src.core.directional_bias import compute_directional_bias_adjustment
from src.core.strategy_coordinator import StrategyCoordinator


def _make_coordinator(cfg=None) -> StrategyCoordinator:
    pm = MagicMock()
    pm.cfg = cfg or {}
    pm.performance_monitor = None
    pm.exchange_clients = {}
    pm.get_strategy_allocation.return_value = 0.1

    rm = MagicMock()
    return StrategyCoordinator(pm, rm, market_data_pipeline=None, config=cfg or {})


def test_directional_bias_adjustment_aligned_breakout_up_boosts_long():
    signal = {"side": "buy", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}
    cfg = {"enabled": True, "mode": "quality_adjust_only", "weight": 0.10, "max_quality_delta": 0.08}

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is True
    assert out["delta"] > 0
    assert out["reason"] == "directional_bias.aligned_boost"


def test_directional_bias_adjustment_countertrend_penalizes_short_on_breakout_up():
    signal = {"side": "short", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}
    cfg = {"enabled": True, "mode": "quality_adjust_only", "weight": 0.10, "max_quality_delta": 0.08}

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is True
    assert out["delta"] < 0
    assert out["reason"] == "directional_bias.countertrend_penalty"


def test_directional_bias_adjustment_at_level_applies_penalty():
    signal = {"side": "buy", "level_zone_snapshot": {"zone": "AT_LEVEL"}}
    cfg = {"enabled": True, "mode": "quality_adjust_only", "at_level_penalty": 0.05, "max_quality_delta": 0.08}

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is True
    assert out["delta"] == pytest.approx(-0.05, rel=1e-6)
    assert out["reason"] == "directional_bias.at_level_penalty"


def test_strategy_quality_applies_directional_bias_delta_when_enabled():
    base_cfg = {"signals": {"directional_bias": {"enabled": False}}}
    cfg = {
        "signals": {
            "directional_bias": {
                "enabled": True,
                "mode": "quality_adjust_only",
                "weight": 0.10,
                "max_quality_delta": 0.08,
                "at_level_penalty": 0.05,
            }
        }
    }
    signal = {"symbol": "BTC/USDT", "side": "buy", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}

    coord_base = _make_coordinator(base_cfg)
    out_base = coord_base._compute_signal_quality(copy.deepcopy(signal))

    coord_adj = _make_coordinator(cfg)
    out_adj = coord_adj._compute_signal_quality(copy.deepcopy(signal))

    assert out_adj["value"] > out_base["value"]
    assert "directional_bias" in out_adj
    assert out_adj["directional_bias"]["applied"] is True


def test_directional_bias_rollout_observe_reports_would_delta_without_applying():
    signal = {"symbol": "BTC/USDT:USDT", "side": "buy", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}
    cfg = {
        "enabled": True,
        "mode": "quality_adjust_only",
        "weight": 0.10,
        "max_quality_delta": 0.08,
        "rollout": {"mode": "observe", "canary_symbols": ["BTC/USDT:USDT"]},
    }

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is False
    assert out["delta"] == pytest.approx(0.0, rel=1e-6)
    assert out["would_delta"] > 0
    assert out["reason"] == "directional_bias.observe_only"


def test_directional_bias_rollout_out_of_scope_returns_noop():
    signal = {"symbol": "ETH/USDT:USDT", "side": "buy", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}
    cfg = {
        "enabled": True,
        "mode": "quality_adjust_only",
        "weight": 0.10,
        "max_quality_delta": 0.08,
        "rollout": {"mode": "enforce", "canary_symbols": ["BTC/USDT:USDT"]},
    }

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is False
    assert out["delta"] == pytest.approx(0.0, rel=1e-6)
    assert out["reason"] == "directional_bias.rollout_out_of_scope"


def test_directional_bias_rollout_accepts_mapping_style_canary_token():
    signal = {"symbol": "BTC/USDT:USDT", "side": "buy", "level_zone_snapshot": {"zone": "BREAKOUT_UP_CONFIRMED"}}
    cfg = {
        "enabled": True,
        "mode": "quality_adjust_only",
        "weight": 0.10,
        "max_quality_delta": 0.08,
        "rollout": {"mode": "observe", "canary_symbols": [{"BTC/USDT": "USDT"}]},
    }

    out = compute_directional_bias_adjustment(signal, cfg)
    assert out["enabled"] is True
    assert out["applied"] is False
    assert out["reason"] == "directional_bias.observe_only"
