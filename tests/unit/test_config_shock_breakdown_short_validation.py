import copy

import pytest

from src.config.schema import ConfigSafetyError, validate_config_safety


def _base_config() -> dict:
    return {
        "risk": {"max_position_size": 0.25},
        "universe": {"prefetch": {"startup_candle_count": 2000}},
        "ml": {"enabled": False},
        "strategies": {
            "shock_breakdown_short": {
                "enabled": True,
                "timeframe": "5m",
                "shock_state": "ARMED",
                "min_shock_score": 0.60,
                "breakdown_lookback_bars": 20,
                "breakdown_confirm_bps": 5.0,
                "momentum_lookback_bars": 3,
                "min_momentum_pct": 0.003,
                "volume_ma_window": 20,
                "min_volume_mult": 1.20,
                "cooldown_seconds": 600,
                "take_profit_pct": 0.010,
                "stop_loss_pct": 0.006,
                "rollout": {"mode": "observe", "canary_symbols": ["BTC/USDT:USDT"]},
                "exit_settings": {"max_hold_seconds": 900},
            }
        },
    }


def test_shock_breakdown_short_validation_accepts_valid_block():
    validate_config_safety(_base_config())


def test_shock_breakdown_short_validation_rejects_non_mapping_block():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"] = "enabled"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short" in str(exc.value)


def test_shock_breakdown_short_validation_rejects_invalid_rollout_mode():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"]["rollout"]["mode"] = "aggressive"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short.rollout.mode" in str(exc.value)


def test_shock_breakdown_short_validation_rejects_invalid_canary_symbol():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"]["rollout"]["canary_symbols"] = ["BTC/USDT:USDT", "BAD TOKEN"]

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short.rollout.canary_symbols" in str(exc.value)


def test_shock_breakdown_short_validation_rejects_out_of_range_min_shock_score():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"]["min_shock_score"] = 1.2

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short.min_shock_score" in str(exc.value)


def test_shock_breakdown_short_validation_rejects_non_positive_stop_loss_pct():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"]["stop_loss_pct"] = 0

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short.stop_loss_pct" in str(exc.value)


def test_shock_breakdown_short_validation_rejects_non_positive_max_hold_seconds():
    cfg = _base_config()
    cfg["strategies"]["shock_breakdown_short"]["exit_settings"]["max_hold_seconds"] = 0

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.shock_breakdown_short.exit_settings.max_hold_seconds" in str(exc.value)


def test_shock_breakdown_short_validation_noop_when_block_missing():
    cfg = _base_config()
    cfg_no_block = copy.deepcopy(cfg)
    del cfg_no_block["strategies"]["shock_breakdown_short"]

    validate_config_safety(cfg_no_block)
