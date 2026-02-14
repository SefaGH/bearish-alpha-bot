import pytest

from src.config.schema import ConfigSafetyError, validate_config_safety


def _base_config() -> dict:
    return {
        "risk": {"max_position_size": 0.25},
        "universe": {"prefetch": {"startup_candle_count": 2000}},
        "ml": {"enabled": False},
        "signals": {
            "directional_bias": {
                "enabled": True,
                "mode": "quality_adjust_only",
                "weight": 0.10,
                "max_quality_delta": 0.08,
                "at_level_penalty": 0.05,
                "rollout": {"mode": "observe", "canary_symbols": ["BTC/USDT:USDT"]},
            }
        },
    }


def test_directional_bias_validation_accepts_valid_config():
    validate_config_safety(_base_config())


def test_directional_bias_validation_rejects_invalid_mode():
    cfg = _base_config()
    cfg["signals"]["directional_bias"]["mode"] = "observe"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "signals.directional_bias.mode" in str(exc.value)


def test_directional_bias_validation_rejects_negative_weight():
    cfg = _base_config()
    cfg["signals"]["directional_bias"]["weight"] = -0.1

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "signals.directional_bias.weight" in str(exc.value)


def test_directional_bias_validation_rejects_invalid_rollout_mode():
    cfg = _base_config()
    cfg["signals"]["directional_bias"]["rollout"]["mode"] = "pilot"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "signals.directional_bias.rollout.mode" in str(exc.value)


def test_directional_bias_validation_rejects_invalid_rollout_canary_symbol_token():
    cfg = _base_config()
    cfg["signals"]["directional_bias"]["rollout"]["canary_symbols"] = ["BTC /USDT:USDT"]

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "signals.directional_bias.rollout.canary_symbols" in str(exc.value)


def test_directional_bias_validation_accepts_single_item_mapping_canary_symbol_token():
    cfg = _base_config()
    cfg["signals"]["directional_bias"]["rollout"]["canary_symbols"] = [{"BTC/USDT": "USDT"}]

    validate_config_safety(cfg)
