import copy

import pytest

from src.config.schema import ConfigSafetyError, validate_config_safety


def _base_config() -> dict:
    return {
        "risk": {"max_position_size": 0.25},
        "universe": {"prefetch": {"startup_candle_count": 2000}},
        "ml": {"enabled": False},
        "strategies": {
            "rsi_zone_router": {
                "enabled": True,
                "source": {"mode": "consensus", "mr_mode": "slow_only", "slow_tf": "30m", "fast_tf": "5m"},
                "thresholds": {
                    "ob_floor": 10.0,
                    "ob_cap": 45.0,
                    "str_floor": 55.0,
                    "str_cap": 90.0,
                    "min_gap": 8.0,
                },
                "transition": {
                    "width": 5.0,
                    "no_trade_new_entry": True,
                    "mismatch_extreme_override": {
                        "enabled": True,
                        "low_side_enabled": True,
                        "high_side_enabled": False,
                        "min_penetration": 1.0,
                    },
                    "shock_override": {
                        "enabled": True,
                        "mode": "observe",
                        "canary_symbols": ["*"],
                        "allow_strategies": ["adaptive_str", "short_the_rip"],
                        "state": "ARMED",
                        "min_score": 0.60,
                        "min_adx": 25.0,
                    },
                },
            }
        },
    }


def test_rsi_zone_router_validation_accepts_valid_mismatch_override_block():
    validate_config_safety(_base_config())


def test_rsi_zone_router_validation_rejects_non_mapping_override_block():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["mismatch_extreme_override"] = "enabled"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.mismatch_extreme_override" in str(exc.value)


@pytest.mark.parametrize("key", ["enabled", "low_side_enabled", "high_side_enabled"])
def test_rsi_zone_router_validation_rejects_non_boolean_override_flags(key: str):
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["mismatch_extreme_override"][key] = "true"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert f"strategies.rsi_zone_router.transition.mismatch_extreme_override.{key}" in str(exc.value)


def test_rsi_zone_router_validation_rejects_negative_min_penetration():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["mismatch_extreme_override"]["min_penetration"] = -0.1

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.mismatch_extreme_override.min_penetration" in str(exc.value)


def test_rsi_zone_router_validation_rejects_non_finite_min_penetration():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["mismatch_extreme_override"]["min_penetration"] = float("inf")

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.mismatch_extreme_override.min_penetration" in str(exc.value)


def test_rsi_zone_router_validation_noop_when_router_block_missing():
    cfg = _base_config()
    cfg_no_router = copy.deepcopy(cfg)
    del cfg_no_router["strategies"]["rsi_zone_router"]

    validate_config_safety(cfg_no_router)


@pytest.mark.parametrize("mr_mode", ["slow_only", "follow_source"])
def test_rsi_zone_router_validation_accepts_mr_mode_values(mr_mode: str):
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["source"]["mr_mode"] = mr_mode

    validate_config_safety(cfg)


def test_rsi_zone_router_validation_rejects_invalid_mr_mode():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["source"]["mr_mode"] = "invalid_mode"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.source.mr_mode" in str(exc.value)


def test_rsi_zone_router_validation_rejects_non_string_mr_mode():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["source"]["mr_mode"] = 123

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.source.mr_mode" in str(exc.value)


def test_rsi_zone_router_validation_rejects_non_mapping_shock_override_block():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"] = "enabled"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.shock_override" in str(exc.value)


def test_rsi_zone_router_validation_rejects_invalid_shock_override_mode():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"]["mode"] = "aggressive"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.shock_override.mode" in str(exc.value)


def test_rsi_zone_router_validation_rejects_invalid_shock_override_canary_symbols():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"]["canary_symbols"] = ["BTC/USDT:USDT", "BAD TOKEN"]

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.shock_override.canary_symbols" in str(exc.value)


def test_rsi_zone_router_validation_rejects_out_of_range_shock_override_min_score():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"]["min_score"] = 1.2

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.shock_override.min_score" in str(exc.value)


def test_rsi_zone_router_validation_rejects_negative_shock_override_min_adx():
    cfg = _base_config()
    cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"]["min_adx"] = -1

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.rsi_zone_router.transition.shock_override.min_adx" in str(exc.value)
