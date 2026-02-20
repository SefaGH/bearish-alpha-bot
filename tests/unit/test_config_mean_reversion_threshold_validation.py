import copy

import pytest

from src.config.schema import ConfigSafetyError, validate_config_safety


def _base_config() -> dict:
    return {
        "risk": {"max_position_size": 0.25},
        "universe": {"prefetch": {"startup_candle_count": 2000}},
        "ml": {"enabled": False},
        "strategies": {
            "mean_reversion": {
                "high_adx_z_threshold": 1.9,
                "high_adx_z_threshold_floor": 1.6,
            }
        },
    }


def test_mean_reversion_threshold_validation_accepts_valid_values():
    validate_config_safety(_base_config())


@pytest.mark.parametrize("key", ["high_adx_z_threshold", "high_adx_z_threshold_floor"])
def test_mean_reversion_threshold_validation_rejects_non_numeric_values(key: str):
    cfg = _base_config()
    cfg["strategies"]["mean_reversion"][key] = "not-a-number"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert f"strategies.mean_reversion.{key}" in str(exc.value)


@pytest.mark.parametrize("key", ["high_adx_z_threshold", "high_adx_z_threshold_floor"])
def test_mean_reversion_threshold_validation_rejects_non_positive_values(key: str):
    cfg = _base_config()
    cfg["strategies"]["mean_reversion"][key] = 0

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert f"strategies.mean_reversion.{key}" in str(exc.value)


def test_mean_reversion_threshold_validation_noop_when_block_missing():
    cfg = _base_config()
    cfg_no_mr = copy.deepcopy(cfg)
    del cfg_no_mr["strategies"]["mean_reversion"]

    validate_config_safety(cfg_no_mr)
