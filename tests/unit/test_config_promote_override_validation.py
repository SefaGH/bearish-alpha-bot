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
                "fast_watch": {
                    "promote_override": {
                        "enabled": True,
                        "mode": "observe",
                        "canary_symbols": [],
                    }
                }
            }
        },
    }


@pytest.mark.parametrize("mode", ["observe", "enforce", "off", "disabled"])
def test_promote_override_mode_accepts_allowed_values(mode: str):
    cfg = _base_config()
    cfg["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["mode"] = mode
    validate_config_safety(cfg)


def test_promote_override_mode_rejects_invalid_value():
    cfg = _base_config()
    cfg["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["mode"] = "shadow"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)

    msg = str(exc.value)
    assert "strategies.mean_reversion.fast_watch.promote_override.mode" in msg
    assert "allowed: observe|enforce|off|disabled" in msg


def test_promote_override_canary_symbols_accepts_list_and_csv():
    cfg_list = _base_config()
    cfg_list["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["canary_symbols"] = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "*",
    ]
    validate_config_safety(cfg_list)

    cfg_csv = _base_config()
    cfg_csv["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["canary_symbols"] = (
        "BTC/USDT:USDT,ETH/USDT:USDT,*"
    )
    validate_config_safety(cfg_csv)


def test_promote_override_canary_symbols_rejects_invalid_type():
    cfg = _base_config()
    cfg["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["canary_symbols"] = {
        "symbol": "BTC/USDT:USDT"
    }

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)

    assert "strategies.mean_reversion.fast_watch.promote_override.canary_symbols" in str(exc.value)


def test_promote_override_canary_symbols_rejects_invalid_tokens():
    cfg_non_string = _base_config()
    cfg_non_string["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["canary_symbols"] = [
        "BTC/USDT:USDT",
        123,
    ]
    with pytest.raises(ConfigSafetyError) as exc_non_string:
        validate_config_safety(cfg_non_string)
    assert "canary_symbols[1]" in str(exc_non_string.value)

    cfg_whitespace = _base_config()
    cfg_whitespace["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]["canary_symbols"] = [
        "BTC /USDT:USDT",
    ]
    with pytest.raises(ConfigSafetyError) as exc_whitespace:
        validate_config_safety(cfg_whitespace)
    assert "canary_symbols[0]" in str(exc_whitespace.value)


def test_promote_override_rollout_validation_is_noop_when_block_missing():
    cfg = _base_config()
    cfg_no_block = copy.deepcopy(cfg)
    del cfg_no_block["strategies"]["mean_reversion"]["fast_watch"]["promote_override"]

    validate_config_safety(cfg_no_block)
