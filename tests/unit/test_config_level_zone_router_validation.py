import copy

import pytest

from src.config.schema import ConfigSafetyError, validate_config_safety


def _base_config() -> dict:
    return {
        "risk": {"max_position_size": 0.25},
        "universe": {"prefetch": {"startup_candle_count": 2000}},
        "ml": {"enabled": False},
        "strategies": {
            "level_zone_router": {
                "enabled": True,
                "source": {"mode": "consensus", "timeframes": ["15m", "1h"]},
                "levels": {
                    "pivot_left": 5,
                    "pivot_right": 3,
                    "lookback_bars": 200,
                    "band_pct": 0.005,
                    "smc_cluster_pct": 0.01,
                    "min_cluster_n": 2,
                    "kmin": 2,
                    "kmax": 8,
                    "touch_proximity_bps": 30.0,
                },
                "zones": {
                    "near_level_bps": 50.0,
                    "decision_zone_low": 0.4,
                    "decision_zone_high": 0.6,
                    "no_trade_new_entry": True,
                },
                "breakout": {"min_close_bars": 2, "min_volume_mult": 1.5},
                "rollout": {"mode": "observe", "canary_symbols": ["BTC/USDT:USDT"]},
            }
        },
    }


def test_level_zone_router_valid_config_passes():
    validate_config_safety(_base_config())


def test_level_zone_router_rejects_invalid_source_mode():
    cfg = _base_config()
    cfg["strategies"]["level_zone_router"]["source"]["mode"] = "shadow"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.level_zone_router.source.mode" in str(exc.value)


def test_level_zone_router_rejects_invalid_decision_zone_order():
    cfg = _base_config()
    cfg["strategies"]["level_zone_router"]["zones"]["decision_zone_low"] = 0.8
    cfg["strategies"]["level_zone_router"]["zones"]["decision_zone_high"] = 0.2

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "decision_zone_low must be <= decision_zone_high" in str(exc.value)


def test_level_zone_router_validation_noop_when_block_missing():
    cfg = _base_config()
    cfg_no_router = copy.deepcopy(cfg)
    del cfg_no_router["strategies"]["level_zone_router"]

    validate_config_safety(cfg_no_router)


def test_level_zone_router_rejects_invalid_rollout_mode():
    cfg = _base_config()
    cfg["strategies"]["level_zone_router"]["rollout"]["mode"] = "pilot"

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.level_zone_router.rollout.mode" in str(exc.value)


def test_level_zone_router_rejects_invalid_rollout_canary_symbol_token():
    cfg = _base_config()
    cfg["strategies"]["level_zone_router"]["rollout"]["canary_symbols"] = ["BTC /USDT:USDT"]

    with pytest.raises(ConfigSafetyError) as exc:
        validate_config_safety(cfg)
    assert "strategies.level_zone_router.rollout.canary_symbols" in str(exc.value)


def test_level_zone_router_accepts_single_item_mapping_canary_symbol_token():
    cfg = _base_config()
    cfg["strategies"]["level_zone_router"]["rollout"]["canary_symbols"] = [{"BTC/USDT": "USDT"}]

    validate_config_safety(cfg)
