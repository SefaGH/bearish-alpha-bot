import copy
import logging
from pathlib import Path

import pytest
import yaml

from src.config.live_trading_config import LiveTradingConfiguration


def test_format_risk_summary_prefers_computed_max_risk():
    summary = LiveTradingConfiguration._format_risk_summary(
        {
            'computed_max_risk_usd': 25.0,
            'per_trade_risk_pct': 1.0,
        },
        capital_val=2500.0,
    )
    assert "1.00%" in summary
    assert "25.00 USDT" in summary


def test_format_risk_summary_falls_back_to_env(monkeypatch):
    monkeypatch.setenv('PER_TRADE_RISK_PCT', '2')
    summary = LiveTradingConfiguration._format_risk_summary({}, capital_val=1000.0)
    assert "2.00%" in summary
    assert "20.00 USDT" in summary
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)


def test_normalize_risk_config_converts_percent_value(monkeypatch):
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)
    cfg = {'risk': {'per_trade_risk_pct': 1.0, 'equity_usd': 100.0}}
    LiveTradingConfiguration()._normalize_risk_config(cfg)
    assert cfg['risk']['per_trade_risk_pct'] == pytest.approx(0.01, rel=1e-4)
    assert cfg['risk']['computed_max_risk_usd'] == pytest.approx(1.0, rel=1e-4)


def test_normalize_risk_config_uses_env_default(monkeypatch):
    monkeypatch.setenv('PER_TRADE_RISK_PCT', '2')
    cfg = {'risk': {'equity_usd': 50.0}}
    LiveTradingConfiguration()._normalize_risk_config(cfg)
    assert cfg['risk']['per_trade_risk_pct'] == pytest.approx(0.02, rel=1e-4)
    assert cfg['risk']['computed_max_risk_usd'] == pytest.approx(1.0, rel=1e-4)
    monkeypatch.delenv('PER_TRADE_RISK_PCT', raising=False)


def _set_path(config, dotted_path, value):
    cursor = config
    parts = dotted_path.split('.')
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def _base_mtf_cfg():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    return copy.deepcopy(base_config["signals"]["short_the_rip"]["mtf_confirmation"])


def test_allowlist_type_coercion_handles_appconfig_strings():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)

    overrides = {
        "signals.short_the_rip.mtf_confirmation.enabled": "true",
        "signals.short_the_rip.mtf_confirmation.require_15m": "false",
        "signals.short_the_rip.mtf_confirmation.require_1h": "false",
        "pyramiding.enabled": "true",
        "ml.reinforcement_learning.training_mode": "false",
        "risk.min_stop_pct": "0.5",
        "strategies.regime_routing.bullish.preferred_strategies": '["trend_follower","breakout_hunter"]',
    }

    for path, value in overrides.items():
        _set_path(cfg, path, value)

    schema = LiveTradingConfiguration._build_type_schema(base_config)
    LiveTradingConfiguration._apply_type_coercion_allowlist(cfg, schema)

    mtf_cfg = cfg["signals"]["short_the_rip"]["mtf_confirmation"]
    assert mtf_cfg["enabled"] is True
    assert mtf_cfg["require_15m"] is False
    assert mtf_cfg["require_1h"] is False
    assert cfg["pyramiding"]["enabled"] is True
    assert cfg["ml"]["reinforcement_learning"]["training_mode"] is False
    assert cfg["risk"]["min_stop_pct"] == pytest.approx(0.5)
    assert cfg["strategies"]["regime_routing"]["bullish"]["preferred_strategies"] == [
        "trend_follower",
        "breakout_hunter",
    ]


def test_schema_type_coercion_casts_rr_dynamic_values():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)

    _set_path(cfg, "risk.rr_dynamic.base_target_rr", "2.5")
    _set_path(cfg, "risk.rr_dynamic.enabled", "false")
    _set_path(cfg, "signals.short_the_rip.mtf_confirmation.require_15m", "false")

    schema = LiveTradingConfiguration._build_type_schema(base_config)
    LiveTradingConfiguration._apply_schema_type_coercion(cfg, schema)

    assert cfg["risk"]["rr_dynamic"]["base_target_rr"] == pytest.approx(2.5)
    assert cfg["risk"]["rr_dynamic"]["enabled"] is False
    assert cfg["signals"]["short_the_rip"]["mtf_confirmation"]["require_15m"] is False


def test_mtf_policy_defaults_to_hard_when_enabled():
    mtf_cfg = _base_mtf_cfg()
    mtf_cfg.pop("15m_mode", None)
    mtf_cfg.pop("1h_mode", None)
    mtf_cfg["enabled"] = True

    cfg = {"signals": {"short_the_rip": {"mtf_confirmation": mtf_cfg}}}
    LiveTradingConfiguration()._normalize_str_mtf_config(cfg)

    policy = cfg["signals"]["short_the_rip"]["mtf_confirmation_effective"]
    assert policy.tf_15m.mode == "hard"
    assert policy.tf_1h.mode == "hard"


def test_mtf_policy_disabled_forces_modes_off():
    mtf_cfg = _base_mtf_cfg()
    mtf_cfg["enabled"] = False
    mtf_cfg["15m_mode"] = "hard"
    mtf_cfg["1h_mode"] = "soft"

    cfg = {"signals": {"short_the_rip": {"mtf_confirmation": mtf_cfg}}}
    LiveTradingConfiguration()._normalize_str_mtf_config(cfg)

    policy = cfg["signals"]["short_the_rip"]["mtf_confirmation_effective"]
    assert policy.tf_15m.mode == "off"
    assert policy.tf_1h.mode == "off"


def test_mtf_policy_rejects_null_threshold():
    mtf_cfg = _base_mtf_cfg()
    mtf_cfg["rsi_15m_min"] = None

    cfg = {"signals": {"short_the_rip": {"mtf_confirmation": mtf_cfg}}}
    with pytest.raises(ValueError):
        LiveTradingConfiguration()._normalize_str_mtf_config(cfg)


def test_mtf_policy_rejects_invalid_mode():
    mtf_cfg = _base_mtf_cfg()
    mtf_cfg["15m_mode"] = "maybe"

    cfg = {"signals": {"short_the_rip": {"mtf_confirmation": mtf_cfg}}}
    with pytest.raises(ValueError):
        LiveTradingConfiguration()._normalize_str_mtf_config(cfg)


def test_mtf_policy_rejects_invalid_on_missing():
    mtf_cfg = _base_mtf_cfg()
    mtf_cfg["on_missing_15m"] = "nope"

    cfg = {"signals": {"short_the_rip": {"mtf_confirmation": mtf_cfg}}}
    with pytest.raises(ValueError):
        LiveTradingConfiguration()._normalize_str_mtf_config(cfg)


def test_mtf_bool_string_coercion_prevents_truthy():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)
    _set_path(cfg, "signals.short_the_rip.mtf_confirmation.missing_15m_is_fatal", "false")

    schema = LiveTradingConfiguration._build_type_schema(base_config)
    LiveTradingConfiguration._apply_schema_type_coercion(cfg, schema)
    LiveTradingConfiguration()._normalize_str_mtf_config(cfg)

    policy = cfg["signals"]["short_the_rip"]["mtf_confirmation_effective"]
    assert policy.tf_15m.missing_is_fatal is False


def test_appconfig_symbol_segment_preserves_case():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    schema = LiveTradingConfiguration._build_type_schema(base_config)

    nested = LiveTradingConfiguration._flatten_to_nested({
        "signals.short_the_rip.symbols.btc/usdt:usdt.rsi_threshold": "55",
    })
    LiveTradingConfiguration._apply_schema_type_coercion(nested, schema)

    assert (
        nested["signals"]["short_the_rip"]["symbols"]["BTC/USDT:USDT"]["rsi_threshold"]
        == 55
    )


def test_schema_defaults_keep_inline_list_types():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)

    schema = LiveTradingConfiguration._build_type_schema(base_config)
    LiveTradingConfiguration._apply_schema_type_coercion(cfg, schema)

    models = cfg["ml"]["price_prediction"]["models"]
    assert isinstance(models, list)
    assert "lstm" in models


def test_heuristic_coercion_for_unknown_keys():
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)

    cfg["extras"] = {
        "int_val": "10",
        "float_val": "0.003",
        "bool_val": "true",
        "list_literal": "['lstm','transformer']",
        "list_comma": "1,2,3",
    }

    schema = LiveTradingConfiguration._build_type_schema(base_config)
    LiveTradingConfiguration._apply_schema_type_coercion(cfg, schema)
    LiveTradingConfiguration._apply_heuristic_type_coercion(cfg, schema)

    extras = cfg["extras"]
    assert extras["int_val"] == 10
    assert extras["float_val"] == pytest.approx(0.003)
    assert extras["bool_val"] is True
    assert extras["list_literal"] == ["lstm", "transformer"]
    assert extras["list_comma"] == [1, 2, 3]


def test_operational_schema_casts_runtime_keys(monkeypatch):
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = copy.deepcopy(base_config)
    cfg.update({
        "debug_mode": "false",
        "ccxt_timeout_ms": "12000",
        "ticker_cache_ttl_s": "0.5",
        "ticker_max_attempts": "3",
        "ticker_retry_base_delay_s": "0.25",
        "telegram_chat_id": "123456",
        "exchanges": "bingx,kucoin",
        "trading_duration": "900",
        "trading_mode": "paper",
    })

    monkeypatch.delenv('CONFIG_STRICT_TYPE_CHECK', raising=False)
    canonical = LiveTradingConfiguration._build_type_schema(base_config)
    operational = LiveTradingConfiguration._build_operational_schema()
    combined = LiveTradingConfiguration._merge_operational_schema(canonical, operational)
    LiveTradingConfiguration._apply_schema_type_coercion(cfg, combined)

    assert cfg["debug_mode"] is False
    assert cfg["ccxt_timeout_ms"] == 12000
    assert cfg["ticker_cache_ttl_s"] == pytest.approx(0.5)
    assert cfg["ticker_max_attempts"] == 3
    assert cfg["ticker_retry_base_delay_s"] == pytest.approx(0.25)
    assert cfg["telegram_chat_id"] == 123456
    assert cfg["exchanges"] == ["bingx", "kucoin"]
    assert cfg["trading_duration"] == 900
    assert cfg["trading_mode"] == "paper"


def test_operational_appconfig_keys_do_not_warn(caplog, monkeypatch):
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = LiveTradingConfiguration()
    cfg._appconfig_normalized_paths = ["debug_mode", "ccxt_timeout_ms"]

    canonical = LiveTradingConfiguration._build_type_schema(base_config)
    operational = LiveTradingConfiguration._build_operational_schema()
    monkeypatch.delenv('CONFIG_STRICT_TYPE_CHECK', raising=False)

    with caplog.at_level(logging.INFO):
        cfg._warn_unknown_appconfig_keys(canonical, operational)

    assert not any(
        record.levelno >= logging.WARNING
        and "AppConfig keys not in canonical or operational schema" in record.message
        for record in caplog.records
    )


def test_unknown_appconfig_key_warns(caplog, monkeypatch):
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = LiveTradingConfiguration()
    cfg._appconfig_normalized_paths = ["mystery_key"]

    canonical = LiveTradingConfiguration._build_type_schema(base_config)
    operational = LiveTradingConfiguration._build_operational_schema()
    monkeypatch.delenv('CONFIG_STRICT_TYPE_CHECK', raising=False)

    with caplog.at_level(logging.WARNING):
        cfg._warn_unknown_appconfig_keys(canonical, operational)

    assert any(
        "AppConfig keys not in canonical or operational schema" in record.message
        for record in caplog.records
    )


def test_unknown_appconfig_key_strict_raises(monkeypatch):
    base_config = yaml.safe_load(
        Path("config/config.example.yaml").read_text(encoding="utf-8")
    )
    cfg = LiveTradingConfiguration()
    cfg._appconfig_normalized_paths = ["mystery_key"]

    canonical = LiveTradingConfiguration._build_type_schema(base_config)
    operational = LiveTradingConfiguration._build_operational_schema()
    monkeypatch.setenv('CONFIG_STRICT_TYPE_CHECK', 'true')

    with pytest.raises(ValueError):
        cfg._warn_unknown_appconfig_keys(canonical, operational)


def test_normalize_rollout_canary_symbols_repairs_mapping_tokens():
    cfg = {
        "strategies": {
            "level_zone_router": {
                "rollout": {
                    "canary_symbols": [{"BTC/USDT": "USDT"}],
                }
            },
            "rsi_zone_router": {
                "transition": {
                    "shock_override": {
                        "canary_symbols": [{"BTC/USDT": "USDT"}],
                    }
                }
            },
            "shock_breakdown_short": {
                "rollout": {
                    "canary_symbols": [{"BTC/USDT": "USDT"}],
                }
            },
        },
        "signals": {
            "directional_bias": {
                "rollout": {
                    "canary_symbols": '[{"BTC/USDT":"USDT"}]',
                }
            }
        },
    }

    loader = LiveTradingConfiguration()
    loader._normalize_rollout_canary_symbols(cfg)

    assert cfg["strategies"]["level_zone_router"]["rollout"]["canary_symbols"] == ["BTC/USDT:USDT"]
    assert (
        cfg["strategies"]["rsi_zone_router"]["transition"]["shock_override"]["canary_symbols"]
        == ["BTC/USDT:USDT"]
    )
    assert cfg["strategies"]["shock_breakdown_short"]["rollout"]["canary_symbols"] == ["BTC/USDT:USDT"]
    assert cfg["signals"]["directional_bias"]["rollout"]["canary_symbols"] == ["BTC/USDT:USDT"]


def test_normalize_rollout_canary_symbols_repairs_indexed_dict_shape():
    cfg = {
        "strategies": {
            "level_zone_router": {
                "rollout": {
                    "canary_symbols": {"0": {"BTC/USDT": "USDT"}},
                }
            }
        }
    }

    loader = LiveTradingConfiguration()
    loader._normalize_rollout_canary_symbols(cfg)

    assert cfg["strategies"]["level_zone_router"]["rollout"]["canary_symbols"] == ["BTC/USDT:USDT"]
