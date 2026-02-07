import pytest

from src.core.smart_entry_policy import apply_smart_entry_policy


@pytest.fixture()
def policy_cfg():
    return {
        "enabled": True,
        "volatility_threshold_bps": 5.0,
        "params": {
            "LONG": {"atr_multiplier": 0.90, "timeout_seconds": 300, "gate_bps": 5.0},
            "SHORT": {"atr_multiplier": 0.85, "timeout_seconds": 240, "gate_bps": 12.0},
        },
        "force_override": False,
    }


def test_smart_entry_low_vol_forces_market(policy_cfg):
    # atr/entry * 10000 = (0.1/30000)*10000 = 0.033 bps < 5 => market
    signal = {"side": "buy", "entry": 30000.0, "atr": 0.1}
    execution_params = {}

    out_signal, out_params, decision = apply_smart_entry_policy(
        signal=signal,
        execution_params=execution_params,
        policy_cfg=policy_cfg,
    )

    assert decision.applied is False
    assert out_params["order_type"] == "market"
    assert "limit_price" not in out_signal


def test_smart_entry_high_vol_injects_limit(policy_cfg):
    # atr/entry * 10000 = (15/30000)*10000 = 5 bps => limit (since < thr is market)
    signal = {"side": "long", "entry": 30000.0, "atr": 15.0}
    execution_params = {}

    out_signal, out_params, decision = apply_smart_entry_policy(
        signal=signal,
        execution_params=execution_params,
        policy_cfg=policy_cfg,
    )

    assert decision.applied is True
    assert out_params["order_type"] == "limit"
    assert out_params["timeout_seconds"] == 300.0
    assert out_params["max_chase_bps"] == 5.0
    assert out_params["market_fallback"] is True

    assert out_signal["_execution_price_locked"] is True
    assert pytest.approx(out_signal["limit_price"], rel=1e-12) == 30000.0 - (0.90 * 15.0)


def test_smart_entry_respects_explicit_overrides(policy_cfg):
    signal = {"side": "short", "entry": 30000.0, "atr": 30.0}
    execution_params = {"order_type": "market"}

    out_signal, out_params, decision = apply_smart_entry_policy(
        signal=signal,
        execution_params=execution_params,
        policy_cfg=policy_cfg,
    )

    assert decision.applied is False
    assert decision.reason == "explicit_execution_override"
    assert out_params["order_type"] == "market"
    assert "limit_price" not in out_signal


def test_smart_entry_missing_atr_can_use_conservative_limit(policy_cfg):
    cfg = dict(policy_cfg)
    cfg["force_market_on_missing_atr"] = False
    cfg["fallback_timeout_seconds"] = 45

    signal = {"side": "buy", "entry": 30000.0}
    execution_params = {}

    out_signal, out_params, decision = apply_smart_entry_policy(
        signal=signal,
        execution_params=execution_params,
        policy_cfg=cfg,
    )

    assert decision.applied is False
    assert decision.reason == "missing_atr_conservative_limit"
    assert out_params["order_type"] == "limit"
    assert out_params["timeout_seconds"] == 45.0
    assert out_params["market_fallback_on_timeout_enabled"] is False
    assert out_signal["limit_price"] == pytest.approx(30000.0, rel=1e-12)


def test_smart_entry_extreme_bucket_bans_market_fallback(policy_cfg):
    cfg = dict(policy_cfg)
    cfg["extreme_market_ban"] = True
    cfg["force_market_on_low_vol"] = False

    signal = {"side": "buy", "entry": 30000.0, "atr": 0.1, "volume_bucket": "EXTREME"}
    execution_params = {}

    out_signal, out_params, decision = apply_smart_entry_policy(
        signal=signal,
        execution_params=execution_params,
        policy_cfg=cfg,
    )

    assert decision.applied is False
    assert decision.reason.startswith("low_vol_conservative_limit_extreme")
    assert out_params["order_type"] == "limit"
    assert out_params["market_fallback_on_timeout_enabled"] is False
    assert out_params["disable_market_fallback_on_extreme_bucket"] is True
    assert out_signal["limit_price"] == pytest.approx(30000.0, rel=1e-12)
