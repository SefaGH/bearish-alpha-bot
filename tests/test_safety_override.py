from src.safety.safety_override import SafetyOverride


def _base_signal(
    *,
    base_threshold: float = 68.0,
    current_threshold: float = 55.0,
    close: float = 100.0,
    ema21: float = 99.0,
    ema50: float = 98.0,
    rsi: float = 60.0,
    candle_open: float = 100.0,
    candle_close: float = 101.0,
    volume: float = 10.0,
    volume_ma20: float = 20.0,
    resistance_distance_bps: float | None = 100.0,
    volume_bucket: str = "LOW",
):
    meta = {
        "adaptive_threshold": {
            "base_threshold": base_threshold,
            "current_threshold": current_threshold,
            "delta": base_threshold - current_threshold,
            "lowered": current_threshold < base_threshold,
        },
        "safety_snapshot": {
            "close": close,
            "rsi": rsi,
            "ema21": ema21,
            "ema50": ema50,
            "candle_open": candle_open,
            "candle_close": candle_close,
            "volume": volume,
            "volume_ma20": volume_ma20,
            "resistance_distance_bps": resistance_distance_bps,
        },
    }
    return {
        "strategy_name": "adaptive_str",
        "side": "sell",
        "volume_bucket": volume_bucket,
        "meta": meta,
    }


def test_safety_override_blocks_when_aggressive_and_0_of_3():
    guard = SafetyOverride({"enabled": True, "apply_to_strategies": ["adaptive_str"]})
    sig = _base_signal(rsi=60.0, candle_close=101.0, resistance_distance_bps=100.0)
    res = guard.check_veto("adaptive_str", sig)
    assert res.is_vetoed is True
    assert res.reason == "safety_override.blocked"
    assert res.meta_data.get("score") == "0/3"
    assert res.meta_data.get("fails") == ["trend_mismatch", "no_volume_confirm", "resistance_far"]


def test_safety_override_allows_when_aggressive_and_2_of_3():
    guard = SafetyOverride({"enabled": True, "apply_to_strategies": ["adaptive_str"]})
    sig = _base_signal(rsi=70.0, candle_open=101.0, candle_close=100.0, resistance_distance_bps=100.0)
    res = guard.check_veto("adaptive_str", sig)
    assert res.is_vetoed is False
    assert res.reason == "safety_override.pass"
    assert res.meta_data.get("score") == "2/3"


def test_safety_override_handles_missing_resistance_as_na():
    guard = SafetyOverride({"enabled": True, "apply_to_strategies": ["adaptive_str"]})
    sig = _base_signal(rsi=70.0, candle_open=101.0, candle_close=100.0, resistance_distance_bps=None)
    res = guard.check_veto("adaptive_str", sig)
    assert res.is_vetoed is False
    assert res.reason == "safety_override.pass"
    assert res.meta_data.get("score") == "2/2"
    assert "resistance_missing" in (res.meta_data.get("na") or [])


def test_safety_override_fail_closed_on_insufficient_context():
    guard = SafetyOverride(
        {
            "enabled": True,
            "apply_to_strategies": ["adaptive_str"],
            "fail_closed_on_insufficient_context": True,
        }
    )
    sig = _base_signal(
        ema21=None,  # makes trend NA
        ema50=None,
        volume=None,  # makes volume NA
        volume_ma20=None,
        resistance_distance_bps=None,  # makes resistance NA
    )
    res = guard.check_veto("adaptive_str", sig)
    assert res.is_vetoed is True
    assert res.reason == "safety_override.insufficient_context"


def test_safety_override_inactive_when_threshold_not_lowered():
    guard = SafetyOverride({"enabled": True, "apply_to_strategies": ["adaptive_str"]})
    sig = _base_signal(base_threshold=68.0, current_threshold=68.0)
    res = guard.check_veto("adaptive_str", sig)
    assert res.is_vetoed is False
    assert res.reason == "safety_override_inactive"
