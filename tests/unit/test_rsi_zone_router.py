from core.rsi_zone_router import RsiZone, RsiZoneSnapshot, is_strategy_allowed, resolve_zone, snapshot_log_context


def _thresholds() -> dict:
    return {"ob_threshold": 35.0, "str_threshold": 65.0, "min_gap_applied": False}


def _router_cfg(
    *,
    override_enabled: bool = False,
    low_side_enabled: bool = True,
    high_side_enabled: bool = False,
    min_penetration: float = 0.0,
) -> dict:
    return {
        "source": {"mode": "consensus"},
        "transition": {
            "width": 5.0,
            "mismatch_extreme_override": {
                "enabled": override_enabled,
                "low_side_enabled": low_side_enabled,
                "high_side_enabled": high_side_enabled,
                "min_penetration": min_penetration,
            },
        },
    }


def test_snapshot_log_context_extracts_expected_fields():
    snapshot = {
        "rsi_slow": 62.345,
        "rsi_fast": 61.0,
        "ob_threshold": 33.2,
        "str_threshold": 66.8,
        "mode": "consensus",
        "meta": {"consensus_status": "aligned"},
    }

    out = snapshot_log_context(snapshot)

    assert out["rsi_level"] == "62.34"
    assert out["rsi_slow"] == "62.34"
    assert out["rsi_fast"] == "61.00"
    assert out["ob_threshold"] == "33.20"
    assert out["str_threshold"] == "66.80"
    assert out["consensus_status"] == "aligned"


def test_snapshot_log_context_falls_back_to_slow_only_mode():
    snapshot = {
        "rsi_slow": 49.9,
        "rsi_fast": None,
        "ob_threshold": 35.0,
        "str_threshold": 65.0,
        "mode": "slow_only",
        "meta": {},
    }

    out = snapshot_log_context(snapshot)

    assert out["consensus_status"] == "slow_only"
    assert out["rsi_level"] == "49.90"


def test_snapshot_log_context_supports_dataclass_snapshot():
    snapshot = RsiZoneSnapshot(
        symbol="BTC/USDT:USDT",
        ts_ms=1,
        rsi_slow=50.0,
        rsi_fast=None,
        mode="consensus",
        ob_threshold=30.0,
        str_threshold=70.0,
        zone=RsiZone.MR,
        transition_width=5.0,
        version="v1",
        meta={"consensus_status": "mismatch_transition"},
    )

    out = snapshot_log_context(snapshot)

    assert out["rsi_level"] == "50.00"
    assert out["rsi_fast"] is None
    assert out["consensus_status"] == "mismatch_transition"


def test_resolve_zone_mismatch_without_override_stays_transition():
    out = resolve_zone(
        symbol="BTC/USDT:USDT",
        rsi_slow=33.0,
        rsi_fast=50.0,
        ts_ms=1,
        thresholds=_thresholds(),
        router_cfg=_router_cfg(override_enabled=False),
    )

    assert out.zone == RsiZone.TRANSITION_LOW
    assert out.meta["consensus_status"] == "mismatch_transition"
    assert out.meta["mismatch_override_applied_side"] is None


def test_resolve_zone_mismatch_low_override_applies_when_penetration_met():
    out = resolve_zone(
        symbol="BTC/USDT:USDT",
        rsi_slow=33.0,
        rsi_fast=50.0,
        ts_ms=1,
        thresholds=_thresholds(),
        router_cfg=_router_cfg(
            override_enabled=True,
            low_side_enabled=True,
            high_side_enabled=False,
            min_penetration=1.0,
        ),
    )

    assert out.zone == RsiZone.OVERSOLD
    assert out.meta["consensus_status"] == "mismatch_extreme_override_low"
    assert out.meta["mismatch_override_applied_side"] == "low"
    assert out.meta["mismatch_override_low_trigger"] is True


def test_resolve_zone_mismatch_low_override_requires_min_penetration():
    out = resolve_zone(
        symbol="BTC/USDT:USDT",
        rsi_slow=34.5,
        rsi_fast=50.0,
        ts_ms=1,
        thresholds=_thresholds(),
        router_cfg=_router_cfg(
            override_enabled=True,
            low_side_enabled=True,
            high_side_enabled=False,
            min_penetration=1.0,
        ),
    )

    assert out.zone == RsiZone.TRANSITION_LOW
    assert out.meta["consensus_status"] == "mismatch_transition"
    assert out.meta["mismatch_override_low_trigger"] is False


def test_resolve_zone_mismatch_high_override_disabled_stays_transition():
    out = resolve_zone(
        symbol="BTC/USDT:USDT",
        rsi_slow=69.0,
        rsi_fast=50.0,
        ts_ms=1,
        thresholds=_thresholds(),
        router_cfg=_router_cfg(
            override_enabled=True,
            low_side_enabled=True,
            high_side_enabled=False,
            min_penetration=1.0,
        ),
    )

    assert out.zone == RsiZone.TRANSITION_HIGH
    assert out.meta["consensus_status"] == "mismatch_transition"
    assert out.meta["mismatch_override_high_trigger"] is False


def test_resolve_zone_mismatch_high_override_applies_when_enabled():
    out = resolve_zone(
        symbol="BTC/USDT:USDT",
        rsi_slow=68.0,
        rsi_fast=50.0,
        ts_ms=1,
        thresholds=_thresholds(),
        router_cfg=_router_cfg(
            override_enabled=True,
            low_side_enabled=False,
            high_side_enabled=True,
            min_penetration=2.0,
        ),
    )

    assert out.zone == RsiZone.OVERBOUGHT
    assert out.meta["consensus_status"] == "mismatch_extreme_override_high"
    assert out.meta["mismatch_override_applied_side"] == "high"
    assert out.meta["mismatch_override_high_trigger"] is True


def test_is_strategy_allowed_mr_allows_between_ob_and_str_thresholds():
    snapshot = {
        "zone": RsiZone.TRANSITION_HIGH.value,
        "rsi_slow": 54.2,
        "ob_threshold": 32.0,
        "str_threshold": 55.0,
        "meta": {"slow_zone": RsiZone.TRANSITION_HIGH.value},
    }
    cfg = {
        "enabled": True,
        "source": {"mode": "consensus", "mr_mode": "follow_source"},
        "transition": {"no_trade_new_entry": True},
    }

    allowed, reason = is_strategy_allowed("mean_reversion", "long", snapshot, cfg)

    assert allowed is True
    assert reason == "rsi_router.allowed"


def test_is_strategy_allowed_ob_still_uses_consensus_zone():
    snapshot = {
        "zone": RsiZone.TRANSITION_LOW.value,
        "rsi_slow": 40.0,
        "ob_threshold": 32.0,
        "str_threshold": 55.0,
        "meta": {"slow_zone": RsiZone.MR.value},
    }
    cfg = {
        "enabled": True,
        "source": {"mode": "consensus", "mr_mode": "slow_only"},
        "transition": {"no_trade_new_entry": True},
    }

    allowed, reason = is_strategy_allowed("adaptive_ob", "long", snapshot, cfg)

    assert allowed is False
    assert reason == "rsi_router.transition_no_trade"


def test_is_strategy_allowed_mr_blocks_when_rsi_at_or_above_str_threshold():
    snapshot = {
        "zone": RsiZone.OVERBOUGHT.value,
        "rsi_slow": 58.0,
        "ob_threshold": 32.0,
        "str_threshold": 55.0,
        "meta": {"slow_zone": RsiZone.OVERBOUGHT.value},
    }
    cfg = {
        "enabled": True,
        "source": {"mode": "consensus", "mr_mode": "follow_source"},
        "transition": {"no_trade_new_entry": True},
    }

    allowed, reason = is_strategy_allowed("mean_reversion", "long", snapshot, cfg)

    assert allowed is False
    assert reason == "rsi_router.zone_mismatch"


def test_is_strategy_allowed_mr_blocks_when_rsi_at_or_below_ob_threshold():
    snapshot = {
        "zone": RsiZone.OVERSOLD.value,
        "rsi_slow": 31.5,
        "ob_threshold": 32.0,
        "str_threshold": 55.0,
        "meta": {"slow_zone": RsiZone.OVERSOLD.value},
    }
    cfg = {
        "enabled": True,
        "source": {"mode": "consensus", "mr_mode": "follow_source"},
        "transition": {"no_trade_new_entry": True},
    }

    allowed, reason = is_strategy_allowed("mr", "long", snapshot, cfg)

    assert allowed is False
    assert reason == "rsi_router.zone_mismatch"


def test_is_strategy_allowed_mr_falls_back_to_legacy_zone_when_thresholds_missing():
    snapshot = {
        "zone": RsiZone.TRANSITION_LOW.value,
        "meta": {"slow_zone": RsiZone.MR.value},
    }
    cfg = {
        "enabled": True,
        "source": {"mode": "consensus"},
        "transition": {"no_trade_new_entry": True},
    }

    allowed, reason = is_strategy_allowed("mr", "long", snapshot, cfg)

    assert allowed is True
    assert reason == "rsi_router.allowed"
