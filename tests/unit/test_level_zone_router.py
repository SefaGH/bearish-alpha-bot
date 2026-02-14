import numpy as np
import pandas as pd

from core.level_zone_router import build_level_zone_snapshot, is_strategy_allowed, snapshot_to_dict


def _router_cfg() -> dict:
    return {
        "enabled": True,
        "source": {"mode": "single_tf", "timeframes": ["15m"]},
        "levels": {
            "method": "smc",
            "pivot_left": 1,
            "pivot_right": 1,
            "lookback_bars": 300,
            "band_pct": 0.005,
            "smc_cluster_pct": 0.02,
            "min_cluster_n": 1,
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
        "breakout": {"min_close_bars": 2, "min_volume_mult": 1.2},
    }


def _wave_df(cycles: int = 6) -> pd.DataFrame:
    close = []
    for _ in range(cycles):
        close.extend([100, 102, 104, 102, 100, 98, 96, 98])
    close_arr = np.asarray(close, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(close_arr), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr + 0.5,
            "low": close_arr - 0.5,
            "close": close_arr,
            "volume": np.full(len(close_arr), 100.0),
        },
        index=idx,
    )


def _breakout_up_df() -> pd.DataFrame:
    base = list([100, 102, 104, 102, 100, 98, 96, 98] * 4)
    close = np.asarray(base + [101, 103, 106, 107], dtype=float)
    volume = np.full(len(close), 100.0)
    volume[-2] = 210.0
    volume[-1] = 220.0
    idx = pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_level_zone_router_in_range_allows_mean_reversion():
    cfg = _router_cfg()
    df = _wave_df()
    snap = build_level_zone_snapshot(
        symbol="BTC/USDT:USDT",
        price=100.0,
        dfs_by_timeframe={"15m": df},
        router_cfg=cfg,
    )
    snap_dict = snapshot_to_dict(snap)
    assert isinstance(snap_dict, dict)
    assert snap_dict["zone"] == "IN_RANGE"

    allowed, reason = is_strategy_allowed("mean_reversion", None, snap_dict, cfg)
    assert allowed is True
    assert reason == "level_router.allowed"


def test_level_zone_router_at_level_blocks_new_entry_when_configured():
    cfg = _router_cfg()
    df = _wave_df()
    snap = build_level_zone_snapshot(
        symbol="BTC/USDT:USDT",
        price=104.4,  # near resistance on this synthetic wave
        dfs_by_timeframe={"15m": df},
        router_cfg=cfg,
    )
    snap_dict = snapshot_to_dict(snap)
    assert isinstance(snap_dict, dict)
    assert snap_dict["zone"] == "AT_LEVEL"

    allowed, reason = is_strategy_allowed("adaptive_ob", None, snap_dict, cfg)
    assert allowed is False
    assert reason == "level_router.at_level"


def test_level_zone_router_breakout_up_biases_to_adaptive_ob():
    cfg = _router_cfg()
    df = _breakout_up_df()
    snap = build_level_zone_snapshot(
        symbol="BTC/USDT:USDT",
        price=107.0,
        dfs_by_timeframe={"15m": df},
        router_cfg=cfg,
    )
    snap_dict = snapshot_to_dict(snap)
    assert isinstance(snap_dict, dict)
    assert snap_dict["zone"] == "BREAKOUT_UP_CONFIRMED"

    ob_allowed, _ = is_strategy_allowed("adaptive_ob", None, snap_dict, cfg)
    str_allowed, str_reason = is_strategy_allowed("adaptive_str", None, snap_dict, cfg)
    assert ob_allowed is True
    assert str_allowed is False
    assert str_reason == "level_router.zone_mismatch"


def test_level_zone_router_rollout_observe_returns_would_block_for_canary_symbol():
    cfg = _router_cfg()
    cfg["rollout"] = {"mode": "observe", "canary_symbols": ["BTC/USDT:USDT"]}
    snapshot = {"symbol": "BTC/USDT:USDT", "zone": "AT_LEVEL"}

    allowed, reason = is_strategy_allowed("adaptive_ob", None, snapshot, cfg)

    assert allowed is True
    assert reason == "level_router.observe_would_block"


def test_level_zone_router_rollout_out_of_scope_is_fail_open():
    cfg = _router_cfg()
    cfg["rollout"] = {"mode": "enforce", "canary_symbols": ["BTC/USDT:USDT"]}
    snapshot = {"symbol": "ETH/USDT:USDT", "zone": "AT_LEVEL"}

    allowed, reason = is_strategy_allowed("adaptive_ob", None, snapshot, cfg)

    assert allowed is True
    assert reason == "level_router.rollout_out_of_scope"


def test_level_zone_router_rollout_enforce_blocks_in_scope_canary_symbol():
    cfg = _router_cfg()
    cfg["rollout"] = {"mode": "enforce", "canary_symbols": ["BTC/USDT:USDT"]}
    snapshot = {"symbol": "BTC/USDT:USDT", "zone": "AT_LEVEL"}

    allowed, reason = is_strategy_allowed("adaptive_ob", None, snapshot, cfg)

    assert allowed is False
    assert reason == "level_router.at_level"
