import pandas as pd
import pytest

from src.strategies.mean_reversion import VWAPMeanReversion


def _build_frames(
    *,
    close: float,
    adx: float,
    lower: float = 98.0,
    upper: float = 102.1,
    vwap: float = 100.0,
    vwap_std: float = 1.0,
    periods: int = 2,
    volumes: list[float] | None = None,
):
    idx = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    if volumes is None:
        volumes = [1.0] * periods
    if len(volumes) != periods:
        raise ValueError("volumes length must match periods")
    df_vwap = pd.DataFrame(
        {
            "close": [close] * periods,
            "volume": [1.0] * periods,
            "vwap": [vwap] * periods,
            "vwap_lower": [lower] * periods,
            "vwap_upper": [upper] * periods,
            "vwap_std": [vwap_std] * periods,
        },
        index=idx,
    )
    df_sig = pd.DataFrame(
        {
            "close": [close] * periods,
            "adx": [adx] * periods,
            "volume": volumes,
        },
        index=idx,
    )
    return df_vwap, df_sig


def _base_cfg() -> dict:
    return {
        "timeframe": "1m",
        "signal_timeframe": "5m",
        "min_rows": 2,
        "min_signal_rows": 2,
        "dynamic_controller": {"enabled": False},
        "rsi_rebound_guard": {"enabled": False},
        "soft_deferral_threshold": 0.0,
    }


@pytest.mark.asyncio
async def test_promote_override_does_not_bypass_adx_veto():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 20.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 40.0,  # allow PROMOTE; ADX veto must still block entry
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(close=102.05, adx=30.0)

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        condition_data={"near": "upper", "trigger_price": 102.1, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 102.05, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
    )

    assert out is None


@pytest.mark.asyncio
async def test_promote_override_uses_abs_for_signed_dist_to_band_bps():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(close=102.05, adx=10.0)

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        condition_data={"near": "upper", "trigger_price": 102.1, "eps_bps": 10},
        # Signed/negative distance should be rejected by abs(dist) gate.
        check_detail={"fast_watch": {"price": 102.05, "touch_confirmed": True, "dist_to_band_bps": -100.0}},
    )

    assert isinstance(out, dict)
    assert out.get("event_type") == "strategy_recheck_decision"


@pytest.mark.asyncio
async def test_promote_override_uses_shock_state_kwarg():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(close=102.05, adx=10.0)

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        condition_data={"near": "upper", "trigger_price": 102.1, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 102.05, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
        regime_data={"trend": "neutral"},
        shock_state="ARMED",
    )

    assert isinstance(out, dict)
    assert out.get("event_type") == "strategy_recheck_decision"


@pytest.mark.asyncio
async def test_promote_override_volume_prefers_upstream_context():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
            "min_volume_strength": 0.50,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    # Local fallback would be LOW, but upstream must win.
    df_vwap, df_sig = _build_frames(
        close=100.2,
        adx=10.0,
        lower=99.0,
        upper=101.0,
        vwap=100.0,
        vwap_std=0.1,
        periods=10,
        volumes=[100, 100, 100, 100, 100, 100, 100, 5, 5, 5],
    )

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        side="short",
        condition_data={"near": "upper", "trigger_price": 101.0, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 100.2, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
        volume_strength=0.90,
        volume_bucket="HIGH",
        volume_source="analyzer",
    )

    assert isinstance(out, dict)
    assert out.get("side") == "sell"
    va = out.get("meta", {}).get("volume_analysis", {})
    assert va.get("source") == "analyzer"
    assert va.get("volume_strength") == pytest.approx(0.90)


@pytest.mark.asyncio
async def test_promote_override_observe_mode_keeps_recheck_legacy_hold():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "observe",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(
        close=100.2,
        adx=10.0,
        lower=99.0,
        upper=101.0,
        vwap=100.0,
        vwap_std=0.1,
        periods=10,
        volumes=[100] * 10,
    )

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        side="short",
        condition_data={"near": "upper", "trigger_price": 101.0, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 100.2, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
    )

    assert isinstance(out, dict)
    assert out.get("event_type") == "strategy_recheck_decision"
    po = out.get("decision_meta", {}).get("promotion_override", {})
    assert po.get("configured_mode") == "observe"
    assert po.get("candidate") is True
    assert po.get("applied") is False
    assert po.get("scope_reason") == "observe_only"


@pytest.mark.asyncio
async def test_promote_override_observe_mode_canary_symbol_applies():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "observe",
            "canary_symbols": ["BTC/USDT:USDT"],
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(
        close=100.2,
        adx=10.0,
        lower=99.0,
        upper=101.0,
        vwap=100.0,
        vwap_std=0.1,
        periods=10,
        volumes=[100] * 10,
    )

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        side="short",
        condition_data={"near": "upper", "trigger_price": 101.0, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 100.2, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
    )

    assert isinstance(out, dict)
    assert out.get("side") == "sell"
    po = out.get("meta", {}).get("promotion_override", {})
    assert po.get("configured_mode") == "observe"
    assert po.get("candidate") is True
    assert po.get("applied") is True
    assert po.get("scope_reason") == "canary_symbol"
    assert po.get("near") == "upper"
    assert po.get("touch_confirmed") is True
    assert po.get("dist_bps") == pytest.approx(1.0)
    assert po.get("z") is not None
    assert po.get("adx") == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_promote_override_volume_uses_local_fallback_when_upstream_missing():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
            "min_volume_strength": 0.50,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    # Recent volume collapse => local fallback should classify as LOW and block PROMOTE.
    df_vwap, df_sig = _build_frames(
        close=100.2,
        adx=10.0,
        lower=99.0,
        upper=101.0,
        vwap=100.0,
        vwap_std=0.1,
        periods=10,
        volumes=[100, 100, 100, 100, 100, 100, 100, 5, 5, 5],
    )

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        side="short",
        condition_data={"near": "upper", "trigger_price": 101.0, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 100.2, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
    )

    assert isinstance(out, dict)
    assert out.get("event_type") == "strategy_recheck_decision"
    assert out.get("decision_meta", {}).get("action") == "HOLD"


def test_get_ema_stack_extracts_from_df_sig_columns():
    strategy = VWAPMeanReversion(_base_cfg())
    _, df_sig = _build_frames(close=100.0, adx=10.0, periods=6, volumes=[1, 1, 1, 1, 1, 1])
    df_sig["ema21"] = [99, 100, 101, 102, 103, 104]
    df_sig["ema50"] = [98, 99, 100, 101, 102, 103]
    df_sig["ema200"] = [97, 98, 99, 100, 101, 102]

    ema_stack = strategy._get_ema_stack(df_sig=df_sig, market_data=None, kwargs={})

    assert ema_stack is not None
    assert ema_stack.get("ema21") == pytest.approx(104.0)
    assert ema_stack.get("ema50") == pytest.approx(103.0)
    assert ema_stack.get("ema200") == pytest.approx(102.0)


@pytest.mark.asyncio
async def test_promote_override_trend_veto_uses_auto_ema_stack_from_df_sig():
    cfg = _base_cfg()
    cfg["adx_threshold"] = 40.0
    cfg["fast_watch"] = {
        "promote_override": {
            "enabled": True,
            "mode": "enforce",
            "min_z_score": 2.0,
            "max_dist_bps": 2.0,
            "max_adx": 20.0,
            "respect_trend_veto": True,
        }
    }
    strategy = VWAPMeanReversion(cfg)
    df_vwap, df_sig = _build_frames(
        close=100.2,
        adx=10.0,
        lower=99.0,
        upper=101.0,
        vwap=100.0,
        vwap_std=0.1,
        periods=10,
        volumes=[100] * 10,
    )
    # Bullish EMA stack should veto MR short PROMOTE for near=upper.
    df_sig["ema21"] = [100.0] * 10
    df_sig["ema50"] = [99.0] * 10
    df_sig["ema200"] = [98.0] * 10

    out = await strategy.generate_signal(
        symbol="BTC/USDT:USDT",
        df_vwap=df_vwap,
        df_sig=df_sig,
        parent_pending_id="pending-1",
        side="short",
        condition_data={"near": "upper", "trigger_price": 101.0, "eps_bps": 10},
        check_detail={"fast_watch": {"price": 100.2, "touch_confirmed": True, "dist_to_band_bps": 1.0}},
        regime_data={"trend": "neutral"},
    )

    assert isinstance(out, dict)
    assert out.get("event_type") == "strategy_recheck_decision"
    assert out.get("decision_meta", {}).get("action") == "HOLD"
