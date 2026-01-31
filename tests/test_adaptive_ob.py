import copy

import pytest
import pandas as pd

from strategies.adaptive_ob import AdaptiveOversoldBounce


def _base_cfg():
    return {
        "min_rr_ratio": 1.5,
        "adaptive_rsi_base": 32.0,
        "adaptive_rsi_range": 8.0,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.2,
        "min_tp_pct": 0.008,
        "max_sl_pct": 0.015,
        "trend_confirmation_rsi_penalty": 5.0,
        "trend_confirmation_min_rsi": 8.0,
    }


def _df_30m(*, close: float, rsi: float, atr: float, ema_fast: float, ema_mid: float):
    return pd.DataFrame(
        [
            {
                "close": close,
                "rsi": rsi,
                "atr": atr,
                "ema_fast": ema_fast,
                "ema_mid": ema_mid,
                "volume": 1000.0,
            }
        ]
    )


def _df_30m_with_adx(*, close: float, rsi: float, atr: float, ema_fast: float, ema_mid: float, adx: float):
    df = _df_30m(close=close, rsi=rsi, atr=atr, ema_fast=ema_fast, ema_mid=ema_mid)
    df["adx"] = float(adx)
    return df


def _df_15m(*, prev_close: float, close: float, rsi: float):
    return pd.DataFrame(
        [
            {"close": prev_close, "rsi": rsi},
            {"close": close, "rsi": rsi},
        ]
    )


def test_trend_penalty_blocks_without_extreme_bypass():
    cfg = _base_cfg()
    cfg["extreme_bypass"] = {
        "enabled": False,
        "triggers": {
            "price_drop_15m_pct": 0.8,
            "rsi_15m_below": 30.0,
            "min_atr_pct": 0.006,
        },
    }

    strategy = AdaptiveOversoldBounce(cfg)
    df_30m = _df_30m(close=90.0, rsi=30.0, atr=1.0, ema_fast=100.0, ema_mid=110.0)
    df_15m = _df_15m(prev_close=100.0, close=99.0, rsi=25.0)

    signal = strategy.signal(df_30m=df_30m, symbol="BTC/USDT:USDT", market_data={"15m": df_15m})
    assert signal is None


def test_extreme_bypass_skips_trend_penalty_and_allows_signal():
    cfg = _base_cfg()
    cfg["extreme_bypass"] = {
        "enabled": True,
        "triggers": {
            "price_drop_15m_pct": 0.8,
            "rsi_15m_below": 30.0,
            "min_atr_pct": 0.006,
        },
    }

    strategy = AdaptiveOversoldBounce(cfg)
    df_30m = _df_30m(close=90.0, rsi=30.0, atr=1.0, ema_fast=100.0, ema_mid=110.0)
    df_15m = _df_15m(prev_close=100.0, close=99.0, rsi=25.0)

    signal = strategy.signal(df_30m=df_30m, symbol="BTC/USDT:USDT", market_data={"15m": df_15m})
    assert signal is not None
    assert signal.get("extreme_bypass") is True
    assert "extreme_bypass_meta" in signal


def test_extreme_bypass_ignores_ml_veto():
    cfg = _base_cfg()
    cfg["extreme_bypass"] = {
        "enabled": True,
        "triggers": {
            "price_drop_15m_pct": 0.8,
            "rsi_15m_below": 30.0,
            "min_atr_pct": 0.006,
        },
    }

    strategy = AdaptiveOversoldBounce(copy.deepcopy(cfg))
    df_30m = _df_30m(close=90.0, rsi=30.0, atr=1.0, ema_fast=100.0, ema_mid=110.0)
    df_15m = _df_15m(prev_close=100.0, close=99.0, rsi=25.0)
    ml_context = {
        "is_healthy": True,
        "regime_confidence": 0.80,
        "regime_prediction": "bearish",
        "price_direction": "down",
        "price_confidence": 0.90,
        "consensus_score": 0.75,
    }

    signal = strategy.signal(
        df_30m=df_30m,
        symbol="BTC/USDT:USDT",
        market_data={"15m": df_15m},
        ml_context=ml_context,
    )
    assert signal is not None
    assert signal.get("extreme_bypass") is True


def test_extreme_bypass_hard_veto_blocks_when_adx_high_and_no_reclaim():
    cfg = _base_cfg()
    cfg["extreme_bypass"] = {
        "enabled": True,
        "triggers": {
            "price_drop_15m_pct": 0.8,
            "rsi_15m_below": 30.0,
            "min_atr_pct": 0.006,
        },
    }

    strategy = AdaptiveOversoldBounce(cfg)
    # ADX high and price below EMA fast -> no reclaim -> hard veto
    df_30m = _df_30m_with_adx(close=90.0, rsi=30.0, atr=1.0, ema_fast=100.0, ema_mid=110.0, adx=55.0)
    df_15m = _df_15m(prev_close=100.0, close=99.0, rsi=10.0)  # 1.0% drop, RSI extreme

    signal = strategy.signal(df_30m=df_30m, symbol="BTC/USDT:USDT", market_data={"15m": df_15m})
    assert signal is None


def test_extreme_bypass_soft_penalty_forces_scalp_and_reduces_position_multiplier():
    cfg = _base_cfg()
    cfg["extreme_bypass"] = {
        "enabled": True,
        "triggers": {
            "price_drop_15m_pct": 0.8,
            "rsi_15m_below": 30.0,
            "min_atr_pct": 0.006,
        },
    }

    strategy = AdaptiveOversoldBounce(copy.deepcopy(cfg))
    df_15m = _df_15m(prev_close=100.0, close=99.0, rsi=25.0)

    # Baseline (no penalty): ADX low
    meta_base = {}
    out_base = strategy._filter_extreme_bypass_signal(
        symbol="BTC/USDT:USDT",
        log_prefix="[TEST]",
        extreme_bypass_meta=meta_base,
        trigger_price=90.0,
        ema_fast=100.0,
        adx_val=20.0,
        market_data={"15m": df_15m},
    )
    assert out_base.get("veto") is False
    assert float(out_base.get("size_multiplier") or 0.0) == pytest.approx(1.0, rel=1e-9)
    assert out_base.get("force_scalp_mode") is False

    # Penalty case: ADX mid-range (30-50) -> size*0.5 + force_scalp_mode
    meta_pen = {}
    out_pen = strategy._filter_extreme_bypass_signal(
        symbol="BTC/USDT:USDT",
        log_prefix="[TEST]",
        extreme_bypass_meta=meta_pen,
        trigger_price=90.0,
        ema_fast=100.0,
        adx_val=40.0,
        market_data={"15m": df_15m},
    )
    assert out_pen.get("veto") is False
    assert float(out_pen.get("size_multiplier") or 0.0) == pytest.approx(0.5, rel=1e-9)
    assert out_pen.get("force_scalp_mode") is True
    assert "adx_mid" in (out_pen.get("penalty_reasons") or [])

    # Also persisted to meta for downstream auditability.
    assert meta_pen.get("filter", {}).get("force_scalp_mode") is True
    assert float(meta_pen.get("filter", {}).get("size_multiplier") or 0.0) == pytest.approx(0.5, rel=1e-9)
