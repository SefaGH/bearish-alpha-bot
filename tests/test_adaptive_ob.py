import copy

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
