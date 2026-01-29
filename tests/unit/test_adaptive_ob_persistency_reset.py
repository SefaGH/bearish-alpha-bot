import pandas as pd

from strategies.adaptive_ob import AdaptiveOversoldBounce


def _df_30m_with_forming(*, rsi_closed: float, close_closed: float = 90.0) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "open": close_closed + 1.0,
                "high": close_closed + 2.0,
                "low": close_closed - 2.0,
                "close": close_closed,
                "rsi": float(rsi_closed),
                "atr": 1.0,
                "ema_fast": 100.0,
                "ema_mid": 110.0,
                "volume": 1000.0,
            },
            {
                # forming row (content not used when fallback_reason is set)
                "open": close_closed,
                "high": close_closed + 1.0,
                "low": close_closed - 3.0,
                "close": close_closed,
                "rsi": float(rsi_closed),
                "atr": 1.0,
                "ema_fast": 100.0,
                "ema_mid": 110.0,
                "volume": 1000.0,
            },
        ]
    )
    df.attrs["includes_forming"] = True
    df.attrs["forming_ts"] = 1769698800000
    # Force used_forming=False while keeping includes_forming=True so persistency runs on closed RSI.
    df.attrs["fallback_reason"] = "unit_test_force_closed"
    return df


def test_persistency_resets_on_condition_false_and_after_signal():
    cfg = {
        "min_rr_ratio": 0.1,
        "symbols": {"BTC/USDT:USDT": {"rsi_threshold": 27.0}},
        "adaptive_ob_persistency_mode": "time",
        "adaptive_ob_persistency_seconds": 0.0,
        "adaptive_ob_persistency_min_samples": 2,
    }

    strategy = AdaptiveOversoldBounce(cfg)

    df_true = _df_30m_with_forming(rsi_closed=20.0)
    df_false = _df_30m_with_forming(rsi_closed=40.0)

    # A: base condition true -> samples=1 -> persistency not met
    assert strategy.signal(df_30m=df_true, symbol="BTC/USDT:USDT", market_data={}) is None

    # B: base condition false -> should reset persistency immediately
    assert strategy.signal(df_30m=df_false, symbol="BTC/USDT:USDT", market_data={}) is None

    # C: base condition true again -> should behave like first sample again
    assert strategy.signal(df_30m=df_true, symbol="BTC/USDT:USDT", market_data={}) is None

    # D: base condition true -> second sample -> should emit a signal
    signal = strategy.signal(df_30m=df_true, symbol="BTC/USDT:USDT", market_data={})
    assert signal is not None

    # E: base condition still true -> persistency must have been reset after emitting signal
    assert strategy.signal(df_30m=df_true, symbol="BTC/USDT:USDT", market_data={}) is None

