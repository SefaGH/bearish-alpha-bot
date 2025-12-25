import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategies.adaptive_ob import AdaptiveOversoldBounce


def _make_minimal_hybrid_df() -> pd.DataFrame:
    # Two 30m bars: last one is the forming candle
    t0 = datetime(2025, 12, 25, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=30)
    idx = pd.DatetimeIndex([t0, t1])

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 5.0],
            # Required indicator columns (strategy validates presence/non-NaN on last CLOSED row)
            "rsi": [25.0, 30.0],
            "atr": [1.0, 1.0],
            "ema_fast": [150.0, 150.0],
            "ema_mid": [160.0, 160.0],
        },
        index=idx,
    )

    forming_ts_ms = int(t1.timestamp() * 1000)
    df.attrs["includes_forming"] = True
    df.attrs["fallback_reason"] = None
    df.attrs["forming_ts"] = forming_ts_ms
    df.attrs["forming_last_update_ts"] = forming_ts_ms
    df.attrs["forming_update_age_ms"] = 250

    return df


def test_adaptive_ob_hybrid_none_fallback_does_not_warn_and_uses_forming(caplog):
    strategy = AdaptiveOversoldBounce(cfg={"debug": {"strategy_logging": False}})
    df = _make_minimal_hybrid_df()

    caplog.set_level(logging.INFO)

    # We only care about hybrid intake/logging behavior; signal output can be None.
    strategy.signal(df_30m=df, df_1h=None, regime_data=None, symbol="BTC/USDT:USDT")

    # Must NOT warn when includes_forming=True and fallback_reason=None.
    assert "Hybrid fallback:" not in caplog.text

    # Must explicitly report that forming data was used.
    assert "Hybrid meta" in caplog.text
    assert "used_forming=True" in caplog.text

    # For readability, None is rendered as 'none' in logs.
    assert "fallback_reason=none" in caplog.text


def test_adaptive_ob_hybrid_real_fallback_forces_used_forming_false_and_warns(caplog):
    strategy = AdaptiveOversoldBounce(cfg={"debug": {"strategy_logging": False}})
    df = _make_minimal_hybrid_df()

    # Simulate a real pipeline fallback while still carrying includes_forming=True.
    df.attrs["fallback_reason"] = "forming_update_stale"

    caplog.set_level(logging.INFO)

    strategy.signal(df_30m=df, df_1h=None, regime_data=None, symbol="BTC/USDT:USDT")

    # Must warn and must indicate we reverted to closed-only.
    assert "Hybrid fallback: forming_update_stale. Reverting to closed-only data." in caplog.text

    # Must explicitly report that forming data was NOT used.
    assert "Hybrid meta" in caplog.text
    assert "used_forming=False" in caplog.text
