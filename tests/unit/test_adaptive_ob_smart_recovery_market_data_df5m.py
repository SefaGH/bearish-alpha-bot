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
    }


def test_smart_recovery_tp_accepts_df5m_market_data_without_dataframe_truthiness_error():
    """Regression: avoid `ValueError: The truth value of a DataFrame is ambiguous`.

    This used to happen when selecting df_5m via:
      market_data.get("5m") or market_data.get("df_5m")
    because pandas DataFrames do not support truth-value testing.
    """

    strategy = AdaptiveOversoldBounce(_base_cfg())

    # Minimal 5m frame (content doesn't matter for the regression as long as it's a DataFrame).
    df_5m = pd.DataFrame(
        {
            "open": [100.0] * 30,
            "high": [101.0 + i * 0.1 for i in range(30)],
            "low": [99.0 - i * 0.05 for i in range(30)],
            "close": [100.0 + i * 0.02 for i in range(30)],
            "volume": [1000.0] * 30,
        }
    )

    tp, meta = strategy._calculate_smart_recovery_tp(
        symbol="BTC/USDT:USDT",
        entry_price=100.0,
        stop_price=98.0,
        atr_value=1.0,
        min_tp_pct=0.005,
        baseline_target_price=101.0,
        current_target_price=101.2,
        tp_band_meta=None,
        market_data={"5m": df_5m},
        cfg={
            "candidates": {"include_pivot": True, "include_band": False},
            "reachability": {"max_atr_mult": 10.0},
            "priority": {"fibo": 0, "pivot": 1, "atr": 0, "band": 0},
            "barrier": {"penalty_points": 0},
        },
        crash_leg=None,
        triggers={"active": True, "reasons": ["test"]},
    )

    assert isinstance(meta, dict)
    assert meta.get("enabled") is True
    # We don't assert on tp selection specifics here; the test is about not crashing.
