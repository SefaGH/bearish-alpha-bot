import numpy as np
import pandas as pd

from core.key_level_detector import KeyLevelDetector


def _build_wave_df(cycles: int = 6) -> pd.DataFrame:
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


def test_detect_from_df_returns_unknown_for_empty_dataframe():
    detector = KeyLevelDetector()
    levels = detector.detect_from_df(
        symbol="BTC/USDT:USDT",
        timeframe="15m",
        df=pd.DataFrame(),
        price=100.0,
    )
    assert levels.state == "unknown"
    assert levels.nearest_resistance is None
    assert levels.nearest_support is None
    assert levels.position_in_range is None


def test_detect_from_df_finds_support_and_resistance_on_wave_pattern():
    detector = KeyLevelDetector(
        {
            "pivot_left": 1,
            "pivot_right": 1,
            "lookback_bars": 300,
            "min_cluster_n": 1,
            "smc_cluster_pct": 0.02,
            "touch_proximity_bps": 40.0,
        }
    )
    df = _build_wave_df()
    levels = detector.detect_from_df(
        symbol="BTC/USDT:USDT",
        timeframe="15m",
        df=df,
        price=100.0,
    )

    assert levels.state == "ok"
    assert levels.nearest_resistance is not None
    assert levels.nearest_support is not None
    assert levels.nearest_resistance.level > 100.0
    assert levels.nearest_support.level < 100.0
    assert levels.distance_to_resistance_bps is not None and levels.distance_to_resistance_bps > 0
    assert levels.distance_to_support_bps is not None and levels.distance_to_support_bps > 0
    assert levels.position_in_range is not None
    assert 0.0 <= levels.position_in_range <= 1.0
    assert levels.touch_count_resistance > 0
    assert levels.touch_count_support > 0
