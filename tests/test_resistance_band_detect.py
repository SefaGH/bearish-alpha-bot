import math

import numpy as np
import pandas as pd

from tools.resistance_band_detect import (
    available_pivot_prices,
    choose_k,
    compute_walk_forward_bands,
    confirmed_pivot_highs,
    kmeans_levels,
    nearest_upper_kmeans_band,
    nearest_upper_smc_band,
    timeframe_to_ms,
)


def test_confirmed_pivot_requires_right_window():
    highs = np.array([1, 2, 3, 4, 10, 4, 3, 2, 1], dtype=float)
    piv = confirmed_pivot_highs(highs, left=2, right=2)
    assert bool(piv[4]) is True

    # At t=4, pivot at i=4 is NOT confirmed yet (needs right=2 bars)
    t = 4
    avail = available_pivot_prices(pivot_mask=piv, pivot_prices=highs, t=t, right=2)
    assert avail.size == 0

    # At t=6, pivot at i=4 is confirmed (i <= t-right)
    t = 6
    avail = available_pivot_prices(pivot_mask=piv, pivot_prices=highs, t=t, right=2)
    assert avail.size == 1
    assert float(avail[0]) == 10.0


def test_walk_forward_uses_only_confirmed_pivots():
    highs = np.array([1, 2, 3, 4, 10, 4, 3, 2, 1], dtype=float)
    lows = highs - 0.5
    close = highs - 0.2
    open_ = highs - 0.3
    tf = "5m"
    step = timeframe_to_ms(tf)
    base_ts = 1700000000000
    df = pd.DataFrame(
        {
            "ts_ms": [base_ts + i * step for i in range(len(highs))],
            "open": open_,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": np.ones_like(highs),
        }
    )

    bands = compute_walk_forward_bands(
        df=df,
        timeframe=tf,
        methods=["smc"],
        pivot_left=2,
        pivot_right=2,
        smc_swing_length=50,
        smc_liquidity_range_pct=0.01,
        sr_lookback_bars=100,
        band_mode="pct",
        band_pct=0.01,
        atr_period=3,
        band_atr_mult=0.2,
        smc_cluster_pct=0.005,
        kmin=2,
        kmax=4,
        random_state=1,
        exclude_last=0,
        eval_start_ms=None,
        eval_end_ms=None,
        eval_horizon_bars=0,
    )

    ts_set = {b.ts_ms for b in bands}
    # No band before pivot is confirmed
    assert (base_ts + 4 * step) not in ts_set
    assert (base_ts + 5 * step) not in ts_set
    # Band should exist once pivot becomes confirmed (t=6)
    assert (base_ts + 6 * step) in ts_set


def test_exclude_last_matches_pivot_age_for_last_eval():
    highs = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1, 2, 4, 2, 1, 2, 6, 2, 1], dtype=float)
    # Keep eval price below historical pivot highs so a nearest upper band exists.
    close = np.ones_like(highs, dtype=float)
    exclude_last = 2
    left = 2
    right = 2

    t_full = len(highs) - exclude_last - 1

    piv_full = confirmed_pivot_highs(highs, left=left, right=right)
    pivots_full = available_pivot_prices(pivot_mask=piv_full, pivot_prices=highs, t=t_full, right=right)
    res_full = nearest_upper_smc_band(
        price=float(close[t_full]),
        pivots=pivots_full,
        band_mode="pct",
        band_pct=0.01,
        atr=None,
        band_atr_mult=0.2,
        smc_cluster_pct=0.005,
    )

    trunc_highs = highs[:-exclude_last]
    trunc_close = close[:-exclude_last]
    t_trunc = len(trunc_highs) - 1
    piv_trunc = confirmed_pivot_highs(trunc_highs, left=left, right=right)
    pivots_trunc = available_pivot_prices(pivot_mask=piv_trunc, pivot_prices=trunc_highs, t=t_trunc, right=right)
    res_trunc = nearest_upper_smc_band(
        price=float(trunc_close[t_trunc]),
        pivots=pivots_trunc,
        band_mode="pct",
        band_pct=0.01,
        atr=None,
        band_atr_mult=0.2,
        smc_cluster_pct=0.005,
    )

    assert res_full is not None
    assert res_trunc is not None
    level_full, lo_full, hi_full, _ = res_full
    level_trunc, lo_trunc, hi_trunc, _ = res_trunc
    assert math.isclose(level_full, level_trunc, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(lo_full, lo_trunc, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(hi_full, hi_trunc, rel_tol=0, abs_tol=1e-9)


def test_choose_k_is_clamped():
    assert choose_k(0, kmin=3, kmax=8) == 0
    assert choose_k(2, kmin=3, kmax=8) == 2
    assert choose_k(50, kmin=3, kmax=8) <= 8


def test_kmeans_helpers_handle_small_inputs_without_crashing():
    pivots = np.array([100.0], dtype=float)
    centers_sorted, labels, centers = kmeans_levels(pivots, kmin=3, kmax=8, random_state=1)
    assert centers_sorted.tolist() == [100.0]
    assert centers.tolist() == [100.0]
    assert labels.tolist() == [0]

    res = nearest_upper_kmeans_band(price=99.0, pivots=pivots, band_pct=0.01, kmin=3, kmax=8, random_state=1)
    assert res is not None
    level, band_low, band_high, meta = res
    assert level == 100.0
    assert band_low < level < band_high
    assert meta.get("k") == 1

    assert nearest_upper_kmeans_band(price=101.0, pivots=pivots, band_pct=0.01, kmin=3, kmax=8, random_state=1) is None


def test_smc_lib_liquidity_requires_confirmed_multiple_swing_highs():
    highs = np.array([1, 2, 3, 10, 3, 2, 3, 10.05, 3, 2, 1], dtype=float)
    lows = highs - 0.5
    close = np.ones_like(highs, dtype=float)
    open_ = close.copy()
    tf = "5m"
    step = timeframe_to_ms(tf)
    base_ts = 1700000000000
    df = pd.DataFrame(
        {
            "ts_ms": [base_ts + i * step for i in range(len(highs))],
            "open": open_,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": np.ones_like(highs),
        }
    )

    bands = compute_walk_forward_bands(
        df=df,
        timeframe=tf,
        methods=["smc_lib"],
        pivot_left=2,
        pivot_right=2,
        smc_swing_length=1,
        smc_liquidity_range_pct=0.01,
        sr_lookback_bars=100,
        band_mode="pct",
        band_pct=0.01,
        atr_period=3,
        band_atr_mult=0.2,
        smc_cluster_pct=0.005,
        kmin=2,
        kmax=4,
        random_state=1,
        exclude_last=0,
        eval_start_ms=None,
        eval_end_ms=None,
        eval_horizon_bars=0,
    )

    ts_set = {b.ts_ms for b in bands}
    # Liquidity needs multiple confirmed swing highs: band appears only after the 2nd swing is confirmed.
    assert (base_ts + 7 * step) not in ts_set
    assert (base_ts + 8 * step) in ts_set
