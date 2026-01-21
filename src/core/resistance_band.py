from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Band:
    level: float
    band_low: float
    band_high: float
    method: str
    timeframe: str
    meta: Dict[str, Any]


def _as_closed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a trailing forming candle when present (lookahead-safe)."""
    try:
        includes_forming = bool(getattr(df, "attrs", {}).get("includes_forming", False))
    except Exception:
        includes_forming = False
    if includes_forming and len(df) >= 2:
        return df.iloc[:-1].copy()
    return df


def confirmed_pivot_highs(high: Sequence[float], left: int, right: int) -> np.ndarray:
    """
    Pivot-high definition (offline detection):
      high[i] is a pivot high iff it is strictly greater than all highs in the
      left window and >= all highs in the right window.

    Lookahead-free usage:
      At evaluation index t, only pivots with i <= t - right may be used.
    """
    arr = np.asarray(high, dtype=float)
    n = int(len(arr))
    piv = np.zeros(n, dtype=bool)
    if n == 0 or left < 1 or right < 1:
        return piv
    for i in range(left, n - right):
        lv = arr[i - left : i]
        rv = arr[i + 1 : i + 1 + right]
        if lv.size == 0 or rv.size == 0:
            continue
        if arr[i] > float(np.max(lv)) and arr[i] >= float(np.max(rv)):
            piv[i] = True
    return piv


def confirmed_pivot_lows(low: Sequence[float], left: int, right: int) -> np.ndarray:
    """
    Pivot-low definition:
      low[i] is a pivot low iff it is strictly lower than all lows in the
      left window and <= all lows in the right window.
    """
    arr = np.asarray(low, dtype=float)
    n = int(len(arr))
    piv = np.zeros(n, dtype=bool)
    if n == 0 or left < 1 or right < 1:
        return piv
    for i in range(left, n - right):
        lv = arr[i - left : i]
        rv = arr[i + 1 : i + 1 + right]
        if lv.size == 0 or rv.size == 0:
            continue
        if arr[i] < float(np.min(lv)) and arr[i] <= float(np.min(rv)):
            piv[i] = True
    return piv


def available_pivot_prices(
    *,
    pivot_mask: np.ndarray,
    pivot_prices: np.ndarray,
    t: int,
    right: int,
    lookback_bars: Optional[int] = None,
) -> np.ndarray:
    idx = np.flatnonzero(pivot_mask)
    if idx.size == 0:
        return np.array([], dtype=float)
    confirmed_max = int(t) - int(right)
    idx = idx[idx <= confirmed_max]
    if lookback_bars is not None and int(lookback_bars) > 0:
        idx = idx[idx >= max(0, int(t) - int(lookback_bars))]
    return pivot_prices[idx]


def cluster_1d_pct(values: np.ndarray, pct: float) -> List[np.ndarray]:
    """Simple 1D clustering by relative proximity (sorted scan)."""
    if values.size == 0:
        return []
    vals = np.sort(values.astype(float))
    clusters: List[List[float]] = []
    cur: List[float] = [float(vals[0])]
    for v in vals[1:]:
        center = float(np.median(cur))
        tol = abs(center) * float(pct)
        if abs(float(v) - center) <= tol:
            cur.append(float(v))
        else:
            clusters.append(cur)
            cur = [float(v)]
    clusters.append(cur)
    return [np.asarray(c, dtype=float) for c in clusters]


def choose_k(n_pivots: int, kmin: int, kmax: int) -> int:
    if n_pivots <= 0:
        return 0
    k = int(max(kmin, min(kmax, max(1, n_pivots // 10))))
    return min(k, n_pivots)


def _kmeans_levels(values: np.ndarray, *, kmin: int, kmax: int, random_state: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if values.size == 0:
        return None
    if values.size == 1:
        center = float(values.reshape(-1)[0])
        centers = np.asarray([center], dtype=float)
        labels = np.asarray([0], dtype=int)
        return centers.copy(), labels, centers.copy()

    k = choose_k(int(values.size), kmin=kmin, kmax=kmax)
    if k <= 0:
        return None

    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None

    x = values.reshape(-1, 1).astype(float)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.reshape(-1).astype(float)
    order = np.argsort(centers)
    centers_sorted = centers[order]
    return centers_sorted, labels, centers


def nearest_upper_band(
    *,
    price: float,
    pivots: np.ndarray,
    method: str,
    band_pct: float,
    smc_cluster_pct: float,
    min_cluster_n: int,
    kmin: int,
    kmax: int,
    random_state: int,
) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
    if pivots.size == 0:
        return None

    method = str(method).strip().lower()
    if method == "kmeans":
        res = _kmeans_levels(pivots, kmin=kmin, kmax=kmax, random_state=random_state)
        if not res:
            return None
        centers_sorted, labels, centers = res
        above = centers_sorted[centers_sorted > float(price)]
        if above.size == 0:
            return None
        level = float(np.min(above))
        center_idx = int(np.argmin(np.abs(centers - level)))
        members = pivots[labels == center_idx].astype(float)
        if members.size <= 1:
            width = level * float(band_pct)
            p25 = None
            p75 = None
        else:
            p25 = float(np.percentile(members, 25))
            p75 = float(np.percentile(members, 75))
            width = max((p75 - p25) / 2.0, level * float(band_pct))
        band_low = level - width
        band_high = level + width
        meta = {
            "k": int(len(centers)),
            "cluster_n": int(members.size),
            "band_width": float(width),
            "members_p25": p25,
            "members_p75": p75,
        }
        return float(level), float(band_low), float(band_high), meta

    # default: smc-style clustering
    clusters = cluster_1d_pct(pivots, pct=float(smc_cluster_pct))
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in clusters:
        if int(c.size) < int(min_cluster_n):
            continue
        center = float(np.median(c))
        if center > float(price):
            candidates.append((center, c))
    if not candidates:
        return None

    center, cluster_vals = min(candidates, key=lambda x: x[0] - float(price))
    width = center * float(band_pct)
    band_low = center * (1.0 - float(band_pct))
    band_high = center * (1.0 + float(band_pct))
    meta = {
        "cluster_n": int(cluster_vals.size),
        "cluster_min": float(np.min(cluster_vals)),
        "cluster_max": float(np.max(cluster_vals)),
        "cluster_span": float(np.max(cluster_vals) - np.min(cluster_vals)),
        "band_width": float(width),
    }
    return float(center), float(band_low), float(band_high), meta


def nearest_lower_band(
    *,
    price: float,
    pivots: np.ndarray,
    method: str,
    band_pct: float,
    smc_cluster_pct: float,
    min_cluster_n: int,
    kmin: int,
    kmax: int,
    random_state: int,
) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
    if pivots.size == 0:
        return None

    method = str(method).strip().lower()
    if method == "kmeans":
        res = _kmeans_levels(pivots, kmin=kmin, kmax=kmax, random_state=random_state)
        if not res:
            return None
        centers_sorted, labels, centers = res
        below = centers_sorted[centers_sorted < float(price)]
        if below.size == 0:
            return None
        level = float(np.max(below))
        center_idx = int(np.argmin(np.abs(centers - level)))
        members = pivots[labels == center_idx].astype(float)
        if members.size <= 1:
            width = level * float(band_pct)
            p25 = None
            p75 = None
        else:
            p25 = float(np.percentile(members, 25))
            p75 = float(np.percentile(members, 75))
            width = max((p75 - p25) / 2.0, level * float(band_pct))
        band_low = level - width
        band_high = level + width
        meta = {
            "k": int(len(centers)),
            "cluster_n": int(members.size),
            "band_width": float(width),
            "members_p25": p25,
            "members_p75": p75,
        }
        return float(level), float(band_low), float(band_high), meta

    clusters = cluster_1d_pct(pivots, pct=float(smc_cluster_pct))
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in clusters:
        if int(c.size) < int(min_cluster_n):
            continue
        center = float(np.median(c))
        if center < float(price):
            candidates.append((center, c))
    if not candidates:
        return None

    center, cluster_vals = min(candidates, key=lambda x: float(price) - x[0])
    width = center * float(band_pct)
    band_low = center * (1.0 - float(band_pct))
    band_high = center * (1.0 + float(band_pct))
    meta = {
        "cluster_n": int(cluster_vals.size),
        "cluster_min": float(np.min(cluster_vals)),
        "cluster_max": float(np.max(cluster_vals)),
        "cluster_span": float(np.max(cluster_vals) - np.min(cluster_vals)),
        "band_width": float(width),
    }
    return float(center), float(band_low), float(band_high), meta


def compute_band(
    *,
    df: pd.DataFrame,
    timeframe: str,
    price: float,
    side: str,
    method: str,
    pivot_left: int,
    pivot_right: int,
    lookback_bars: int,
    band_pct: float,
    smc_cluster_pct: float,
    min_cluster_n: int,
    kmin: int,
    kmax: int,
    random_state: int,
) -> Optional[Band]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df_closed = _as_closed_df(df)
    if df_closed.empty:
        return None

    for col in ("high", "low", "close"):
        if col not in df_closed.columns:
            return None

    highs = df_closed["high"].astype(float).to_numpy()
    lows = df_closed["low"].astype(float).to_numpy()
    t = int(len(df_closed) - 1)
    if t <= 0:
        return None

    side_n = str(side or "").strip().lower()
    is_long = side_n in {"buy", "long"}

    if is_long:
        piv_mask = confirmed_pivot_highs(highs, left=int(pivot_left), right=int(pivot_right))
        pivots = available_pivot_prices(
            pivot_mask=piv_mask,
            pivot_prices=highs,
            t=t,
            right=int(pivot_right),
            lookback_bars=int(lookback_bars),
        )
        res = nearest_upper_band(
            price=float(price),
            pivots=pivots,
            method=method,
            band_pct=float(band_pct),
            smc_cluster_pct=float(smc_cluster_pct),
            min_cluster_n=int(min_cluster_n),
            kmin=int(kmin),
            kmax=int(kmax),
            random_state=int(random_state),
        )
    else:
        piv_mask = confirmed_pivot_lows(lows, left=int(pivot_left), right=int(pivot_right))
        pivots = available_pivot_prices(
            pivot_mask=piv_mask,
            pivot_prices=lows,
            t=t,
            right=int(pivot_right),
            lookback_bars=int(lookback_bars),
        )
        res = nearest_lower_band(
            price=float(price),
            pivots=pivots,
            method=method,
            band_pct=float(band_pct),
            smc_cluster_pct=float(smc_cluster_pct),
            min_cluster_n=int(min_cluster_n),
            kmin=int(kmin),
            kmax=int(kmax),
            random_state=int(random_state),
        )

    if not res:
        return None
    level, band_low, band_high, meta = res
    if not (math.isfinite(level) and math.isfinite(band_low) and math.isfinite(band_high)):
        return None
    if band_high < band_low:
        band_low, band_high = band_high, band_low
    return Band(
        level=float(level),
        band_low=float(band_low),
        band_high=float(band_high),
        method=str(method).strip().lower(),
        timeframe=str(timeframe),
        meta=meta or {},
    )
