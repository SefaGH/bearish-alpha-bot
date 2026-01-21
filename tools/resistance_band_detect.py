#!/usr/bin/env python3
"""
Resistance band detection (offline + live-sim) with lookahead-safe walk-forward.

Methods:
  - SMC-style liquidity highs (confirmed pivot-high bands)
  - KMeans clustering on confirmed pivot-high prices

Production code is not modified; this is offline/diagnostic tooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Avoid noisy joblib warnings in constrained environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _dt_utc_from_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _sanitize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")


def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def timeframe_to_ms(tf: str) -> int:
    return timeframe_to_minutes(tf) * 60_000


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def confirmed_pivot_highs(high: Sequence[float], left: int, right: int) -> np.ndarray:
    """
    Pivot-high definition (offline detection):
      high[i] is a pivot high iff it is strictly greater than all highs in the
      left window and >= all highs in the right window.

    Lookahead-free usage:
      At evaluation index t, only pivots with i <= t - right may be used.
    """
    arr = np.asarray(high, dtype=float)
    n = len(arr)
    piv = np.zeros(n, dtype=bool)
    if n == 0 or left < 1 or right < 1:
        return piv
    for i in range(left, n - right):
        lv = arr[i - left : i]
        rv = arr[i + 1 : i + 1 + right]
        if not len(lv) or not len(rv):
            continue
        if arr[i] > float(np.max(lv)) and arr[i] >= float(np.max(rv)):
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
    confirmed_max = t - right
    idx = idx[idx <= confirmed_max]
    if lookback_bars is not None and lookback_bars > 0:
        idx = idx[idx >= max(0, t - lookback_bars)]
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
        tol = abs(center) * pct
        if abs(float(v) - center) <= tol:
            cur.append(float(v))
        else:
            clusters.append(cur)
            cur = [float(v)]
    clusters.append(cur)
    return [np.asarray(c, dtype=float) for c in clusters]


@dataclass(frozen=True)
class Band:
    level: float
    band_low: float
    band_high: float
    method: str
    timeframe: str
    ts_ms: int
    price: float
    meta: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        dist = (self.level - self.price) / (self.price + 1e-12)
        return {
            "timestamp": datetime.fromtimestamp(self.ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "ts_ms": self.ts_ms,
            "timeframe": self.timeframe,
            "method": self.method,
            "price": self.price,
            "nearest_res_level": self.level,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "distance_pct_to_level": dist,
            "distance_pct_to_band_low": (self.band_low - self.price) / (self.price + 1e-12),
            "distance_pct_to_band_high": (self.band_high - self.price) / (self.price + 1e-12),
            "is_above_price": bool(self.level > self.price),
            **{f"meta_{k}": v for k, v in (self.meta or {}).items()},
        }


def choose_k(n_pivots: int, kmin: int, kmax: int) -> int:
    if n_pivots <= 0:
        return 0
    k = int(max(kmin, min(kmax, max(1, n_pivots // 10))))
    return min(k, n_pivots)


def kmeans_levels(values: np.ndarray, *, kmin: int, kmax: int, random_state: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (centers_sorted, labels, centers_unsorted) or None if insufficient pivots."""
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
    from sklearn.cluster import KMeans

    x = values.reshape(-1, 1).astype(float)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_.reshape(-1).astype(float)
    order = np.argsort(centers)
    centers_sorted = centers[order]
    return centers_sorted, labels, centers


def nearest_upper_smc_band(
    *,
    price: float,
    pivots: np.ndarray,
    band_mode: str,
    band_pct: float,
    atr: Optional[float],
    band_atr_mult: float,
    smc_cluster_pct: float,
    min_cluster_n: int = 1,
) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
    if pivots.size == 0:
        return None

    clusters = cluster_1d_pct(pivots, pct=smc_cluster_pct)
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in clusters:
        if int(c.size) < int(min_cluster_n):
            continue
        center = float(np.median(c))
        if center > price:
            candidates.append((center, c))
    if not candidates:
        return None

    center, cluster_vals = min(candidates, key=lambda x: x[0] - price)
    if band_mode == "atr":
        if atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
            width = center * band_pct
            mode_eff = "pct_fallback"
        else:
            width = float(atr) * band_atr_mult
            mode_eff = "atr"
        band_low = center - width
        band_high = center + width
    else:
        width = center * band_pct
        band_low = center * (1.0 - band_pct)
        band_high = center * (1.0 + band_pct)
        mode_eff = "pct"

    meta = {
        "cluster_n": int(cluster_vals.size),
        "cluster_min": float(np.min(cluster_vals)),
        "cluster_max": float(np.max(cluster_vals)),
        "cluster_span": float(np.max(cluster_vals) - np.min(cluster_vals)),
        "band_mode": mode_eff,
        "band_width": float(width),
    }
    return float(center), float(band_low), float(band_high), meta


def nearest_upper_kmeans_band(
    *,
    price: float,
    pivots: np.ndarray,
    band_pct: float,
    kmin: int,
    kmax: int,
    random_state: int,
) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
    if pivots.size == 0:
        return None
    if pivots.size == 1:
        level = float(pivots.reshape(-1)[0])
        if level <= price:
            return None
        width = level * band_pct
        meta = {
            "k": 1,
            "cluster_n": 1,
            "band_width": float(width),
            "members_p25": level,
            "members_p75": level,
        }
        return float(level), float(level - width), float(level + width), meta
    res = kmeans_levels(pivots, kmin=kmin, kmax=kmax, random_state=random_state)
    if not res:
        return None
    centers_sorted, labels, centers = res
    above = centers_sorted[centers_sorted > price]
    if above.size == 0:
        return None
    level = float(np.min(above))

    # Identify the cluster (centers is unsorted)
    center_idx = int(np.argmin(np.abs(centers - level)))
    members = pivots[labels == center_idx].astype(float)
    if members.size <= 1:
        width = level * band_pct
        p25 = None
        p75 = None
    else:
        p25 = float(np.percentile(members, 25))
        p75 = float(np.percentile(members, 75))
        width = max((p75 - p25) / 2.0, level * band_pct)

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


def compute_walk_forward_bands(
    *,
    df: pd.DataFrame,
    timeframe: str,
    methods: Sequence[str],
    pivot_left: int,
    pivot_right: int,
    smc_swing_length: int,
    smc_liquidity_range_pct: float,
    sr_lookback_bars: int,
    band_mode: str,
    band_pct: float,
    atr_period: int,
    band_atr_mult: float,
    smc_cluster_pct: float,
    kmin: int,
    kmax: int,
    random_state: int,
    exclude_last: int,
    eval_start_ms: Optional[int],
    eval_end_ms: Optional[int],
    eval_horizon_bars: int,
) -> List[Band]:
    methods = list(dict.fromkeys(list(methods)))
    if df.empty:
        return []

    df_eval = df.sort_values("ts_ms").reset_index(drop=True).copy()

    df_eval["atr"] = compute_atr(df_eval, period=atr_period)

    highs = df_eval["high"].to_numpy(dtype=float)
    piv_mask = confirmed_pivot_highs(highs, left=pivot_left, right=pivot_right)
    piv_mask_lib = (
        confirmed_pivot_highs(highs, left=int(smc_swing_length), right=int(smc_swing_length))
        if "smc_lib" in set(methods)
        else None
    )

    ts = df_eval["ts_ms"].to_numpy(dtype=np.int64)
    closes = df_eval["close"].to_numpy(dtype=float)
    atrs = df_eval["atr"].to_numpy(dtype=float)

    idxs = np.arange(len(df_eval), dtype=int)
    if eval_start_ms is not None:
        idxs = idxs[ts[idxs] >= int(eval_start_ms)]
    if eval_end_ms is not None:
        idxs = idxs[ts[idxs] <= int(eval_end_ms)]

    out: List[Band] = []
    for t in idxs:
        price = float(closes[t])
        confirm_delay = max(int(pivot_right), int(exclude_last or 0))
        pivots = available_pivot_prices(
            pivot_mask=piv_mask,
            pivot_prices=highs,
            t=int(t),
            right=confirm_delay,
            lookback_bars=sr_lookback_bars,
        )
        pivots_lib: Optional[np.ndarray]
        if piv_mask_lib is None:
            pivots_lib = None
        else:
            confirm_delay_lib = max(int(smc_swing_length), int(exclude_last or 0))
            pivots_lib = available_pivot_prices(
                pivot_mask=piv_mask_lib,
                pivot_prices=highs,
                t=int(t),
                right=confirm_delay_lib,
                lookback_bars=sr_lookback_bars,
            )
        atr_val = float(atrs[t]) if math.isfinite(float(atrs[t])) else None

        for method in methods:
            if method == "smc":
                res = nearest_upper_smc_band(
                    price=price,
                    pivots=pivots,
                    band_mode=band_mode,
                    band_pct=band_pct,
                    atr=atr_val,
                    band_atr_mult=band_atr_mult,
                    smc_cluster_pct=smc_cluster_pct,
                )
            elif method == "kmeans":
                res = nearest_upper_kmeans_band(
                    price=price,
                    pivots=pivots,
                    band_pct=band_pct,
                    kmin=kmin,
                    kmax=kmax,
                    random_state=random_state,
                )
            elif method == "smc_lib":
                piv = pivots_lib if pivots_lib is not None else np.array([], dtype=float)
                res = nearest_upper_smc_band(
                    price=price,
                    pivots=piv,
                    band_mode=band_mode,
                    band_pct=band_pct,
                    atr=atr_val,
                    band_atr_mult=band_atr_mult,
                    smc_cluster_pct=float(smc_liquidity_range_pct),
                    min_cluster_n=2,
                )
            else:
                continue

            if not res:
                continue
            level, band_low, band_high, meta = res
            if method == "smc_lib":
                meta = dict(meta or {})
                meta["smc_swing_length"] = int(smc_swing_length)
                meta["smc_liquidity_range_pct"] = float(smc_liquidity_range_pct)

            if eval_horizon_bars and eval_horizon_bars > 0:
                horizon_end = min(len(df_eval), t + 1 + int(eval_horizon_bars))
                future_high = df_eval["high"].iloc[t + 1 : horizon_end].max() if horizon_end > (t + 1) else float("nan")
                meta = dict(meta or {})
                meta["eval_horizon_bars"] = int(eval_horizon_bars)
                meta["future_high_max"] = float(future_high) if pd.notna(future_high) else None
                if pd.notna(future_high):
                    meta["touch_level_within_h"] = bool(float(future_high) >= float(level))
                    meta["touch_band_low_within_h"] = bool(float(future_high) >= float(band_low))

            out.append(
                Band(
                    level=float(level),
                    band_low=float(band_low),
                    band_high=float(band_high),
                    method=method,
                    timeframe=timeframe,
                    ts_ms=int(ts[t]),
                    price=price,
                    meta=meta or {},
                )
            )

    return out


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_cached_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "ts_ms" not in df.columns:
        return None
    df["ts_ms"] = df["ts_ms"].astype(np.int64)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"]).reset_index(drop=True)


def save_cached_ohlcv(path: Path, df: pd.DataFrame) -> None:
    df = df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def fetch_ohlcv_ccxt(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int],
    end_ms: Optional[int],
    limit: int = 1000,
    verbose: bool = False,
) -> pd.DataFrame:
    import ccxt

    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})

    # Best-effort: swap default for swap-like symbols
    try:
        if ":" in symbol:
            ex.options = dict(getattr(ex, "options", {}) or {})
            ex.options["defaultType"] = ex.options.get("defaultType", "swap")
    except Exception:
        pass

    resolved_symbol = symbol
    try:
        markets = ex.load_markets()
        if symbol not in markets and ":" in symbol:
            alt = symbol.split(":", 1)[0]
            if alt in markets:
                resolved_symbol = alt
    except Exception:
        resolved_symbol = symbol

    tf_ms = timeframe_to_ms(timeframe)
    cur = since_ms
    rows: List[List[Any]] = []
    while True:
        chunk = ex.fetch_ohlcv(resolved_symbol, timeframe=timeframe, since=cur, limit=limit)
        if not chunk:
            break
        rows.extend(chunk)
        last_ts = int(chunk[-1][0])
        if end_ms is not None and last_ts >= int(end_ms):
            break
        nxt = last_ts + tf_ms
        if cur is not None and nxt <= cur:
            break
        cur = nxt
        if verbose:
            print(f"[fetch] {exchange_id} {resolved_symbol} {timeframe} last_ts={last_ts} rows={len(rows)}")
        if len(chunk) < max(10, limit // 2):
            if end_ms is None:
                break
    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["ts_ms"] = df["ts_ms"].astype(np.int64)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    if end_ms is not None:
        df = df[df["ts_ms"] <= int(end_ms) + tf_ms]
    return df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"]).reset_index(drop=True)


def get_ohlcv(
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    cache_dir: Path,
    since_ms: Optional[int],
    end_ms: Optional[int],
    no_fetch: bool,
    verbose: bool,
) -> pd.DataFrame:
    _ensure_cache_dir(cache_dir)
    cache_path = cache_dir / f"{exchange_id}_{_sanitize_symbol(symbol)}_{timeframe}.csv"
    cached = load_cached_ohlcv(cache_path)
    if cached is not None and not cached.empty:
        have_min = int(cached["ts_ms"].min())
        have_max = int(cached["ts_ms"].max())
        need_min = since_ms if since_ms is not None else have_min
        need_max = end_ms if end_ms is not None else have_max
        if need_min >= have_min and need_max <= have_max:
            df = cached
            if since_ms is not None:
                df = df[df["ts_ms"] >= int(since_ms)]
            if end_ms is not None:
                df = df[df["ts_ms"] <= int(end_ms)]
            return df.reset_index(drop=True)

    if no_fetch:
        raise RuntimeError(f"Cache miss for {exchange_id} {symbol} {timeframe} and --no-fetch is set: {cache_path}")

    fetched = fetch_ohlcv_ccxt(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        end_ms=end_ms,
        verbose=verbose,
    )
    if cached is None or cached.empty:
        merged = fetched
    else:
        merged = pd.concat([cached, fetched], ignore_index=True)
    save_cached_ohlcv(cache_path, merged)

    df = merged
    if since_ms is not None:
        df = df[df["ts_ms"] >= int(since_ms)]
    if end_ms is not None:
        df = df[df["ts_ms"] <= int(end_ms)]
    return df.reset_index(drop=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def agreement_rows(bands: List[Band]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int, str], Dict[str, Band]] = {}
    for b in bands:
        by_key.setdefault((b.ts_ms, b.timeframe), {})[b.method] = b

    def _pair_metrics(a: Band, b: Band) -> Tuple[float, bool]:
        price = float(a.price)
        diff_pct = abs(float(a.level) - float(b.level)) / (price + 1e-12)
        agree = (float(b.level) >= float(a.band_low) and float(b.level) <= float(a.band_high)) or (
            float(a.level) >= float(b.band_low) and float(a.level) <= float(b.band_high)
        )
        return diff_pct, bool(agree)

    out: List[Dict[str, Any]] = []
    for (ts_ms, tf), m in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(m) < 2:
            continue
        price = float(next(iter(m.values())).price)
        row: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "ts_ms": ts_ms,
            "timeframe": tf,
            "price": price,
            "n_methods": int(len(m)),
        }
        for method, band in m.items():
            row[f"level_{method}"] = float(band.level)
            row[f"band_low_{method}"] = float(band.band_low)
            row[f"band_high_{method}"] = float(band.band_high)

        present = sorted(m.keys())
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a = present[i]
                b = present[j]
                diff_pct, agree = _pair_metrics(m[a], m[b])
                row[f"level_diff_pct_{a}_{b}"] = float(diff_pct)
                row[f"agreement_{a}_{b}"] = bool(agree)
        out.append(row)
    return out


def render_overlay_png(
    *,
    out_png: Path,
    df: pd.DataFrame,
    timeframe: str,
    bands: List[Band],
    window_start_ms: Optional[int],
    window_end_ms: Optional[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    df = df.sort_values("ts_ms").reset_index(drop=True)
    if window_start_ms is not None:
        df = df[df["ts_ms"] >= int(window_start_ms)]
    if window_end_ms is not None:
        df = df[df["ts_ms"] <= int(window_end_ms)]
    if df.empty:
        return

    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    close = df["close"].astype(float).to_numpy()

    bdf = [b for b in bands if b.timeframe == timeframe]
    if window_start_ms is not None:
        bdf = [b for b in bdf if b.ts_ms >= int(window_start_ms)]
    if window_end_ms is not None:
        bdf = [b for b in bdf if b.ts_ms <= int(window_end_ms)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, close, color="black", linewidth=1.2, label="close")

    for method, color in [("smc", "#1f77b4"), ("kmeans", "#ff7f0e"), ("smc_lib", "#2ca02c")]:
        mb = [b for b in bdf if b.method == method]
        if not mb:
            continue
        m_ts = [datetime.fromtimestamp(b.ts_ms / 1000, tz=timezone.utc) for b in mb]
        levels = [b.level for b in mb]
        lows = [b.band_low for b in mb]
        highs = [b.band_high for b in mb]
        ax.plot(m_ts, levels, color=color, linewidth=1.0, label=f"{method} level")
        ax.fill_between(m_ts, lows, highs, color=color, alpha=0.12, step="pre", label=f"{method} band")

    ax.set_title(f"Resistance Bands Overlay ({timeframe})")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_report_md(
    *,
    out_md: Path,
    exchange_id: str,
    symbol: str,
    timeframes: Sequence[str],
    params: Dict[str, Any],
    bands_rows: List[Dict[str, Any]],
    agreement: List[Dict[str, Any]],
    out_png: Optional[Path],
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def _summ(vals: List[float]) -> str:
        if not vals:
            return "—"
        s = sorted(vals)
        med = s[len(s) // 2]
        return f"n={len(vals)} median={med:.5f} min={s[0]:.5f} max={s[-1]:.5f}"

    lines: List[str] = []
    lines.append("# Resistance Band Audit")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Exchange: `{exchange_id}`")
    lines.append(f"- Symbol: `{symbol}`")
    lines.append(f"- Timeframes: `{','.join(timeframes)}`")
    if out_png:
        lines.append(f"- Overlay PNG: `{out_png}`")
    lines.append("")

    lines.append("## Lookahead-Safe Design")
    lines.append("- Pivot highs are *confirmed* only after `pivot_right` bars; at eval index `t`, we only use pivots with `i <= t - pivot_right`.")
    lines.append(
        "- `smc_lib` swing highs are *confirmed* only after `smc_swing_length` bars; at eval index `t`, we only use swings with `i <= t - smc_swing_length`."
    )
    lines.append("- `exclude_last` is a live-safety guard: in `--live` it shifts evaluation back N bars; offline it acts as an extra pivot-age/confirmation delay (scaled per timeframe).")
    lines.append("")

    lines.append("## Parameters")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## Methods")
    lines.append("- `smc`: confirmed pivot-high extraction + 1D clustering of pivot highs (liquidity highs), then nearest band above price.")
    lines.append("- `kmeans`: KMeans clustering on confirmed pivot-high prices (walk-forward), then nearest cluster band above price.")
    lines.append("- `smc_lib`: SmartMoneyConcepts README-style `swing_highs_lows` + `liquidity` approximation (confirmed swing highs clustered by `range_percent`).")
    lines.append("")

    if agreement:
        pair_cols = sorted({c for r in agreement for c in r.keys() if c.startswith("agreement_")})
        lines.append("## Method Agreement (Pairwise)")
        lines.append(f"- Rows: **{len(agreement)}**")
        for col in pair_cols:
            pair = col.replace("agreement_", "")
            agree_rate = sum(1 for r in agreement if r.get(col)) / max(1, len(agreement))
            diff_col = f"level_diff_pct_{pair}"
            diffs = [float(r[diff_col]) for r in agreement if r.get(diff_col) is not None]
            lines.append(f"- `{pair}`: agreement={agree_rate:.1%} | level_diff_pct: {_summ(diffs)}")
        lines.append("")

    lines.append("## Per-Timeframe Summary")
    df = pd.DataFrame(bands_rows) if bands_rows else pd.DataFrame()
    if not df.empty:
        for tf in timeframes:
            lines.append(f"### {tf}")
            tf_all = df[df["timeframe"] == tf]
            denom = max(1, len(tf_all))
            for method in sorted(set(df["method"].tolist())):
                sub = tf_all[tf_all["method"] == method]
                if sub.empty:
                    continue
                dist_vals = [float(x) for x in sub["distance_pct_to_level"].tolist() if x is not None and math.isfinite(float(x))]
                lines.append(f"- `{method}` rows={len(sub)}/{denom} | dist_to_level_pct: {_summ(dist_vals)}")
            lines.append("")

        # Spot-check: 2026-01-20 22:40 (common OB reject window)
        spot_ts = "2026-01-20T22:40:00Z"
        spot = df[df["timestamp"] == spot_ts].copy()
        if not spot.empty:
            cols = ["timeframe", "method", "price", "nearest_res_level", "band_low", "band_high", "distance_pct_to_level"]
            lines.append(f"## Spot-Check ({spot_ts})")
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, r in spot.sort_values(["timeframe", "method"]).iterrows():
                lines.append(
                    "| {timeframe} | {method} | {price:.2f} | {level:.2f} | {lo:.2f} | {hi:.2f} | {dist:.5f} |".format(
                        timeframe=r["timeframe"],
                        method=r["method"],
                        price=float(r["price"]),
                        level=float(r["nearest_res_level"]),
                        lo=float(r["band_low"]),
                        hi=float(r["band_high"]),
                        dist=float(r["distance_pct_to_level"]),
                    )
                )
            lines.append("")

        # Small sample per timeframe (first 5 rows per method)
        lines.append("## Samples")
        for tf in timeframes:
            tf_all = df[df["timeframe"] == tf]
            if tf_all.empty:
                continue
            lines.append(f"### {tf}")
            cols = ["timestamp", "method", "price", "nearest_res_level", "band_low", "band_high"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            sample = tf_all.sort_values(["ts_ms", "method"]).head(10)
            for _, r in sample.iterrows():
                lines.append(
                    "| {ts} | {method} | {price:.2f} | {level:.2f} | {lo:.2f} | {hi:.2f} |".format(
                        ts=r["timestamp"],
                        method=r["method"],
                        price=float(r["price"]),
                        level=float(r["nearest_res_level"]),
                        lo=float(r["band_low"]),
                        hi=float(r["band_high"]),
                    )
                )
            lines.append("")

    lines.append("## Notes")
    lines.append("- KMeans clusters only historical confirmed pivot highs within `sr_lookback_bars` for each eval point (walk-forward).")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect nearest upper resistance bands (SMC pivots + KMeans).")
    parser.add_argument("--exchange", default="bingx", help="ccxt exchange id (default: bingx)")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC/USDT:USDT)")
    parser.add_argument("--timeframes", default="5m,30m", help="Comma-separated timeframes (e.g., 5m,30m,1h)")
    parser.add_argument("--start", help="ISO8601 UTC, e.g. 2026-01-20T21:30:00Z")
    parser.add_argument("--end", help="ISO8601 UTC, e.g. 2026-01-20T23:30:00Z")
    parser.add_argument("--lookback-bars", type=int, help="Fetch last N bars (used by --live or single-shot)")
    parser.add_argument("--method", default="both", choices=["smc", "kmeans", "smc_lib", "both", "all"])
    parser.add_argument("--live", action="store_true", help="Live-sim: compute for last bar and print JSONL to stdout")
    parser.add_argument(
        "--exclude-last",
        type=int,
        default=0,
        help="Live guard: shift eval back N bars (scaled per timeframe); offline: extra pivot-age delay.",
    )
    parser.add_argument("--history-minutes", type=int, default=720, help="History buffer before start (default: 12h)")
    parser.add_argument("--sr-lookback-bars", type=int, default=300, help="Pivot lookback bars for SR (default: 300)")
    parser.add_argument("--pivot-left", type=int, default=3)
    parser.add_argument("--pivot-right", type=int, default=3)
    parser.add_argument("--smc-swing-length", type=int, default=50, help="SmartMoneyConcepts swing_length (default: 50)")
    parser.add_argument(
        "--smc-liquidity-range-pct",
        type=float,
        default=0.01,
        help="SmartMoneyConcepts liquidity range_percent (default: 0.01 = 1%%)",
    )
    parser.add_argument("--band-mode", default="pct", choices=["pct", "atr"], help="SMC band sizing (pct or ATR)")
    parser.add_argument("--band-pct", type=float, default=0.003, help="Band half-width as pct (default: 0.3%%)")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--band-atr-mult", type=float, default=0.2, help="Band half-width as ATR*mult (default: 0.2)")
    parser.add_argument("--smc-cluster-pct", type=float, default=0.0015, help="SMC cluster proximity pct (default: 0.15%%)")
    parser.add_argument("--kmin", type=int, default=3)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--eval-horizon-bars", type=int, default=0, help="Offline validation: look forward H bars for touch metrics")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache") / "ohlcv")
    parser.add_argument("--no-fetch", action="store_true", help="Do not call ccxt; require cache to exist")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-csv", type=Path, help="Write bands CSV")
    parser.add_argument("--out-jsonl", type=Path, help="Write bands JSONL")
    parser.add_argument("--out-agreement-csv", type=Path, help="Write pairwise agreement CSV (requires >=2 methods)")
    parser.add_argument("--out-md", type=Path, help="Write markdown report (offline only)")
    parser.add_argument("--out-png", type=Path, help="Write overlay PNG (plots first timeframe only; offline only)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    timeframes = [t.strip() for t in str(args.timeframes).split(",") if t.strip()]
    if not timeframes:
        raise SystemExit("No timeframes provided")

    base_tf_ms = min(timeframe_to_ms(tf) for tf in timeframes)
    if args.method == "both":
        methods = ["smc", "kmeans"]
    elif args.method == "all":
        methods = ["smc", "kmeans", "smc_lib"]
    else:
        methods = [args.method]
    methods = list(dict.fromkeys(methods))
    start_ms = _ms(_dt_utc_from_iso(args.start)) if args.start else None
    end_ms = _ms(_dt_utc_from_iso(args.end)) if args.end else None

    if args.live and not args.lookback_bars:
        raise SystemExit("--live requires --lookback-bars")
    if not args.live and start_ms is None and end_ms is None and not args.lookback_bars:
        raise SystemExit("Provide --start/--end or --lookback-bars")

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    bands: List[Band] = []
    bands_rows: List[Dict[str, Any]] = []
    exclude_effective: Dict[str, int] = {}

    for tf in timeframes:
        tf_ms = timeframe_to_ms(tf)
        exclude_last_tf = 0
        if args.exclude_last and int(args.exclude_last) > 0:
            exclude_last_tf = int(math.ceil(int(args.exclude_last) * (base_tf_ms / tf_ms)))
        exclude_effective[tf] = int(exclude_last_tf)

        if args.live:
            since_ms = now_ms - int(args.lookback_bars) * tf_ms
            end_fetch = now_ms
        else:
            if start_ms is not None:
                since_ms = max(0, int(start_ms) - int(args.history_minutes) * 60_000)
                end_fetch = int(end_ms) if end_ms is not None else None
            else:
                since_ms = now_ms - int(args.lookback_bars or 500) * tf_ms
                end_fetch = now_ms

            if end_fetch is not None and args.eval_horizon_bars and args.eval_horizon_bars > 0:
                end_fetch = end_fetch + int(args.eval_horizon_bars) * tf_ms

        df = get_ohlcv(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=tf,
            cache_dir=args.cache_dir,
            since_ms=since_ms,
            end_ms=end_fetch,
            no_fetch=bool(args.no_fetch),
            verbose=bool(args.verbose),
        )
        if df.empty:
            continue

        single_shot = (start_ms is None and end_ms is None and bool(args.lookback_bars))
        if args.live or single_shot:
            df_eval = df.iloc[: -exclude_last_tf] if exclude_last_tf and len(df) > exclude_last_tf else df
            if df_eval.empty:
                continue
            eval_ts_ms = int(df_eval["ts_ms"].iloc[-1])
            eval_start = eval_ts_ms
            eval_end = eval_ts_ms
            confirm_delay_bars = 0
        else:
            eval_start = start_ms
            eval_end = end_ms
            confirm_delay_bars = int(exclude_last_tf)

        tf_bands = compute_walk_forward_bands(
            df=df,
            timeframe=tf,
            methods=methods,
            pivot_left=int(args.pivot_left),
            pivot_right=int(args.pivot_right),
            smc_swing_length=int(args.smc_swing_length),
            smc_liquidity_range_pct=float(args.smc_liquidity_range_pct),
            sr_lookback_bars=int(args.sr_lookback_bars),
            band_mode=str(args.band_mode),
            band_pct=float(args.band_pct),
            atr_period=int(args.atr_period),
            band_atr_mult=float(args.band_atr_mult),
            smc_cluster_pct=float(args.smc_cluster_pct),
            kmin=int(args.kmin),
            kmax=int(args.kmax),
            random_state=int(args.random_state),
            exclude_last=confirm_delay_bars,
            eval_start_ms=eval_start,
            eval_end_ms=eval_end,
            eval_horizon_bars=int(args.eval_horizon_bars),
        )
        bands.extend(tf_bands)

        if args.out_png and not args.live and tf == timeframes[0]:
            render_overlay_png(
                out_png=args.out_png,
                df=df,
                timeframe=tf,
                bands=tf_bands,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
            )

    bands_rows = [b.to_row() for b in bands]
    agree = agreement_rows(bands) if len(methods) >= 2 else []

    if args.out_csv:
        write_csv(args.out_csv, bands_rows)
    if args.out_jsonl:
        write_jsonl(args.out_jsonl, bands_rows)
    if args.out_agreement_csv and agree:
        write_csv(args.out_agreement_csv, agree)
    if args.out_md and not args.live:
        params = {
            "pivot_left": args.pivot_left,
            "pivot_right": args.pivot_right,
            "smc_swing_length": args.smc_swing_length,
            "smc_liquidity_range_pct": args.smc_liquidity_range_pct,
            "sr_lookback_bars": args.sr_lookback_bars,
            "band_mode": args.band_mode,
            "band_pct": args.band_pct,
            "atr_period": args.atr_period,
            "band_atr_mult": args.band_atr_mult,
            "smc_cluster_pct": args.smc_cluster_pct,
            "kmin": args.kmin,
            "kmax": args.kmax,
            "exclude_last_input": args.exclude_last,
            "exclude_last_effective_bars": exclude_effective,
            "eval_horizon_bars": args.eval_horizon_bars,
        }
        write_report_md(
            out_md=args.out_md,
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframes=timeframes,
            params=params,
            bands_rows=bands_rows,
            agreement=agree,
            out_png=args.out_png,
        )

    if args.live:
        for row in sorted(bands_rows, key=lambda r: (r.get("ts_ms", 0), r.get("timeframe", ""), r.get("method", ""))):
            print(json.dumps(row, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
