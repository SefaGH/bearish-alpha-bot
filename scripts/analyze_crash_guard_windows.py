#!/usr/bin/env python3
"""
False-positive oriented window analysis for Crash Guard (adaptive_ob).

This script picks (or accepts) analysis windows and reports, per window:
  - blocked_rate / accepted_rate (within eligible volume buckets)
  - strict_extreme_share (share of panic samples treated as strict_extreme)
  - missing_reclaim count (blocked strict_extreme cases missing reclaim)
  - volume bucket distribution (real VolumeAnalyzer bucket, not assumed)
  - MFE/MAE proxy for accepted decisions over 5m and 10m horizons (1m OHLCV highs/lows)

Eligibility:
  The crash guard only applies when `volume_bucket` is in `panic_volume_buckets`
  (default ["HIGH","EXTREME"]). Rates are computed over eligible samples.

Window selection (if --windows not provided):
  Uses 1h OHLCV over a search range to pick three 2h windows:
    1) calm/range: smallest range+abs(return)
    2) trend-up high-vol: large positive return with high volume
    3) spike/pump: very high volume with positive return
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml  # repo-local minimal yaml shim

from src.core.ccxt_client import CcxtClient
from src.core.indicators import add_indicators
from src.core.strategy_coordinator import StrategyCoordinator
from src.core.volume_analyzer import VolumeAnalyzer


def _parse_iso_utc(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_pandas_freq(tf: str) -> str:
    s = str(tf or "").strip().lower()
    if s.endswith("m"):
        return f"{int(s[:-1])}min"
    if s.endswith("h"):
        return f"{int(s[:-1])}h"
    if s.endswith("d"):
        return f"{int(s[:-1])}d"
    # Fallback: assume minutes
    return "1min"


def _floor(dt: datetime, freq: str) -> datetime:
    return pd.Timestamp(dt).floor(_to_pandas_freq(freq)).to_pydatetime().replace(tzinfo=timezone.utc)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        if getattr(out.index, "tz", None) is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    except Exception:
        out.index = pd.to_datetime(out.index, utc=True)
    return out


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh.read()) or {}


def _normalize_inline_list(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return value
    try:
        return json.loads(s)
    except Exception:
        return value


class ReplayMarketDataPipeline:
    """
    Provides time-sliced OHLCV for VolumeAnalyzer + Crash Guard.

    - VolumeAnalyzer requests trade_tf (default 5m) + baselines (default 1h, 4h)
      via get_latest_ohlcv(symbol, tf, limit=...)
    - Crash Guard requests panic_tf (default 5m) via include_forming=True
    """

    def __init__(self, *, dfs_by_tf: Dict[str, pd.DataFrame], df_5m_ind: pd.DataFrame):
        self._dfs_by_tf = {str(k).lower(): _ensure_utc_index(v) for k, v in (dfs_by_tf or {}).items()}
        self._df_5m_ind = _ensure_utc_index(df_5m_ind)
        self._now: datetime = datetime.now(timezone.utc)

    def set_now(self, now: datetime) -> None:
        self._now = now.astimezone(timezone.utc)

    async def get_latest_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = None,
        limit: int = None,
        include_forming: bool = False,
        hybrid_policy: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        tf = str(timeframe or "").strip().lower()
        if tf == "5m":
            # Crash Guard needs indicators for panic_tf; VolumeAnalyzer can live with them too.
            df = self._df_5m_ind
        else:
            df = self._dfs_by_tf.get(tf)
            if df is None:
                return None

        cutoff = pd.Timestamp(_floor(self._now, tf)).tz_convert("UTC")
        out = df.loc[df.index < cutoff].copy()

        if out.empty:
            return out
        if limit is not None:
            out = out.tail(int(limit))
        return out


def _make_coordinator(*, market_data_pipeline: Any, cfg: Dict[str, Any]) -> StrategyCoordinator:
    pm = type("PM", (), {})()
    pm.cfg = cfg or {}
    rm = type("RM", (), {})()
    return StrategyCoordinator(pm, rm, market_data_pipeline=market_data_pipeline, config=cfg)


def _compute_reversal_meta(df_1m_ind: pd.DataFrame, *, now: datetime, hook_delta: float) -> Dict[str, bool]:
    cutoff = pd.Timestamp(_floor(now, "1min")).tz_convert("UTC")
    df = _ensure_utc_index(df_1m_ind).loc[_ensure_utc_index(df_1m_ind).index < cutoff]
    if df is None or df.empty or len(df) < 2:
        return {"rsi_hook": False, "bull_candle": False, "reclaim": False}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_hook = False
    try:
        rsi_hook = (float(last.get("rsi")) - float(prev.get("rsi"))) >= float(hook_delta)
    except Exception:
        rsi_hook = False

    bull_candle = False
    try:
        bull_candle = bool(float(last.get("close")) > float(last.get("open")) and float(last.get("close")) > float(prev.get("close")))
    except Exception:
        bull_candle = False

    reclaim = False
    try:
        reclaim = bool(float(last.get("close")) > float(last.get("ema_fast")))
    except Exception:
        reclaim = False

    return {"rsi_hook": bool(rsi_hook), "bull_candle": bool(bull_candle), "reclaim": bool(reclaim)}


def _mfe_mae_proxy(
    df_1m_raw: pd.DataFrame,
    *,
    now: datetime,
    horizon_min: int,
) -> Optional[Tuple[float, float]]:
    df = _ensure_utc_index(df_1m_raw)
    t0 = pd.Timestamp(_floor(now, "1min")).tz_convert("UTC")
    past = df.loc[df.index < t0]
    if past.empty:
        return None
    entry = float(past.iloc[-1]["close"])
    if entry <= 0:
        return None
    t1 = t0 + pd.Timedelta(minutes=int(horizon_min))
    fut = df.loc[(df.index >= t0) & (df.index < t1)]
    if fut.empty:
        return None
    try:
        mfe = (float(fut["high"].max()) - entry) / entry
        mae = (float(fut["low"].min()) - entry) / entry
        return float(mfe), float(mae)
    except Exception:
        return None


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = pd.Series(values)
    try:
        return float(s.quantile(q))
    except Exception:
        return None


@dataclass(frozen=True)
class Window:
    name: str
    start: datetime
    end: datetime


def _pick_windows_from_1h(df_1h: pd.DataFrame, *, duration_h: int = 2) -> List[Window]:
    df = _ensure_utc_index(df_1h)
    if df is None or df.empty:
        return []

    # Basic per-hour descriptors.
    tmp = df.copy()
    tmp["ret"] = (tmp["close"] - tmp["open"]) / tmp["open"].replace(0, pd.NA)
    tmp["range"] = (tmp["high"] - tmp["low"]) / tmp["close"].replace(0, pd.NA)
    tmp["vol"] = tmp.get("volume", pd.Series(index=tmp.index, data=0.0))
    tmp = tmp.dropna(subset=["ret", "range"])
    if tmp.empty:
        return []

    # Calm: smallest range + abs(ret).
    tmp["calm_score"] = tmp["range"].abs() + (tmp["ret"].abs() * 0.5)
    calm_idx = tmp["calm_score"].idxmin()

    # Trend-up high vol: large positive return among top-quantile volume.
    vol_q = float(tmp["vol"].quantile(0.75))
    cand_up = tmp.loc[(tmp["vol"] >= vol_q) & (tmp["ret"] > 0)].copy()
    if cand_up.empty:
        cand_up = tmp.loc[tmp["ret"] > 0].copy()
    cand_up["up_score"] = cand_up["ret"] * (1.0 + (cand_up["vol"] / float(tmp["vol"].median() or 1.0)).clip(0, 10))
    up_idx = cand_up["up_score"].idxmax()

    # Spike/pump: very high volume with positive return (volume * positive body).
    cand_pump = tmp.loc[tmp["ret"] > 0].copy()
    cand_pump["pump_score"] = cand_pump["vol"] * cand_pump["ret"].clip(lower=0)
    pump_idx = cand_pump["pump_score"].idxmax() if not cand_pump.empty else tmp["vol"].idxmax()

    def _mk(name: str, idx: pd.Timestamp) -> Window:
        start = idx.to_pydatetime().replace(tzinfo=timezone.utc)
        end = (idx + pd.Timedelta(hours=int(duration_h))).to_pydatetime().replace(tzinfo=timezone.utc)
        return Window(name=name, start=start, end=end)

    # De-overlap by shifting later windows if necessary.
    windows = [_mk("calm_range", calm_idx), _mk("trend_up_high_vol", up_idx), _mk("news_spike_pump", pump_idx)]
    windows_sorted = sorted(windows, key=lambda w: w.start)
    out: List[Window] = []
    for w in windows_sorted:
        if not out:
            out.append(w)
            continue
        last = out[-1]
        if w.start < last.end:
            # shift forward to after last.end (keep duration)
            shift = last.end
            out.append(Window(name=w.name, start=shift, end=shift + (w.end - w.start)))
        else:
            out.append(w)
    return out


async def _screen_window_eligible_share(
    *,
    client: CcxtClient,
    symbol: str,
    cfg: Dict[str, Any],
    window: Window,
    baseline_short: pd.DataFrame,
    baseline_medium: pd.DataFrame,
    step_min: int = 5,
) -> Tuple[float, Dict[str, int]]:
    vol_cfg = cfg.get("volume_analyzer", {}) or {}
    strat_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {})
    crash_cfg = (strat_cfg.get("crash_guard") or {}) if isinstance(strat_cfg, dict) else {}
    if isinstance(crash_cfg, dict) and "panic_volume_buckets" in crash_cfg:
        crash_cfg["panic_volume_buckets"] = _normalize_inline_list(crash_cfg.get("panic_volume_buckets"))
    allowed_raw = (crash_cfg.get("panic_volume_buckets") or ["HIGH", "EXTREME"]) if isinstance(crash_cfg, dict) else ["HIGH", "EXTREME"]
    try:
        allowed = {str(b).upper().strip() for b in allowed_raw if b}
    except Exception:
        allowed = {"HIGH", "EXTREME"}

    # Fetch just 5m for the window (plus a small lookback) for volume context.
    since_fast = window.start - timedelta(hours=6)
    df_5m = await client.ohlcv(
        symbol,
        "5m",
        limit=1440,
        add_indicators=False,
        since=int(since_fast.timestamp() * 1000),
    )
    df_5m = _ensure_utc_index(df_5m) if df_5m is not None else pd.DataFrame()
    df_5m = df_5m.loc[(df_5m.index >= pd.Timestamp(since_fast)) & (df_5m.index <= pd.Timestamp(window.end))]

    mdp = ReplayMarketDataPipeline(
        dfs_by_tf={
            str(vol_cfg.get("baseline_short_tf", "1h") or "1h"): baseline_short,
            str(vol_cfg.get("baseline_medium_tf", "4h") or "4h"): baseline_medium,
        },
        df_5m_ind=df_5m,
    )
    volume_analyzer = VolumeAnalyzer(mdp, config=vol_cfg)

    total = 0
    eligible = 0
    buckets: Dict[str, int] = {}

    t = window.start.replace(second=0, microsecond=0)
    step = timedelta(minutes=max(1, int(step_min)))
    while t <= window.end:
        total += 1
        mdp.set_now(t)
        ctx = await volume_analyzer.compute_context(symbol, trade_timeframe="5m")
        bucket = str(getattr(ctx, "bucket", "NORMAL") or "NORMAL").upper().strip()
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if bucket in allowed:
            eligible += 1
        t = (t + step).replace(tzinfo=timezone.utc)

    share = (eligible / total) if total else 0.0
    return float(share), buckets


async def _fetch_tf(
    client: CcxtClient,
    *,
    symbol: str,
    tf: str,
    since: datetime,
    limit: int,
) -> pd.DataFrame:
    df = await client.ohlcv(symbol, tf, limit=limit, add_indicators=False, since=int(since.timestamp() * 1000))
    if df is None or df.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(df)
    return df


async def analyze_window(
    *,
    exchange: str,
    symbol: str,
    cfg: Dict[str, Any],
    window: Window,
    sample_step_min: int,
    horizons_min: List[int],
    lookback_days_volume: int,
    lookback_hours_fast: int,
) -> Dict[str, Any]:
    ind_cfg = cfg.get("indicators", {}) or {}
    vol_cfg = cfg.get("volume_analyzer", {}) or {}
    strat_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {})
    crash_cfg = (strat_cfg.get("crash_guard") or {}) if isinstance(strat_cfg, dict) else {}
    if isinstance(crash_cfg, dict):
        if "panic_volume_buckets" in crash_cfg:
            crash_cfg["panic_volume_buckets"] = _normalize_inline_list(crash_cfg.get("panic_volume_buckets"))
    hook_delta = float(strat_cfg.get("hook_delta", 1.0) or 1.0)

    # VolumeAnalyzer config needs enough baseline history.
    short_tf = str(vol_cfg.get("baseline_short_tf", "1h") or "1h")
    med_tf = str(vol_cfg.get("baseline_medium_tf", "4h") or "4h")

    client = CcxtClient(str(exchange).lower())

    # Fetch baselines (1h/4h) for volume buckets.
    since_vol = window.end - timedelta(days=int(lookback_days_volume))
    df_1h = await _fetch_tf(client, symbol=symbol, tf=short_tf, since=since_vol, limit=1440)
    df_4h = await _fetch_tf(client, symbol=symbol, tf=med_tf, since=since_vol, limit=1440)

    # Fetch fast (1m/5m) for this window + indicator warmup + forward horizon.
    since_fast = window.start - timedelta(hours=int(lookback_hours_fast))
    max_h = max([int(x) for x in horizons_min] or [10])
    until_fast = window.end + timedelta(minutes=max_h)

    df_1m_raw = await _fetch_tf(client, symbol=symbol, tf="1m", since=since_fast, limit=1440)
    df_5m_raw = await _fetch_tf(client, symbol=symbol, tf="5m", since=since_fast, limit=1440)

    df_1m_raw = df_1m_raw.loc[(df_1m_raw.index >= pd.Timestamp(since_fast)) & (df_1m_raw.index <= pd.Timestamp(until_fast))]
    df_5m_raw = df_5m_raw.loc[(df_5m_raw.index >= pd.Timestamp(since_fast)) & (df_5m_raw.index <= pd.Timestamp(until_fast))]

    df_1m_ind = add_indicators(df_1m_raw, ind_cfg) if not df_1m_raw.empty else df_1m_raw
    df_5m_ind = add_indicators(df_5m_raw, ind_cfg) if not df_5m_raw.empty else df_5m_raw

    mdp = ReplayMarketDataPipeline(
        dfs_by_tf={short_tf: df_1h, med_tf: df_4h},
        df_5m_ind=df_5m_ind,
    )
    coordinator = _make_coordinator(market_data_pipeline=mdp, cfg=cfg)
    volume_analyzer = VolumeAnalyzer(mdp, config=vol_cfg)

    allowed_buckets_raw = crash_cfg.get("panic_volume_buckets") or ["HIGH", "EXTREME"]
    try:
        allowed_buckets = {str(b).upper().strip() for b in allowed_buckets_raw if b}
    except Exception:
        allowed_buckets = {"HIGH", "EXTREME"}

    total = 0
    eligible = 0
    accepted = 0
    blocked = 0
    accepted_overall = 0
    blocked_overall = 0

    panic_true = 0
    strict_extreme = 0
    missing_reclaim = 0

    bucket_counts: Dict[str, int] = {}

    mfe_mae: Dict[int, Dict[str, List[float]]] = {h: {"mfe": [], "mae": []} for h in horizons_min}

    t = window.start.replace(second=0, microsecond=0)
    step = timedelta(minutes=max(1, int(sample_step_min)))
    while t <= window.end:
        total += 1
        mdp.set_now(t)

        ctx = await volume_analyzer.compute_context(symbol, trade_timeframe="5m")
        bucket = str(getattr(ctx, "bucket", "NORMAL") or "NORMAL").upper().strip()
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        blocked_this = False

        if bucket in allowed_buckets:
            eligible += 1
            is_panic, panic_meta = await coordinator._compute_panic_state(symbol=symbol, volume_bucket=bucket, crash_cfg=crash_cfg)
            if is_panic:
                panic_true += 1

                meta = _compute_reversal_meta(df_1m_ind, now=t, hook_delta=hook_delta)
                rsi_hook = bool(meta.get("rsi_hook"))
                bull_candle = bool(meta.get("bull_candle"))
                reclaim = bool(meta.get("reclaim"))

                ema_gap_atr = None
                try:
                    ema_gap_atr = float(panic_meta.get("ema_fast_gap_atr")) if panic_meta.get("ema_fast_gap_atr") is not None else None
                except Exception:
                    ema_gap_atr = None
                try:
                    extreme_gap_th = float(crash_cfg.get("extreme_gap_atr_threshold", 0.0) or 0.0)
                except Exception:
                    extreme_gap_th = 0.0

                strict = bool(bucket == "EXTREME")
                if not strict and extreme_gap_th > 0 and ema_gap_atr is not None and ema_gap_atr >= extreme_gap_th:
                    strict = True
                if strict:
                    strict_extreme += 1
                    reversal_ok = bool(rsi_hook and reclaim)
                else:
                    reversal_ok = bool(rsi_hook and (bull_candle or reclaim))

                if not reversal_ok:
                    blocked += 1
                    blocked_this = True
                    if strict and not reclaim:
                        missing_reclaim += 1
                else:
                    accepted += 1
            else:
                accepted += 1
        else:
            # Not eligible: crash guard does not apply => treat as accepted overall.
            pass

        if blocked_this:
            blocked_overall += 1
        else:
            accepted_overall += 1
            for h in horizons_min:
                mm = _mfe_mae_proxy(df_1m_raw, now=t, horizon_min=int(h))
                if mm is not None:
                    mfe_mae[int(h)]["mfe"].append(mm[0])
                    mfe_mae[int(h)]["mae"].append(mm[1])

        t = (t + step).replace(tzinfo=timezone.utc)

    eligible_share = (eligible / total) if total else 0.0
    blocked_rate = (blocked / eligible) if eligible else 0.0
    accepted_rate = (accepted / eligible) if eligible else 0.0
    strict_share = (strict_extreme / panic_true) if panic_true else 0.0
    accepted_rate_overall = (accepted_overall / total) if total else 0.0
    blocked_rate_overall = (blocked_overall / total) if total else 0.0

    mfe_mae_stats: Dict[str, Any] = {}
    for h in horizons_min:
        mfe = mfe_mae[int(h)]["mfe"]
        mae = mfe_mae[int(h)]["mae"]
        mfe_mae_stats[str(h)] = {
            "n": len(mfe),
            "mfe_mean": float(pd.Series(mfe).mean()) if mfe else None,
            "mfe_median": float(pd.Series(mfe).median()) if mfe else None,
            "mfe_p10": _pct(mfe, 0.10),
            "mfe_p90": _pct(mfe, 0.90),
            "mae_mean": float(pd.Series(mae).mean()) if mae else None,
            "mae_median": float(pd.Series(mae).median()) if mae else None,
            "mae_p10": _pct(mae, 0.10),
            "mae_p90": _pct(mae, 0.90),
        }

    return {
        "window": {"name": window.name, "start_utc": window.start.isoformat(), "end_utc": window.end.isoformat()},
        "counts": {
            "total_samples": total,
            "eligible_samples": eligible,
            "accepted": accepted,
            "blocked": blocked,
            "accepted_overall": accepted_overall,
            "blocked_overall": blocked_overall,
            "panic_true": panic_true,
            "strict_extreme": strict_extreme,
            "missing_reclaim": missing_reclaim,
        },
        "rates": {
            "eligible_share": eligible_share,
            "accepted_rate": accepted_rate,
            "blocked_rate": blocked_rate,
            "accepted_rate_overall": accepted_rate_overall,
            "blocked_rate_overall": blocked_rate_overall,
            "strict_extreme_share": strict_share,
        },
        "volume_bucket_counts": dict(sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "mfe_mae": mfe_mae_stats,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze crash-guard false-positive risk across multiple windows.")
    ap.add_argument("--exchange", default="bingx")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--config", default="config/config.example.yaml")
    ap.add_argument("--sample-step-min", type=int, default=1)
    ap.add_argument("--horizons-min", default="5,10")
    ap.add_argument("--lookback-days-volume", type=int, default=30)
    ap.add_argument("--lookback-hours-fast", type=int, default=6)
    ap.add_argument(
        "--windows",
        nargs="*",
        help="Optional explicit windows: name,start_utc,end_utc (repeatable)",
    )
    ap.add_argument("--auto-search-start-utc", default="2026-01-19T00:00:00Z")
    ap.add_argument("--auto-search-end-utc", default="2026-01-29T23:59:00Z")
    ap.add_argument("--auto-duration-hours", type=int, default=2)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))

    horizons = [int(x.strip()) for x in str(args.horizons_min).split(",") if x.strip()]
    if not horizons:
        horizons = [5, 10]

    windows: List[Window] = []
    if args.windows:
        for raw in args.windows:
            parts = str(raw).split(",", 2)
            if len(parts) != 3:
                raise SystemExit(f"Invalid --windows entry: {raw!r} (expected name,start,end)")
            name, s, e = parts
            windows.append(Window(name=str(name), start=_parse_iso_utc(s), end=_parse_iso_utc(e)))
    else:
        # Auto-pick with eligibility screening.
        search_start = _parse_iso_utc(args.auto_search_start_utc)
        search_end = _parse_iso_utc(args.auto_search_end_utc)
        client = CcxtClient(str(args.exchange).lower())
        df_1h = await client.ohlcv(
            args.symbol,
            "1h",
            limit=1440,
            add_indicators=False,
            since=int(search_start.timestamp() * 1000),
        )
        df_1h = _ensure_utc_index(df_1h) if df_1h is not None else pd.DataFrame()
        df_1h = df_1h.loc[(df_1h.index >= pd.Timestamp(search_start)) & (df_1h.index <= pd.Timestamp(search_end))]

        # Baselines for screening (reuse across candidates).
        since_vol = search_end - timedelta(days=int(args.lookback_days_volume))
        vol_cfg = cfg.get("volume_analyzer", {}) or {}
        short_tf = str(vol_cfg.get("baseline_short_tf", "1h") or "1h")
        med_tf = str(vol_cfg.get("baseline_medium_tf", "4h") or "4h")
        baseline_short = await _fetch_tf(client, symbol=args.symbol, tf=short_tf, since=since_vol, limit=1440)
        baseline_medium = await _fetch_tf(client, symbol=args.symbol, tf=med_tf, since=since_vol, limit=1440)

        # Build candidate anchors from 1h descriptors.
        tmp = _ensure_utc_index(df_1h).copy()
        tmp["ret"] = (tmp["close"] - tmp["open"]) / tmp["open"].replace(0, pd.NA)
        tmp["range"] = (tmp["high"] - tmp["low"]) / tmp["close"].replace(0, pd.NA)
        tmp["vol"] = tmp.get("volume", pd.Series(index=tmp.index, data=0.0))
        tmp = tmp.dropna(subset=["ret", "range"])
        if tmp.empty:
            raise SystemExit("Auto-pick failed: empty 1h descriptors (no data).")
        tmp["calm_score"] = tmp["range"].abs() + (tmp["ret"].abs() * 0.5)
        vol_q = float(tmp["vol"].quantile(0.75))
        cand_up = tmp.loc[(tmp["vol"] >= vol_q) & (tmp["ret"] > 0)].copy()
        if cand_up.empty:
            cand_up = tmp.loc[tmp["ret"] > 0].copy()
        cand_up["up_score"] = cand_up["ret"] * (1.0 + (cand_up["vol"] / float(tmp["vol"].median() or 1.0)).clip(0, 10))
        cand_pump = tmp.loc[tmp["ret"] > 0].copy()
        cand_pump["pump_score"] = cand_pump["vol"] * cand_pump["ret"].clip(lower=0)

        calm_candidates = list(tmp.nsmallest(40, "calm_score").index)
        up_candidates = list(cand_up.nlargest(40, "up_score").index)
        pump_candidates = list(cand_pump.nlargest(40, "pump_score").index) if not cand_pump.empty else list(tmp.nlargest(40, "vol").index)

        def _overlaps(a: Window, b: Window) -> bool:
            return not (a.end <= b.start or a.start >= b.end)

        async def _pick_best(
            name: str,
            candidates: List[pd.Timestamp],
            *,
            mode: str,
            avoid: Optional[List[Window]] = None,
        ) -> Window:
            """
            mode:
              - "max_eligible": pick candidate with maximum eligible_share (prefers HIGH/EXTREME windows)
              - "min_eligible": pick candidate with minimum eligible_share (prefers NORMAL/LOW windows)
            """
            avoid = avoid or []
            best: Optional[Tuple[Tuple[float, float], Window]] = None
            for idx in candidates:
                start = idx.to_pydatetime().replace(tzinfo=timezone.utc)
                w = Window(name=name, start=start, end=start + timedelta(hours=int(args.auto_duration_hours)))
                if any(_overlaps(w, other) for other in avoid):
                    continue
                share, _ = await _screen_window_eligible_share(
                    client=client,
                    symbol=args.symbol,
                    cfg=cfg,
                    window=w,
                    baseline_short=baseline_short,
                    baseline_medium=baseline_medium,
                    step_min=5,
                )
                if mode == "max_eligible":
                    # Require some meaningful HIGH/EXTREME presence.
                    if share < 0.20:
                        continue
                    key = (share, 0.0)
                else:  # min_eligible
                    key = (-share, 0.0)
                if best is None or key > best[0]:
                    best = (key, w)
            if best is None:
                # Fall back to the first non-overlapping candidate.
                for idx in candidates:
                    start = idx.to_pydatetime().replace(tzinfo=timezone.utc)
                    w = Window(name=name, start=start, end=start + timedelta(hours=int(args.auto_duration_hours)))
                    if any(_overlaps(w, other) for other in avoid):
                        continue
                    return w
                # As a last resort, return the first.
                idx0 = candidates[0]
                start0 = idx0.to_pydatetime().replace(tzinfo=timezone.utc)
                return Window(name=name, start=start0, end=start0 + timedelta(hours=int(args.auto_duration_hours)))
            return best[1]

        w_up = await _pick_best("trend_up_high_vol", up_candidates, mode="max_eligible")
        w_pump = await _pick_best("news_spike_pump", pump_candidates, mode="max_eligible", avoid=[w_up])
        w_calm = await _pick_best("calm_range", calm_candidates, mode="min_eligible", avoid=[w_up, w_pump])

        windows = sorted([w_up, w_pump, w_calm], key=lambda w: w.start)

    if not windows:
        raise SystemExit("No windows available (provide --windows or widen auto-search range).")

    print("")
    print("Crash Guard Window Analysis (real volume_bucket via VolumeAnalyzer)")
    print(f"symbol={args.symbol} | sample_step_min={int(args.sample_step_min)} | horizons_min={horizons}")
    print("")
    print("Selected windows (UTC):")
    for w in windows:
        print(f"- {w.name}: {w.start.isoformat()} -> {w.end.isoformat()}")

    reports: List[Dict[str, Any]] = []
    for w in windows:
        rep = await analyze_window(
            exchange=args.exchange,
            symbol=args.symbol,
            cfg=cfg,
            window=w,
            sample_step_min=int(args.sample_step_min),
            horizons_min=horizons,
            lookback_days_volume=int(args.lookback_days_volume),
            lookback_hours_fast=int(args.lookback_hours_fast),
        )
        reports.append(rep)

    print("")
    for rep in reports:
        w = rep["window"]
        counts = rep["counts"]
        rates = rep["rates"]
        print(f"== {w['name']} | {w['start_utc']} -> {w['end_utc']} ==")
        print(
            "eligible_share={:.3f} accepted_rate={:.3f} blocked_rate={:.3f} strict_extreme_share={:.3f} missing_reclaim={}".format(
                rates["eligible_share"],
                rates["accepted_rate"],
                rates["blocked_rate"],
                rates["strict_extreme_share"],
                counts["missing_reclaim"],
            )
        )
        print(f"volume_buckets={rep['volume_bucket_counts']}")
        print(f"mfe_mae={rep['mfe_mae']}")
        print("")

    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
