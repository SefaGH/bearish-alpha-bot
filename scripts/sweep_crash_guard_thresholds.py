#!/usr/bin/env python3
"""
Sensitivity sweep for Crash Guard thresholds (adaptive_ob).

Grid:
  panic_ema_gap_atr_threshold in {2.5, 3.0, 3.5}
  extreme_gap_atr_threshold   in {4.5, 5.0, 5.5}

Metrics per combo:
  - blocked_rate: fraction of sampled timestamps that would be blocked under panic guard
  - strict_extreme_share: share of panic_state samples that are treated as strict_extreme
  - missing_reclaim_count: number of blocks where reclaim is missing under strict_extreme rule
  - pump_pass: HIGH/EXTREME + pump/up move does NOT enter panic_state

Note: For the time-window sampling metric we assume `volume_bucket="HIGH"` so the only way to
enter strict_extreme is via EMA-gap crossing `extreme_gap_atr_threshold`. This approximates
how aggressive the guard is during high-volume conditions without requiring coordinator bucket SSOT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml  # repo-local minimal yaml shim

from src.core.ccxt_client import CcxtClient
from src.core.indicators import add_indicators
from src.core.strategy_coordinator import StrategyCoordinator


def _parse_iso_utc(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _floor(dt: datetime, freq: str) -> datetime:
    return pd.Timestamp(dt).floor(freq).to_pydatetime().replace(tzinfo=timezone.utc)


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


def _build_5m_from_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc_index(df1m)
    if df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample("5min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


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


class ReplayMarketDataPipeline:
    def __init__(self, *, df_1m_raw: pd.DataFrame, df_5m_ind: pd.DataFrame):
        self._df_1m_raw = _ensure_utc_index(df_1m_raw)
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
        if tf != "5m":
            return None

        bucket_start = pd.Timestamp(_floor(self._now, "5min")).tz_convert("UTC")
        closed = self._df_5m_ind.loc[self._df_5m_ind.index < bucket_start].copy()
        if closed.empty:
            return closed

        if include_forming:
            minute_cutoff = pd.Timestamp(_floor(self._now, "1min")).tz_convert("UTC")
            part = self._df_1m_raw.loc[(self._df_1m_raw.index >= bucket_start) & (self._df_1m_raw.index < minute_cutoff)]
            if not part.empty:
                forming_row = {
                    "open": float(part["open"].iloc[0]),
                    "high": float(part["high"].max()),
                    "low": float(part["low"].min()),
                    "close": float(part["close"].iloc[-1]),
                    "volume": float(part["volume"].sum()),
                }
                forming_df = pd.DataFrame([forming_row], index=[bucket_start])
                merged = pd.concat([closed, forming_df])
                merged = merged[~merged.index.duplicated(keep="last")]

                # Mirror MarketDataPipeline._merge_forming_candle behavior:
                # recompute RSI only; keep other indicators from closed bars.
                from src.core.indicators import rsi as rsi_series

                merged["rsi"] = rsi_series(merged["close"])
                indicator_cols = [
                    c for c in closed.columns if c not in {"open", "high", "low", "close", "volume", "rsi"}
                ]
                for col in indicator_cols:
                    merged[col] = merged[col].ffill()
                df_out = merged
            else:
                df_out = closed
        else:
            df_out = closed

        if limit is not None:
            df_out = df_out.tail(int(limit))
        return df_out


def _make_coordinator(*, market_data_pipeline: Any, cfg: Dict[str, Any]) -> StrategyCoordinator:
    # Instantiate minimal coordinator: we only call _compute_panic_state().
    pm = type("PM", (), {})()
    pm.cfg = cfg or {}
    rm = type("RM", (), {})()
    return StrategyCoordinator(pm, rm, market_data_pipeline=market_data_pipeline, config=cfg)


async def _pump_pass_for_combo(*, crash_cfg: Dict[str, Any]) -> bool:
    # Reuse the same 2-bar pump scenario as unit test expectations.
    df = pd.DataFrame(
        [
            {"open": 100, "high": 106, "low": 99, "close": 104, "atr": 1.2, "ema_fast": 101},
            {"open": 104, "high": 112, "low": 103, "close": 110, "atr": 1.5, "ema_fast": 102},
        ]
    )

    class PumpMDP:
        async def get_latest_ohlcv(self, symbol: str, timeframe: str, limit: int = None, include_forming: bool = False):
            return df

    coord = _make_coordinator(market_data_pipeline=PumpMDP(), cfg={"strategies": {"adaptive_ob": {"crash_guard": crash_cfg}}})
    is_panic, _ = await coord._compute_panic_state(
        symbol="BTC/USDT:USDT",
        volume_bucket="EXTREME",
        crash_cfg=crash_cfg,
    )
    return bool(is_panic is False)


@dataclass(frozen=True)
class ComboResult:
    panic_gap: float
    extreme_gap: float
    blocked_rate: float
    strict_extreme_share: float
    missing_reclaim_count: int
    pump_pass: bool


async def _evaluate_combo(
    *,
    coordinator: StrategyCoordinator,
    mdp: ReplayMarketDataPipeline,
    df_1m_ind: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    hook_delta: float,
    crash_cfg: Dict[str, Any],
    sample_step_min: int,
) -> ComboResult:
    total = 0
    blocked = 0
    panic_n = 0
    strict_n = 0
    missing_reclaim = 0

    t = window_start.replace(second=0, microsecond=0)
    step = pd.Timedelta(minutes=max(1, int(sample_step_min)))
    while t <= window_end:
        total += 1
        mdp.set_now(t)
        is_panic, panic_meta = await coordinator._compute_panic_state(
            symbol="BTC/USDT:USDT",
            volume_bucket="HIGH",
            crash_cfg=crash_cfg,
        )
        if is_panic:
            panic_n += 1
            ema_gap_atr = None
            try:
                ema_gap_atr = float(panic_meta.get("ema_fast_gap_atr")) if panic_meta.get("ema_fast_gap_atr") is not None else None
            except Exception:
                ema_gap_atr = None

            strict_extreme = False
            try:
                extreme_gap_th = float(crash_cfg.get("extreme_gap_atr_threshold", 0.0) or 0.0)
            except Exception:
                extreme_gap_th = 0.0
            if extreme_gap_th > 0 and ema_gap_atr is not None and ema_gap_atr >= extreme_gap_th:
                strict_extreme = True
            if strict_extreme:
                strict_n += 1

            meta = _compute_reversal_meta(df_1m_ind, now=t, hook_delta=hook_delta)
            rsi_hook = bool(meta.get("rsi_hook"))
            bull_candle = bool(meta.get("bull_candle"))
            reclaim = bool(meta.get("reclaim"))

            if strict_extreme:
                reversal_ok = bool(rsi_hook and reclaim)
                if not reversal_ok:
                    blocked += 1
                    if not reclaim:
                        missing_reclaim += 1
            else:
                reversal_ok = bool(rsi_hook and (bull_candle or reclaim))
                if not reversal_ok:
                    blocked += 1

        t = (pd.Timestamp(t) + step).to_pydatetime().replace(tzinfo=timezone.utc)

    blocked_rate = (blocked / total) if total else 0.0
    strict_share = (strict_n / panic_n) if panic_n else 0.0

    pump_pass = await _pump_pass_for_combo(crash_cfg=crash_cfg)
    return ComboResult(
        panic_gap=float(crash_cfg.get("panic_ema_gap_atr_threshold")),
        extreme_gap=float(crash_cfg.get("extreme_gap_atr_threshold")),
        blocked_rate=float(blocked_rate),
        strict_extreme_share=float(strict_share),
        missing_reclaim_count=int(missing_reclaim),
        pump_pass=bool(pump_pass),
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description="Sensitivity sweep for crash-guard EMA gap thresholds.")
    ap.add_argument("--exchange", default="bingx")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--start-utc", required=True, help="e.g. 2026-01-29T14:55:00Z")
    ap.add_argument("--end-utc", required=True, help="e.g. 2026-01-29T15:20:00Z")
    ap.add_argument("--config", default="config/config.example.yaml")
    ap.add_argument("--sample-step-min", type=int, default=1)
    ap.add_argument("--lookback-hours", type=int, default=6)
    args = ap.parse_args()

    window_start = _parse_iso_utc(args.start_utc)
    window_end = _parse_iso_utc(args.end_utc)
    if window_end <= window_start:
        raise SystemExit("end-utc must be > start-utc")

    cfg = _load_config(Path(args.config))
    ind_cfg = cfg.get("indicators", {}) or {}

    strat_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {})
    hook_delta = float(strat_cfg.get("hook_delta", 1.0) or 1.0)
    base_crash_cfg = (strat_cfg.get("crash_guard") or {}) if isinstance(strat_cfg, dict) else {}
    if isinstance(base_crash_cfg, dict):
        if "panic_volume_buckets" in base_crash_cfg:
            base_crash_cfg["panic_volume_buckets"] = _normalize_inline_list(base_crash_cfg.get("panic_volume_buckets"))

    # Pull baseline values from config where available.
    panic_tf = str(base_crash_cfg.get("panic_tf", "5m") or "5m")
    panic_lookback_bars = int(base_crash_cfg.get("panic_lookback_bars", 3) or 3)
    panic_fast_drop_pct = float(base_crash_cfg.get("panic_fast_drop_pct", 0.008) or 0.008)
    panic_atr_pct = float(base_crash_cfg.get("panic_atr_pct", 0.006) or 0.006)
    panic_bear_body_ratio = float(base_crash_cfg.get("panic_bear_body_ratio", 0.60) or 0.60)

    # Fetch 1m OHLCV with lookback for indicators.
    lookback = pd.Timedelta(hours=max(1, int(args.lookback_hours)))
    since_dt = (pd.Timestamp(window_start) - lookback).to_pydatetime().replace(tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)

    client = CcxtClient(str(args.exchange).lower())
    df_1m_raw = await client.ohlcv(args.symbol, "1m", limit=1440, add_indicators=False, since=since_ms)
    if df_1m_raw is None or df_1m_raw.empty:
        raise SystemExit("Failed to fetch 1m OHLCV; cannot sweep.")
    df_1m_raw = _ensure_utc_index(df_1m_raw)
    df_1m_raw = df_1m_raw.loc[(df_1m_raw.index >= pd.Timestamp(since_dt)) & (df_1m_raw.index <= pd.Timestamp(window_end))]
    df_1m_ind = add_indicators(df_1m_raw, ind_cfg)

    df_5m_raw = _build_5m_from_1m(df_1m_raw)
    df_5m_ind = add_indicators(df_5m_raw, ind_cfg)

    mdp = ReplayMarketDataPipeline(df_1m_raw=df_1m_raw, df_5m_ind=df_5m_ind)

    grid_panic = [2.5, 3.0, 3.5]
    grid_extreme = [4.5, 5.0, 5.5]

    results: List[ComboResult] = []
    for pg in grid_panic:
        for eg in grid_extreme:
            crash_cfg = {
                "enabled": True,
                "panic_volume_buckets": ["HIGH", "EXTREME"],
                "panic_tf": panic_tf,
                "panic_lookback_bars": panic_lookback_bars,
                "panic_fast_drop_pct": panic_fast_drop_pct,
                "panic_atr_pct": panic_atr_pct,
                "panic_bear_body_ratio": panic_bear_body_ratio,
                "panic_ema_gap_atr_threshold": float(pg),
                "extreme_gap_atr_threshold": float(eg),
            }
            coordinator = _make_coordinator(market_data_pipeline=mdp, cfg={"strategies": {"adaptive_ob": {"crash_guard": crash_cfg}}})
            res = await _evaluate_combo(
                coordinator=coordinator,
                mdp=mdp,
                df_1m_ind=df_1m_ind,
                window_start=window_start,
                window_end=window_end,
                hook_delta=hook_delta,
                crash_cfg=crash_cfg,
                sample_step_min=int(args.sample_step_min),
            )
            results.append(res)

    # Print as a compact table.
    print("")
    print("Crash Guard EMA-gap Sensitivity Sweep")
    print(f"Window (UTC): {window_start.isoformat()} -> {window_end.isoformat()} | sample_step_min={int(args.sample_step_min)} | assumed_bucket=HIGH")
    print("")
    print("panic_gap  extreme_gap  blocked_rate  strict_extreme_share  missing_reclaim  pump_pass")
    for r in results:
        print(
            f"{r.panic_gap:7.2f}  {r.extreme_gap:10.2f}  {r.blocked_rate:11.3f}  {r.strict_extreme_share:19.3f}  {r.missing_reclaim_count:14d}  {str(r.pump_pass):>8}"
        )

    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
