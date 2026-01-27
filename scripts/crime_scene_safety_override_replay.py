#!/usr/bin/env python3
"""
Crime-scene replay: validate that AdaptiveShortTheRip would emit a SHORT signal
in a historical UTC window, and that SafetyOverride blocks it.

This is NOT an execution backtest. It is a deterministic signal + veto replay
using historical OHLCV.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from src.core.ccxt_client import CcxtClient
from src.core.indicators import add_indicators
from src.safety.safety_override import SafetyOverride
from src.strategies.adaptive_str import AdaptiveShortTheRip


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_utc(ts: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert("UTC")


def _floor(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    return ts.floor(freq)


def _slice_closed(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.loc[df.index < cutoff].copy()


def _aggregate_partial_from_5m(df5: pd.DataFrame, bucket_start: pd.Timestamp, now: pd.Timestamp) -> Optional[pd.DataFrame]:
    if df5 is None or df5.empty:
        return None
    part = df5.loc[(df5.index >= bucket_start) & (df5.index < now)]
    if part.empty:
        return None
    row = {
        "open": float(part["open"].iloc[0]),
        "high": float(part["high"].max()),
        "low": float(part["low"].min()),
        "close": float(part["close"].iloc[-1]),
        "volume": float(part["volume"].sum()),
    }
    out = pd.DataFrame([row], index=[bucket_start])
    out.index.name = df5.index.name or "timestamp"
    return out


async def _fetch_ohlcv(
    *,
    client: CcxtClient,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    df = await client.ohlcv(symbol, timeframe, limit=limit, add_indicators=False, since=since_ms)
    if df is None or df.empty:
        raise RuntimeError(f"Empty OHLCV: {symbol} {timeframe}")
    logger.info("Fetched %s rows | %s %s | %s -> %s", len(df), symbol, timeframe, df.index.min(), df.index.max())
    return df


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    try:
        if getattr(idx, "tz", None) is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.copy()
            df.index = df.index.tz_convert("UTC")
    except Exception:
        # Fall back to naive-as-UTC assumption.
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def _build_inputs_for_t(
    *,
    t: pd.Timestamp,
    df5: pd.DataFrame,
    df15i: pd.DataFrame,
    df30i: pd.DataFrame,
    df1hi: pd.DataFrame,
    ind_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    is_30m_boundary = bool(t == _floor(t, "30min"))
    cutoff_15m = _floor(t, "15min")
    cutoff_1h = _floor(t, "1h")

    df_15m = _slice_closed(df15i, cutoff_15m)
    df_1h = _slice_closed(df1hi, cutoff_1h)

    if is_30m_boundary:
        df_30m_closed = _slice_closed(df30i, t)
        df_30m_hybrid = df_30m_closed.copy()
        df_30m_hybrid.attrs["includes_forming"] = False
        df_30m_hybrid.attrs["merge_action"] = "sim_closed_only"
        df_30m_hybrid.attrs["fallback_reason"] = None
    else:
        bucket_start = _floor(t, "30min")
        df_30m_closed = _slice_closed(df30i, bucket_start)
        partial = _aggregate_partial_from_5m(df5, bucket_start=bucket_start, now=t)
        if partial is None:
            df_30m_hybrid = df_30m_closed.copy()
            df_30m_hybrid.attrs["includes_forming"] = False
            df_30m_hybrid.attrs["merge_action"] = "sim_no_partial"
            df_30m_hybrid.attrs["fallback_reason"] = "missing_partial_5m"
        else:
            base_ohlcv = df_30m_closed[["open", "high", "low", "close", "volume"]].copy()
            hybrid_ohlcv = pd.concat([base_ohlcv, partial], axis=0)
            hybrid_ohlcv.attrs["timeframe"] = "30m"
            df_30m_hybrid = add_indicators(hybrid_ohlcv, ind_cfg)
            df_30m_hybrid.attrs["includes_forming"] = True
            df_30m_hybrid.attrs["merge_action"] = "sim_partial_5m_rollup"
            df_30m_hybrid.attrs["fallback_reason"] = None
            df_30m_hybrid.attrs["forming_ts"] = int(bucket_start.timestamp() * 1000)
            df_30m_hybrid.attrs["forming_last_update_ts"] = int(t.timestamp() * 1000)
            df_30m_hybrid.attrs["forming_update_age_ms"] = 0

    market_data = {
        "30m_closed": df_30m_closed,
        "30m_hybrid": df_30m_hybrid,
        "15m": df_15m,
        "1h": df_1h,
        "5m": _slice_closed(df5, t),
    }
    return df_30m_closed, market_data, df_1h


async def main() -> int:
    ap = argparse.ArgumentParser(description="Crime-scene replay for SafetyOverride")
    ap.add_argument("--exchange", default="bingx")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--start-utc", required=True, help="e.g. 2026-01-26T23:40:00Z")
    ap.add_argument("--end-utc", required=True, help="e.g. 2026-01-27T05:00:00Z")
    ap.add_argument("--config", default="config/config.example.yaml")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--step-min", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("crime_scene")
    coord_log = logging.getLogger("core.strategy_coordinator")

    window = Window(start=_parse_utc(args.start_utc), end=_parse_utc(args.end_utc))
    if window.end <= window.start:
        raise SystemExit("end_utc must be > start_utc")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ind_cfg = cfg.get("indicators", {}) or {}
    signal_cfg = ((cfg.get("signals") or {}).get("short_the_rip") or {})
    safety_cfg = cfg.get("safety_override", {}) or {}

    if not bool(safety_cfg.get("enabled", False)):
        log.warning("safety_override.enabled is false; this replay cannot validate blocking")

    strategy = AdaptiveShortTheRip(signal_cfg, regime_analyzer=None)
    safety = SafetyOverride(safety_cfg)

    client = CcxtClient(str(args.exchange).lower())

    # Lookback for indicator warmup: fetch enough history so EMA200/RSI aren't NaN.
    start_ms = int(window.start.timestamp() * 1000)
    lookback_30m_ms = int(pd.Timedelta(days=10).total_seconds() * 1000)
    lookback_1h_ms = int(pd.Timedelta(days=15).total_seconds() * 1000)
    lookback_15m_ms = int(pd.Timedelta(days=7).total_seconds() * 1000)
    lookback_5m_ms = int(pd.Timedelta(days=3).total_seconds() * 1000)

    df5 = await _fetch_ohlcv(
        client=client,
        symbol=args.symbol,
        timeframe="5m",
        since_ms=start_ms - lookback_5m_ms,
        limit=1440,
        logger=log,
    )
    df5 = _ensure_utc_index(df5)
    df15 = await _fetch_ohlcv(
        client=client,
        symbol=args.symbol,
        timeframe="15m",
        since_ms=start_ms - lookback_15m_ms,
        limit=800,
        logger=log,
    )
    df15 = _ensure_utc_index(df15)
    df30 = await _fetch_ohlcv(
        client=client,
        symbol=args.symbol,
        timeframe="30m",
        since_ms=start_ms - lookback_30m_ms,
        limit=600,
        logger=log,
    )
    df30 = _ensure_utc_index(df30)
    df1h = await _fetch_ohlcv(
        client=client,
        symbol=args.symbol,
        timeframe="1h",
        since_ms=start_ms - lookback_1h_ms,
        limit=600,
        logger=log,
    )
    df1h = _ensure_utc_index(df1h)

    # Enrich indicators once (we'll slice by time to avoid lookahead).
    df5i = add_indicators(df5, ind_cfg)
    df15i = add_indicators(df15, ind_cfg)
    df30i = add_indicators(df30, ind_cfg)
    df1hi = add_indicators(df1h, ind_cfg)

    step = pd.Timedelta(minutes=int(args.step_min))
    t = window.start.ceil(f"{int(args.step_min)}min")
    seen_signal = False
    blocked = False

    while t <= window.end:
        df_30m_closed, market_data, df_1h_local = _build_inputs_for_t(
            t=t,
            df5=df5i,
            df15i=df15i,
            df30i=df30i,
            df1hi=df1hi,
            ind_cfg=ind_cfg,
        )

        sig = strategy.signal(
            df_30m=df_30m_closed,
            df_1h=df_1h_local,
            regime_data=None,
            symbol=args.symbol,
            market_data=market_data,
            ml_context=None,
        )

        if sig:
            seen_signal = True
            log.info("SIGNAL %s | t=%s | reason=%s", sig.get("side"), t.isoformat(), sig.get("reason"))
            res = safety.check_veto("adaptive_str", sig)
            sig.setdefault("meta", {})["safety_override"] = res.meta_data
            if res.is_vetoed:
                blocked = True
                coord_log.warning(
                    "??  [%s/%s] REJECTED (SafetyOverride): %s | reason=%s score=%s fails=%s",
                    "ADAPTIVE_STR",
                    args.symbol,
                    res.reason,
                    (res.meta_data or {}).get("reason"),
                    (res.meta_data or {}).get("score"),
                    (res.meta_data or {}).get("fails"),
                )
                break

        t = t + step

    if not seen_signal:
        log.warning("No AdaptiveShortTheRip signal detected in window %s -> %s", window.start, window.end)
        return 2
    if not blocked:
        log.warning("Signal detected but SafetyOverride did NOT block it in %s -> %s", window.start, window.end)
        return 3

    log.info("OK: signal detected and blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
