"""Simulate the impact of a minimum stop-distance rule for specific trades.

Inputs:
- A live trading log containing:
  - `TRADE_CLOSED {json}` lines (trade_id, entry/exit, etc)
  - `SIGNAL_BREAKDOWN {json}` lines (entry_price/stop_price/target_price + vol_atr_bps)
  - `R/R Analysis` lines (Actual/Required R/R) just before the signal breakdown

What it does:
1) Maps each `trade_id` to the nearest preceding `SIGNAL_BREAKDOWN` (same symbol/strategy/side).
2) Computes a new stop distance:
   new_distance = max(hard_floor_bps, atr_mult * ATR14)
   - ATR14 is taken from `SIGNAL_BREAKDOWN.volatility.vol_atr_bps` when available.
3) Computes the new R/R and flags whether the trade would still pass the required R/R.
4) (Optional) Fetches 1m OHLCV from BingX via the existing `CcxtClient` and checks whether the
   new stop or the original target would have been hit first.

This is designed as an *analysis* tool, not production trading logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

# Ensure repo root is on sys.path so `src.*` imports work when running `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


TRADE_CLOSED_MARKER = "TRADE_CLOSED "
SIGNAL_BREAKDOWN_MARKER = "SIGNAL_BREAKDOWN "


@dataclass(frozen=True)
class TradeClosed:
    trade_id: str
    symbol: str
    timeframe: str
    side: str  # LONG/SHORT
    strategy: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str | None
    position_size: float | None


@dataclass(frozen=True)
class SignalBreakdown:
    signal_id: str
    symbol: str
    strategy: str
    side: str  # long/short
    timestamp: datetime
    entry_price: float
    stop_price: float
    target_price: float
    vol_atr_bps: float | None
    rr_required: float | None
    rr_actual: float | None


@dataclass(frozen=True)
class SimulationResult:
    trade: TradeClosed
    signal: SignalBreakdown | None
    matched_by: str
    entry_mode: str
    simulated_entry_time: datetime | None
    simulated_entry_price: float | None
    smart_limit_price: float | None
    smart_filled: bool | None
    adjusted_stop: float | None
    adjusted_target: float | None
    hard_floor_price: float | None
    atr_price: float | None
    chosen_stop_distance: float | None
    new_stop_price: float | None
    new_rr: float | None
    rr_required: float | None
    would_pass_rr: bool | None
    simulated_exit_reason: str | None
    simulated_exit_time: datetime | None
    simulated_exit_price: float | None
    simulated_pnl_pct: float | None


def _parse_iso8601(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_json_after_marker(line: str, marker: str) -> dict[str, Any] | None:
    idx = line.find(marker)
    if idx == -1:
        return None
    json_part = line[idx + len(marker) :].strip()
    if not json_part.startswith("{"):
        return None
    try:
        payload = json.loads(json_part)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _to_float_required(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"Missing required numeric field: {field}")
    return float(value)


def _to_float_optional(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


_RR_PRICES_RE = re.compile(
    r"Prices:\s*Entry=\$(?P<entry>[0-9.]+),\s*Stop=\$(?P<stop>[0-9.]+).+?,\s*Target=\$(?P<target>[0-9.]+)",
    re.IGNORECASE,
)
_RR_LINE_RE = re.compile(r"R/R:\s*Actual=(?P<actual>[0-9.]+),\s*Required=(?P<required>[0-9.]+)")


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        yield from f


def parse_log(
    log_path: Path,
    trade_ids: set[str],
    signal_symbol: str | None = None,
) -> tuple[dict[str, TradeClosed], list[SignalBreakdown]]:
    trades: dict[str, TradeClosed] = {}
    signals: list[SignalBreakdown] = []

    pending_rr_required: float | None = None
    pending_rr_actual: float | None = None

    for line in _iter_lines(log_path):
        if "TRADE_CLOSED" in line:
            payload = _extract_json_after_marker(line, TRADE_CLOSED_MARKER)
            if payload and payload.get("event") == "TRADE_CLOSED":
                tid = payload.get("trade_id")
                if isinstance(tid, str) and tid in trade_ids:
                    entry_time_s = payload.get("entry_time")
                    exit_time_s = payload.get("exit_time")
                    if isinstance(entry_time_s, str) and isinstance(exit_time_s, str):
                        trade = TradeClosed(
                            trade_id=tid,
                            symbol=str(payload.get("symbol") or ""),
                            timeframe=str(payload.get("timeframe") or ""),
                            side=str(payload.get("side") or ""),
                            strategy=str(payload.get("strategy") or payload.get("strategy_name") or ""),
                            entry_price=_to_float_required(payload.get("entry_price"), "entry_price"),
                            exit_price=_to_float_required(payload.get("exit_price"), "exit_price"),
                            entry_time=_parse_iso8601(entry_time_s),
                            exit_time=_parse_iso8601(exit_time_s),
                            exit_reason=str(payload.get("exit_reason")) if payload.get("exit_reason") is not None else None,
                            position_size=_to_float_optional(payload.get("position_size")),
                        )
                        trades[tid] = trade
                        if len(trades) == len(trade_ids):
                            # Keep scanning because we also want signal breakdowns; but we can
                            # short-circuit if we don't need signals.
                            pass

        # Capture R/R analysis lines (they appear right before signal breakdown)
        if "R/R: Actual=" in line and "Required=" in line:
            m = _RR_LINE_RE.search(line)
            if m:
                try:
                    pending_rr_actual = float(m.group("actual"))
                    pending_rr_required = float(m.group("required"))
                except ValueError:
                    pending_rr_actual = None
                    pending_rr_required = None

        if "SIGNAL_BREAKDOWN" in line:
            payload = _extract_json_after_marker(line, SIGNAL_BREAKDOWN_MARKER)
            if not payload or payload.get("event") != "signal_breakdown":
                continue

            symbol = payload.get("symbol")
            if signal_symbol is not None and isinstance(symbol, str) and symbol != signal_symbol:
                continue

            ts_s = payload.get("timestamp")
            if not isinstance(ts_s, str):
                continue

            vol_atr_bps = None
            vol = payload.get("volatility")
            if isinstance(vol, dict):
                raw_atr_bps = vol.get("vol_atr_bps")
                if isinstance(raw_atr_bps, (int, float)):
                    vol_atr_bps = float(raw_atr_bps)

            try:
                signals.append(
                    SignalBreakdown(
                        signal_id=str(payload.get("signal_id") or ""),
                        symbol=str(symbol or ""),
                        strategy=str(payload.get("strategy") or ""),
                        side=str(payload.get("side") or ""),
                        timestamp=_parse_iso8601(ts_s),
                        entry_price=_to_float_required(payload.get("entry_price"), "entry_price"),
                        stop_price=_to_float_required(payload.get("stop_price"), "stop_price"),
                        target_price=_to_float_required(payload.get("target_price"), "target_price"),
                        vol_atr_bps=vol_atr_bps,
                        rr_required=pending_rr_required,
                        rr_actual=pending_rr_actual,
                    )
                )
            except Exception:
                # Ignore malformed signal breakdown lines
                continue

            # Reset pending rr so we don't accidentally attach it to a later signal.
            pending_rr_required = None
            pending_rr_actual = None

    return trades, signals


def _side_norm(side: str) -> str:
    s = (side or "").strip().lower()
    if s in {"long", "buy"}:
        return "long"
    if s in {"short", "sell"}:
        return "short"
    if s == "long" or s == "short":
        return s
    if s == "":
        return ""
    # trade_closed uses LONG/SHORT
    if s in {"long", "short"}:
        return s
    if s in {"l", "lo"}:
        return "long"
    if s in {"s", "sh"}:
        return "short"
    if s == "long" or s == "short":
        return s
    if s == "long" or s == "short":
        return s
    if s == "long" or s == "short":
        return s
    # Fallback
    if "long" in s:
        return "long"
    if "short" in s:
        return "short"
    return s


def match_signal_for_trade(
    trade: TradeClosed,
    signals: Iterable[SignalBreakdown],
    max_lookback_s: int,
    max_price_delta: float,
) -> tuple[SignalBreakdown | None, str]:
    trade_side = _side_norm(trade.side)

    candidates: list[SignalBreakdown] = []
    for s in signals:
        if s.symbol != trade.symbol:
            continue
        if s.strategy != trade.strategy:
            continue
        if _side_norm(s.side) != trade_side:
            continue
        if s.timestamp > trade.entry_time + timedelta(seconds=2):
            continue
        dt_s = (trade.entry_time - s.timestamp).total_seconds()
        if dt_s < 0 or dt_s > max_lookback_s:
            continue
        if abs(s.entry_price - trade.entry_price) > max_price_delta:
            continue
        candidates.append(s)

    if not candidates:
        return None, "no_candidate"

    def score(s: SignalBreakdown) -> float:
        dt_s = abs((trade.entry_time - s.timestamp).total_seconds())
        price_penalty = abs(s.entry_price - trade.entry_price) / 10.0
        return dt_s + price_penalty

    best = min(candidates, key=score)
    return best, "timestamp+price"


def _pnl_pct(entry: float, exit_price: float, side: str) -> float:
    if entry <= 0:
        return float("nan")
    if _side_norm(side) == "long":
        return (exit_price - entry) / entry * 100.0
    return (entry - exit_price) / entry * 100.0


def compute_min_stop(
    *,
    entry_price: float,
    side: str,
    hard_floor_bps: float,
    atr_bps: float | None,
    atr_mult: float,
) -> tuple[float, float | None, float]:
    hard_floor_price = entry_price * (hard_floor_bps / 10000.0)
    atr_price = None
    if atr_bps is not None:
        atr_price = entry_price * (atr_bps / 10000.0)

    soft_floor = atr_mult * atr_price if atr_price is not None else 0.0
    distance = max(hard_floor_price, soft_floor)

    if _side_norm(side) == "long":
        new_stop = entry_price - distance
    else:
        new_stop = entry_price + distance

    return new_stop, atr_price, hard_floor_price


def compute_smart_limit_price(
    *,
    current_price: float,
    side: str,
    atr_price: float,
    atr_mult: float,
) -> float:
    """Smart Entry: limit = current - ATR*k (long) or current + ATR*k (short)."""
    if atr_price <= 0:
        return 0.0
    if _side_norm(side) == "long":
        return current_price - (atr_price * atr_mult)
    return current_price + (atr_price * atr_mult)


def _find_limit_fill(
    *,
    df_1m: "Any",
    start_time: datetime,
    side: str,
    limit_price: float,
    timeout_minutes: int,
) -> tuple[bool, datetime | None]:
    if df_1m is None or getattr(df_1m, "empty", True) or limit_price <= 0:
        return False, None

    if df_1m.index.tz is None:
        df_1m = df_1m.copy()
        df_1m.index = df_1m.index.tz_localize("UTC")

    end_time = start_time + timedelta(minutes=timeout_minutes)
    window = df_1m[(df_1m.index >= start_time) & (df_1m.index <= end_time)]
    if window.empty:
        return False, None

    side_n = _side_norm(side)
    for ts, row in window.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if side_n == "long":
            if low <= limit_price:
                return True, ts
        else:
            if high >= limit_price:
                return True, ts

    return False, None


async def _fetch_ohlcv_1m(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> "Any | None":
    """Fetch 1m OHLCV using the project CcxtClient; caches to CSV."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    cache_key = f"{safe_symbol}_1m_{start.strftime('%Y%m%dT%H%M%SZ')}_{end.strftime('%Y%m%dT%H%M%SZ')}.csv"
    cache_path = cache_dir / cache_key

    import pandas as pd

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        return df

    # Lazy import to keep script startup fast.
    from src.core.ccxt_client import CcxtClient

    client = CcxtClient("bingx")

    # CCXT `since` is ms. BingX has a max limit; wrapper clamps at 1440.
    tf_ms = 60_000

    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    frames = []
    last_open_ms = None

    # Fetch in chunks; keep going until we cover the window or stop making progress.
    while since_ms < end_ms:
        df = await client.ohlcv(symbol, timeframe="1m", limit=1440, since=since_ms)
        if df is None or df.empty:
            break

        # Normalize timestamps to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        frames.append(df)

        newest = int(df.index.max().timestamp() * 1000)
        if last_open_ms is not None and newest <= last_open_ms:
            break
        last_open_ms = newest

        since_ms = newest + tf_ms

        # Avoid hammering the API.
        await asyncio.sleep(0.15)

    if not frames:
        return None

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Trim to requested window.
    out = out[(out.index >= start) & (out.index <= end)]

    # Persist
    out_to_save = out.reset_index().rename(columns={"index": "timestamp"})
    out_to_save.to_csv(cache_path, index=False)

    return out


def _simulate_first_touch(
    *,
    df_1m: "Any",
    entry_time: datetime,
    side: str,
    stop_price: float,
    target_price: float,
    tie_break: str,
) -> tuple[str | None, datetime | None, float | None]:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None, None, None

    # Ensure UTC tz-aware index
    if df_1m.index.tz is None:
        df_1m = df_1m.copy()
        df_1m.index = df_1m.index.tz_localize("UTC")

    window = df_1m[df_1m.index >= entry_time]
    if window.empty:
        return None, None, None

    side_n = _side_norm(side)
    for ts, row in window.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if side_n == "long":
            stop_hit = low <= stop_price
            tp_hit = high >= target_price
            if stop_hit and tp_hit:
                if tie_break == "tp":
                    return "take_profit", ts, target_price
                return "stop_loss", ts, stop_price
            if stop_hit:
                return "stop_loss", ts, stop_price
            if tp_hit:
                return "take_profit", ts, target_price
        else:
            # short
            stop_hit = high >= stop_price
            tp_hit = low <= target_price
            if stop_hit and tp_hit:
                if tie_break == "tp":
                    return "take_profit", ts, target_price
                return "stop_loss", ts, stop_price
            if stop_hit:
                return "stop_loss", ts, stop_price
            if tp_hit:
                return "take_profit", ts, target_price

    return None, None, None


def _market_price_at_or_after(df_1m: "Any", ts: datetime) -> float | None:
    if df_1m is None or getattr(df_1m, "empty", True):
        return None

    # Ensure UTC tz-aware index
    if df_1m.index.tz is None:
        df_1m = df_1m.copy()
        df_1m.index = df_1m.index.tz_localize("UTC")

    window = df_1m[df_1m.index >= ts]
    if window.empty:
        try:
            return float(df_1m.iloc[-1]["close"])
        except Exception:
            return None

    first = window.iloc[0]
    try:
        return float(first["open"])
    except Exception:
        return None


async def simulate(
    *,
    log_path: Path,
    trade_ids: list[str],
    hard_floor_bps: float,
    atr_mult: float,
    max_signal_lookback_s: int,
    max_price_delta: float,
    warmup_minutes: int,
    pre_pad_minutes: int,
    post_pad_minutes: int,
    max_sim_minutes: int,
    fetch_ohlcv: bool,
    tie_break: str,
    cache_dir: Path,
    ignore_rr: bool,
    scale_target_to_required_rr: bool,
    smart_entry: bool,
    smart_atr_mult: float,
    smart_timeout_minutes: int,
    smart_only_when_atr_bps_gte: float | None,
    smart_market_fallback: bool = False,
    smart_fallback_block_if_stop_hit: bool = True,
    smart_fallback_stop_reference: str = "signal",
    smart_fallback_max_chase_bps: float | None = None,
    smart_fallback_max_chase_bps_long: float | None = None,
    smart_fallback_max_chase_bps_short: float | None = None,
) -> list[SimulationResult]:
    trade_set = {t.strip() for t in trade_ids if t.strip()}
    trades, signals = parse_log(log_path, trade_set)

    results: list[SimulationResult] = []

    for tid in trade_ids:
        trade = trades.get(tid)
        if trade is None:
            continue

        signal, matched_by = match_signal_for_trade(
            trade,
            signals,
            max_lookback_s=max_signal_lookback_s,
            max_price_delta=max_price_delta,
        )

        if signal is None:
            results.append(
                SimulationResult(
                    trade=trade,
                    signal=None,
                    matched_by=matched_by,
                    entry_mode="unknown",
                    simulated_entry_time=None,
                    simulated_entry_price=None,
                    smart_limit_price=None,
                    smart_filled=None,
                    adjusted_stop=None,
                    adjusted_target=None,
                    hard_floor_price=None,
                    atr_price=None,
                    chosen_stop_distance=None,
                    new_stop_price=None,
                    new_rr=None,
                    rr_required=None,
                    would_pass_rr=None,
                    simulated_exit_reason=None,
                    simulated_exit_time=None,
                    simulated_exit_price=None,
                    simulated_pnl_pct=None,
                )
            )
            continue

        # Base: align signal stop/target to the observed filled entry (market case).
        shift = trade.entry_price - signal.entry_price
        stop_adj = signal.stop_price + shift
        target_adj = signal.target_price + shift

        # For smart-entry we will (optionally) override entry/entry_time by simulating a LIMIT fill.
        sim_entry_price = trade.entry_price
        sim_entry_time = trade.entry_time
        entry_mode = "market"
        smart_limit = None
        smart_filled = None
        df_1m_fill = None

        # ATR snapshot from signal breakdown.
        atr_price_snapshot = None
        if signal.vol_atr_bps is not None:
            atr_price_snapshot = trade.entry_price * (signal.vol_atr_bps / 10000.0)

        # Smart Entry decision
        if smart_entry and fetch_ohlcv:
            if atr_price_snapshot is not None and atr_price_snapshot > 0:
                if smart_only_when_atr_bps_gte is None or (
                    signal.vol_atr_bps is not None and signal.vol_atr_bps >= smart_only_when_atr_bps_gte
                ):
                    smart_limit = compute_smart_limit_price(
                        current_price=trade.entry_price,
                        side=trade.side,
                        atr_price=atr_price_snapshot,
                        atr_mult=smart_atr_mult,
                    )

                    # Fetch a focused OHLCV window around entry for fill detection.
                    fill_start = trade.entry_time - timedelta(minutes=pre_pad_minutes)
                    fill_end = trade.entry_time + timedelta(minutes=max(smart_timeout_minutes, 1))
                    df_1m_fill = await _fetch_ohlcv_1m(symbol=trade.symbol, start=fill_start, end=fill_end, cache_dir=cache_dir)
                    smart_filled, fill_ts = _find_limit_fill(
                        df_1m=df_1m_fill,
                        start_time=trade.entry_time,
                        side=trade.side,
                        limit_price=smart_limit,
                        timeout_minutes=smart_timeout_minutes,
                    )
                    if smart_filled and fill_ts is not None:
                        entry_mode = "smart_limit"
                        sim_entry_price = smart_limit
                        sim_entry_time = fill_ts
                        # When entry changes, keep target as the ORIGINAL absolute target (signal-derived)
                        # to reflect mean-reversion thesis (better entry increases reward).
                        target_adj = signal.target_price
                        stop_adj = signal.stop_price
                    else:
                        # Not filled -> trade would not open under smart-entry policy.
                        entry_mode = "smart_limit_no_fill"

        # If smart entry is enabled and not filled, we can either short-circuit or
        # (optionally) enter at market at timeout ("market fallback").
        if entry_mode == "smart_limit_no_fill":
            fallback_reason = "no_fill"

            if smart_market_fallback and fetch_ohlcv and df_1m_fill is not None:
                fallback_time = trade.entry_time + timedelta(minutes=max(smart_timeout_minutes, 1))
                fallback_entry_price = _market_price_at_or_after(df_1m_fill, fallback_time)

                if fallback_entry_price is not None:
                    # Chase gate: avoid entering too far away from the signal entry.
                    side_n = _side_norm(trade.side)
                    max_bps_side = None
                    if side_n == "long" and smart_fallback_max_chase_bps_long is not None:
                        max_bps_side = float(smart_fallback_max_chase_bps_long)
                    elif side_n == "short" and smart_fallback_max_chase_bps_short is not None:
                        max_bps_side = float(smart_fallback_max_chase_bps_short)
                    elif smart_fallback_max_chase_bps is not None:
                        # Backward-compatible global value.
                        max_bps_side = float(smart_fallback_max_chase_bps)

                    if max_bps_side is not None and max_bps_side >= 0:
                        sig_entry = float(signal.entry_price)
                        if side_n == "long":
                            if float(fallback_entry_price) > sig_entry * (1.0 + (max_bps_side / 10000.0)):
                                fallback_reason = "no_fill_chase_gate"
                                results.append(
                                    SimulationResult(
                                        trade=trade,
                                        signal=signal,
                                        matched_by=matched_by,
                                        entry_mode=entry_mode,
                                        simulated_entry_time=None,
                                        simulated_entry_price=None,
                                        smart_limit_price=smart_limit,
                                        smart_filled=False,
                                        adjusted_stop=stop_adj,
                                        adjusted_target=target_adj,
                                        hard_floor_price=None,
                                        atr_price=atr_price_snapshot,
                                        chosen_stop_distance=None,
                                        new_stop_price=None,
                                        new_rr=None,
                                        rr_required=signal.rr_required,
                                        would_pass_rr=None,
                                        simulated_exit_reason=fallback_reason,
                                        simulated_exit_time=None,
                                        simulated_exit_price=None,
                                        simulated_pnl_pct=None,
                                    )
                                )
                                continue
                        else:
                            if float(fallback_entry_price) < sig_entry * (1.0 - (max_bps_side / 10000.0)):
                                fallback_reason = "no_fill_chase_gate"
                                results.append(
                                    SimulationResult(
                                        trade=trade,
                                        signal=signal,
                                        matched_by=matched_by,
                                        entry_mode=entry_mode,
                                        simulated_entry_time=None,
                                        simulated_entry_price=None,
                                        smart_limit_price=smart_limit,
                                        smart_filled=False,
                                        adjusted_stop=stop_adj,
                                        adjusted_target=target_adj,
                                        hard_floor_price=None,
                                        atr_price=atr_price_snapshot,
                                        chosen_stop_distance=None,
                                        new_stop_price=None,
                                        new_rr=None,
                                        rr_required=signal.rr_required,
                                        would_pass_rr=None,
                                        simulated_exit_reason=fallback_reason,
                                        simulated_exit_time=None,
                                        simulated_exit_price=None,
                                        simulated_pnl_pct=None,
                                    )
                                )
                                continue

                    preview_stop, preview_atr_price, preview_hard_floor = compute_min_stop(
                        entry_price=float(fallback_entry_price),
                        side=trade.side,
                        hard_floor_bps=hard_floor_bps,
                        atr_bps=signal.vol_atr_bps,
                        atr_mult=atr_mult,
                    )

                    wait_window = df_1m_fill[(df_1m_fill.index >= trade.entry_time) & (df_1m_fill.index <= fallback_time)]
                    stop_hit_before_timeout = False
                    if wait_window is not None and not getattr(wait_window, "empty", True):
                        stop_ref = (smart_fallback_stop_reference or "signal").strip().lower()
                        if stop_ref == "min_stop":
                            stop_level = float(preview_stop)
                        else:
                            # Default: signal stop aligned to the observed entry context.
                            stop_level = float(stop_adj)

                        if _side_norm(trade.side) == "long":
                            stop_hit_before_timeout = float(wait_window["low"].min()) <= stop_level
                        else:
                            stop_hit_before_timeout = float(wait_window["high"].max()) >= stop_level

                    if smart_fallback_block_if_stop_hit and stop_hit_before_timeout:
                        fallback_reason = "no_fill_stop_hit"
                        results.append(
                            SimulationResult(
                                trade=trade,
                                signal=signal,
                                matched_by=matched_by,
                                entry_mode=entry_mode,
                                simulated_entry_time=None,
                                simulated_entry_price=None,
                                smart_limit_price=smart_limit,
                                smart_filled=False,
                                adjusted_stop=stop_adj,
                                adjusted_target=target_adj,
                                hard_floor_price=preview_hard_floor,
                                atr_price=preview_atr_price,
                                chosen_stop_distance=None,
                                new_stop_price=preview_stop,
                                new_rr=None,
                                rr_required=signal.rr_required,
                                would_pass_rr=None,
                                simulated_exit_reason=fallback_reason,
                                simulated_exit_time=None,
                                simulated_exit_price=None,
                                simulated_pnl_pct=None,
                            )
                        )
                        continue

                    # Market fallback entry accepted
                    entry_mode = "smart_timeout_market"
                    smart_filled = False
                    sim_entry_price = float(fallback_entry_price)
                    sim_entry_time = fallback_time

                    # Align stop/target to the market entry (like the market case).
                    shift_fb = sim_entry_price - signal.entry_price
                    stop_adj = signal.stop_price + shift_fb
                    target_adj = signal.target_price + shift_fb

            # Still no entry after fallback attempt.
            if entry_mode == "smart_limit_no_fill":
                results.append(
                    SimulationResult(
                        trade=trade,
                        signal=signal,
                        matched_by=matched_by,
                        entry_mode=entry_mode,
                        simulated_entry_time=None,
                        simulated_entry_price=None,
                        smart_limit_price=smart_limit,
                        smart_filled=False,
                        adjusted_stop=stop_adj,
                        adjusted_target=target_adj,
                        hard_floor_price=None,
                        atr_price=atr_price_snapshot,
                        chosen_stop_distance=None,
                        new_stop_price=None,
                        new_rr=None,
                        rr_required=signal.rr_required,
                        would_pass_rr=None,
                        simulated_exit_reason=fallback_reason,
                        simulated_exit_time=None,
                        simulated_exit_price=None,
                        simulated_pnl_pct=None,
                    )
                )
                continue

        new_stop, atr_price, hard_floor_price = compute_min_stop(
            entry_price=sim_entry_price,
            side=trade.side,
            hard_floor_bps=hard_floor_bps,
            atr_bps=signal.vol_atr_bps,
            atr_mult=atr_mult,
        )

        chosen_distance = abs(sim_entry_price - new_stop)
        rr_required = signal.rr_required

        # Optionally move target to preserve the originally required R/R.
        # This is useful to model the "stop floor" change without automatically failing the R/R gate.
        if scale_target_to_required_rr and rr_required is not None and chosen_distance > 0:
            reward_needed = rr_required * chosen_distance
            if _side_norm(trade.side) == "long":
                target_adj = sim_entry_price + reward_needed
            else:
                target_adj = sim_entry_price - reward_needed

        reward = abs(target_adj - sim_entry_price)
        risk = chosen_distance
        new_rr = (reward / risk) if risk > 0 else float("inf")

        would_pass_rr = None
        if rr_required is not None:
            would_pass_rr = new_rr >= rr_required

        sim_reason = None
        sim_time = None
        sim_price = None
        sim_pnl = None

        should_simulate_path = fetch_ohlcv and (ignore_rr or would_pass_rr is None or would_pass_rr is True)
        if should_simulate_path:
            start = sim_entry_time - timedelta(minutes=warmup_minutes + pre_pad_minutes)
            end_by_exit = trade.exit_time + timedelta(minutes=post_pad_minutes)
            end_by_hold = sim_entry_time + timedelta(minutes=max_sim_minutes)
            end = max(end_by_exit, end_by_hold)
            df_1m = await _fetch_ohlcv_1m(symbol=trade.symbol, start=start, end=end, cache_dir=cache_dir)
            sim_reason, sim_time, sim_price = _simulate_first_touch(
                df_1m=df_1m,
                entry_time=sim_entry_time,
                side=trade.side,
                stop_price=new_stop,
                target_price=target_adj,
                tie_break=tie_break,
            )
            if sim_price is not None:
                sim_pnl = _pnl_pct(sim_entry_price, sim_price, trade.side)

        results.append(
            SimulationResult(
                trade=trade,
                signal=signal,
                matched_by=matched_by,
                entry_mode=entry_mode,
                simulated_entry_time=sim_entry_time,
                simulated_entry_price=sim_entry_price,
                smart_limit_price=smart_limit,
                smart_filled=smart_filled,
                adjusted_stop=stop_adj,
                adjusted_target=target_adj,
                hard_floor_price=hard_floor_price,
                atr_price=atr_price,
                chosen_stop_distance=chosen_distance,
                new_stop_price=new_stop,
                new_rr=new_rr,
                rr_required=rr_required,
                would_pass_rr=would_pass_rr,
                simulated_exit_reason=sim_reason,
                simulated_exit_time=sim_time,
                simulated_exit_price=sim_price,
                simulated_pnl_pct=sim_pnl,
            )
        )

    return results


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe(x: float | None) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return f"{x:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate minimum stop-distance effect for specific trade_ids.")
    parser.add_argument("trade_ids", nargs="+", help="trade_id values")
    parser.add_argument("--log", required=True, help="Path to the live trading log to analyze")

    parser.add_argument("--hard-floor-bps", type=float, default=15.0, help="Hard minimum stop distance in bps (default: 15)")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="ATR multiplier for the soft floor (default: 1.5)")

    parser.add_argument("--max-signal-lookback-s", type=int, default=90, help="Max seconds to look back for matching signal (default: 90)")
    parser.add_argument("--max-price-delta", type=float, default=200.0, help="Max abs(entry price delta) for matching (default: 200)")

    parser.add_argument("--warmup-minutes", type=int, default=90)
    parser.add_argument("--pre-pad-minutes", type=int, default=5)
    parser.add_argument("--post-pad-minutes", type=int, default=10)
    parser.add_argument(
        "--max-sim-minutes",
        type=int,
        default=30,
        help="Max simulation horizon after entry when fetching OHLCV (default: 30).",
    )

    parser.add_argument(
        "--smart-entry",
        action="store_true",
        help="Enable Smart Entry limit-fill simulation: limit = current ± ATR*k, fill within timeout.",
    )
    parser.add_argument(
        "--smart-atr-mult",
        type=float,
        default=0.5,
        help="Smart Entry limit offset multiplier (default: 0.5).",
    )
    parser.add_argument(
        "--smart-timeout-minutes",
        type=int,
        default=15,
        help="Max minutes after signal to wait for LIMIT fill (default: 15).",
    )
    parser.add_argument(
        "--smart-only-when-atr-bps-gte",
        type=float,
        default=None,
        help="Only apply Smart Entry when signal vol_atr_bps >= threshold (optional).",
    )
    parser.add_argument(
        "--smart-market-fallback",
        action="store_true",
        help="If the smart LIMIT is not filled by timeout, enter at MARKET at timeout (analysis-only).",
    )
    parser.add_argument(
        "--smart-fallback-allow-stop-hit",
        action="store_true",
        help="Allow market fallback even if stop would have been hit before timeout (not recommended).",
    )
    parser.add_argument(
        "--smart-fallback-stop-ref",
        choices=["signal", "min_stop"],
        default="signal",
        help="When using --smart-market-fallback, define the stop level used to decide if the setup was invalidated before timeout.",
    )
    parser.add_argument(
        "--max-chase-bps",
        type=float,
        default=None,
        help="When using --smart-market-fallback, abort if timeout market price is too far beyond signal entry (in bps).",
    )
    parser.add_argument(
        "--long-max-chase-bps",
        type=float,
        default=None,
        help="Like --max-chase-bps but only for LONG fallback.",
    )
    parser.add_argument(
        "--short-max-chase-bps",
        type=float,
        default=None,
        help="Like --max-chase-bps but only for SHORT fallback.",
    )

    parser.add_argument("--no-ohlcv", action="store_true", help="Skip OHLCV fetch; only compute new stop + R/R")
    parser.add_argument(
        "--ignore-rr",
        action="store_true",
        help="Fetch/simulate OHLCV even if the widened stop would fail the R/R rule.",
    )
    parser.add_argument(
        "--scale-target-to-required-rr",
        action="store_true",
        help="If required R/R is known, move target to maintain it after widening the stop.",
    )
    parser.add_argument("--tie-break", choices=["stop", "tp"], default="stop", help="If stop+tp hit same candle, choose which wins")

    parser.add_argument("--cache-dir", default="data/cache/ohlcv", help="OHLCV cache directory")

    args = parser.parse_args()

    log_path = Path(args.log)
    cache_dir = Path(args.cache_dir)

    results = asyncio.run(
        simulate(
            log_path=log_path,
            trade_ids=args.trade_ids,
            hard_floor_bps=args.hard_floor_bps,
            atr_mult=args.atr_mult,
            max_signal_lookback_s=args.max_signal_lookback_s,
            max_price_delta=args.max_price_delta,
            warmup_minutes=args.warmup_minutes,
            pre_pad_minutes=args.pre_pad_minutes,
            post_pad_minutes=args.post_pad_minutes,
            max_sim_minutes=args.max_sim_minutes,
            fetch_ohlcv=not args.no_ohlcv,
            tie_break=args.tie_break,
            cache_dir=cache_dir,
            ignore_rr=bool(args.ignore_rr),
            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
            smart_entry=bool(args.smart_entry),
            smart_atr_mult=float(args.smart_atr_mult),
            smart_timeout_minutes=int(args.smart_timeout_minutes),
            smart_only_when_atr_bps_gte=(None if args.smart_only_when_atr_bps_gte is None else float(args.smart_only_when_atr_bps_gte)),
            smart_market_fallback=bool(args.smart_market_fallback),
            smart_fallback_block_if_stop_hit=(not bool(args.smart_fallback_allow_stop_hit)),
            smart_fallback_stop_reference=str(args.smart_fallback_stop_ref),
            smart_fallback_max_chase_bps=(None if args.max_chase_bps is None else float(args.max_chase_bps)),
            smart_fallback_max_chase_bps_long=(None if args.long_max_chase_bps is None else float(args.long_max_chase_bps)),
            smart_fallback_max_chase_bps_short=(None if args.short_max_chase_bps is None else float(args.short_max_chase_bps)),
        )
    )

    # Output table
    cols = [
        "trade_id",
        "entry_mode",
        "sim_entry_time",
        "sim_entry_price",
        "limit_price",
        "side",
        "entry_time",
        "exit_reason",
        "actual_pnl_pct",
        "sig_id",
        "sig_ts",
        "rr_req",
        "new_rr",
        "pass_rr",
        "new_stop",
        "target",
        "atr_price",
        "hard_floor",
        "sim_exit_reason",
        "sim_exit_time",
        "sim_exit_price",
        "sim_pnl_pct",
    ]
    print(" | ".join(cols))
    print("-" * 180)

    for r in results:
        t = r.trade
        actual_pnl = _pnl_pct(t.entry_price, t.exit_price, t.side)
        sig = r.signal
        print(
            " | ".join(
                [
                    t.trade_id,
                    r.entry_mode,
                    _fmt_dt(r.simulated_entry_time),
                    _safe(r.simulated_entry_price),
                    _safe(r.smart_limit_price),
                    _side_norm(t.side),
                    _fmt_dt(t.entry_time),
                    t.exit_reason or "",
                    f"{actual_pnl:.4f}",
                    (sig.signal_id if sig else ""),
                    (_fmt_dt(sig.timestamp) if sig else ""),
                    _safe(r.rr_required),
                    _safe(r.new_rr),
                    ("" if r.would_pass_rr is None else str(bool(r.would_pass_rr))),
                    _safe(r.new_stop_price),
                    _safe(r.adjusted_target),
                    _safe(r.atr_price),
                    _safe(r.hard_floor_price),
                    r.simulated_exit_reason or "",
                    _fmt_dt(r.simulated_exit_time),
                    _safe(r.simulated_exit_price),
                    ("" if r.simulated_pnl_pct is None else f"{r.simulated_pnl_pct:.4f}"),
                ]
            )
        )

    missing = [tid for tid in args.trade_ids if tid not in {r.trade.trade_id for r in results}]
    if missing:
        print("\nMissing trade_ids (no TRADE_CLOSED found in this log):")
        for tid in missing:
            print(f"- {tid}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
