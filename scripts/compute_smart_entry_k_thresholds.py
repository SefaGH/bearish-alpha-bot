"""Compute Smart Entry k-thresholds (max k that still fills) from OHLCV.

For each trade_id:
- parse TRADE_CLOSED and match the nearest prior SIGNAL_BREAKDOWN
- compute ATR_price from vol_atr_bps snapshot
- fetch 1m OHLCV and compute window min(low) / max(high) for each timeout
- derive k_max such that the Smart Entry limit would still be touched

Long:  limit = entry - k*ATR  => touched if low_min <= limit  => k <= (entry - low_min)/ATR
Short: limit = entry + k*ATR  => touched if high_max >= limit => k <= (high_max - entry)/ATR

Analysis-only.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class Row:
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    vol_atr_bps: float | None
    atr_price: float | None
    timeout_min: int
    win_low: float | None
    win_high: float | None
    k_max_fill: float | None


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    try:
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(dt)


def _fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _k_max_for_window(*, entry: float, side: str, atr: float, win_low: float | None, win_high: float | None) -> float | None:
    if atr <= 0 or entry <= 0:
        return None

    side_n = (side or "").strip().lower()
    if side_n in ("long", "buy"):
        if win_low is None:
            return None
        return (entry - float(win_low)) / atr

    if side_n in ("short", "sell"):
        if win_high is None:
            return None
        return (float(win_high) - entry) / atr

    return None


async def compute_rows_for_trade(
    *,
    log_path: Path,
    trade_id: str,
    timeouts: list[int],
    pre_pad_minutes: int,
    cache_dir: Path,
    max_signal_lookback_s: int,
    max_price_delta: float,
):
    import scripts.simulate_min_stop_effect as sim

    trades, signals = sim.parse_log(log_path, {trade_id})
    trade = trades.get(trade_id)
    if trade is None:
        return []

    signal, matched_by = sim.match_signal_for_trade(
        trade=trade,
        signals=signals,
        max_lookback_s=max_signal_lookback_s,
        max_price_delta=max_price_delta,
    )
    if signal is None:
        return []

    vol_atr_bps = signal.vol_atr_bps
    atr_price = None
    if vol_atr_bps is not None:
        atr_price = trade.entry_price * (vol_atr_bps / 10000.0)

    # One OHLCV fetch per trade, up to max timeout.
    max_timeout = max(timeouts) if timeouts else 0
    start = trade.entry_time - timedelta(minutes=pre_pad_minutes)
    end = trade.entry_time + timedelta(minutes=max(max_timeout, 1))
    df = await sim._fetch_ohlcv_1m(symbol=trade.symbol, start=start, end=end, cache_dir=cache_dir)

    rows: list[Row] = []

    if df is None or df.empty or atr_price is None or atr_price <= 0:
        for t in timeouts:
            rows.append(
                Row(
                    trade_id=trade.trade_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_time=_fmt_dt(trade.entry_time),
                    entry_price=float(trade.entry_price),
                    vol_atr_bps=vol_atr_bps,
                    atr_price=atr_price,
                    timeout_min=int(t),
                    win_low=None,
                    win_high=None,
                    k_max_fill=None,
                )
            )
        return rows

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    for t in timeouts:
        window = df[(df.index >= trade.entry_time) & (df.index <= trade.entry_time + timedelta(minutes=int(t)))]
        win_low = None
        win_high = None
        if not window.empty:
            win_low = float(window["low"].min())
            win_high = float(window["high"].max())

        k_max = _k_max_for_window(entry=trade.entry_price, side=trade.side, atr=float(atr_price), win_low=win_low, win_high=win_high)

        rows.append(
            Row(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                side=trade.side,
                entry_time=_fmt_dt(trade.entry_time),
                entry_price=float(trade.entry_price),
                vol_atr_bps=vol_atr_bps,
                atr_price=float(atr_price),
                timeout_min=int(t),
                win_low=win_low,
                win_high=win_high,
                k_max_fill=k_max,
            )
        )

    return rows


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="Compute Smart Entry k_max per trade+timeout")
    ap.add_argument("--log", required=True)
    ap.add_argument("trade_ids", nargs="+")
    ap.add_argument("--timeout", type=int, nargs="*", default=[2, 3, 5, 8])
    ap.add_argument("--pre-pad-minutes", type=int, default=5)
    ap.add_argument("--cache-dir", default="data/cache/ohlcv")
    ap.add_argument("--max-signal-lookback-s", type=int, default=90)
    ap.add_argument("--max-price-delta", type=float, default=200.0)

    args = ap.parse_args()

    log_path = Path(args.log)
    cache_dir = Path(args.cache_dir)
    timeouts = [int(x) for x in args.timeout]

    print(
        " | ".join(
            [
                "trade_id",
                "side",
                "entry_price",
                "atr_price",
                "timeout",
                "win_low",
                "win_high",
                "k_max_fill",
            ]
        )
    )
    print("-" * 120)

    for tid in args.trade_ids:
        rows = await compute_rows_for_trade(
            log_path=log_path,
            trade_id=tid,
            timeouts=timeouts,
            pre_pad_minutes=int(args.pre_pad_minutes),
            cache_dir=cache_dir,
            max_signal_lookback_s=int(args.max_signal_lookback_s),
            max_price_delta=float(args.max_price_delta),
        )
        for r in rows:
            print(
                " | ".join(
                    [
                        r.trade_id,
                        str(r.side),
                        _fmt_num(r.entry_price, 2),
                        _fmt_num(r.atr_price, 4),
                        str(r.timeout_min),
                        _fmt_num(r.win_low, 2),
                        _fmt_num(r.win_high, 2),
                        _fmt_num(r.k_max_fill, 4),
                    ]
                )
            )

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
