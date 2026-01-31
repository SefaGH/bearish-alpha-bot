"""Diagnose Smart Entry no-fill cases using log + 1m OHLCV.

Given trade_ids, this script:
- parses `TRADE_CLOSED` + matching `SIGNAL_BREAKDOWN`
- computes Smart Entry limit price: limit = current ± ATR*k
- fetches 1m OHLCV around the entry timestamp
- reports whether/when the limit would have been touched within each timeout

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
class Diagnosis:
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    vol_atr_bps: float | None
    atr_price: float | None
    k: float
    limit_price: float | None
    timeout_min: int
    touched: bool
    first_touch_time: str | None
    window_low: float | None
    window_high: float | None


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    try:
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(dt)


async def diagnose_one(
    *,
    log_path: Path,
    trade_id: str,
    k: float,
    timeout_minutes: int,
    pre_pad_minutes: int,
    cache_dir: Path,
    max_signal_lookback_s: int,
    max_price_delta: float,
):
    import scripts.simulate_min_stop_effect as sim

    trades, signals = sim.parse_log(log_path, {trade_id})
    trade = trades.get(trade_id)
    if trade is None:
        raise SystemExit(f"Trade not found in log: {trade_id}")

    # Match signal using existing helper.
    signal, matched_by = sim.match_signal_for_trade(
        trade=trade,
        signals=signals,
        max_lookback_s=max_signal_lookback_s,
        max_price_delta=max_price_delta,
    )
    if signal is None:
        raise SystemExit(f"No matching SIGNAL_BREAKDOWN found for trade_id={trade_id} (matched_by={matched_by})")

    vol_atr_bps = signal.vol_atr_bps
    atr_price = None
    if vol_atr_bps is not None:
        atr_price = trade.entry_price * (vol_atr_bps / 10000.0)

    limit_price = None
    if atr_price is not None and atr_price > 0:
        limit_price = sim.compute_smart_limit_price(
            current_price=trade.entry_price,
            side=trade.side,
            atr_price=atr_price,
            atr_mult=k,
        )

    # Fetch OHLCV and check touch.
    start = trade.entry_time - timedelta(minutes=pre_pad_minutes)
    end = trade.entry_time + timedelta(minutes=max(timeout_minutes, 1))
    df = await sim._fetch_ohlcv_1m(symbol=trade.symbol, start=start, end=end, cache_dir=cache_dir)

    touched = False
    first_touch_time = None
    win_low = None
    win_high = None

    if df is not None and not df.empty:
        # Focus on [entry_time, entry_time+timeout]
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        window = df[(df.index >= trade.entry_time) & (df.index <= trade.entry_time + timedelta(minutes=timeout_minutes))]
        if not window.empty:
            win_low = float(window["low"].min())
            win_high = float(window["high"].max())

        if limit_price is not None:
            touched, ts = sim._find_limit_fill(
                df_1m=df,
                start_time=trade.entry_time,
                side=trade.side,
                limit_price=limit_price,
                timeout_minutes=timeout_minutes,
            )
            first_touch_time = _fmt_dt(ts)

    return Diagnosis(
        trade_id=trade.trade_id,
        symbol=trade.symbol,
        side=trade.side,
        entry_time=_fmt_dt(trade.entry_time),
        entry_price=float(trade.entry_price),
        vol_atr_bps=vol_atr_bps,
        atr_price=atr_price,
        k=float(k),
        limit_price=limit_price,
        timeout_min=int(timeout_minutes),
        touched=bool(touched),
        first_touch_time=first_touch_time,
        window_low=win_low,
        window_high=win_high,
    )


def _p(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.6f}" if abs(v) < 100 else f"{v:.2f}"
    return str(v)


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="Diagnose Smart Entry no-fill using OHLCV")
    ap.add_argument("--log", required=True)
    ap.add_argument("trade_ids", nargs="+")
    ap.add_argument(
        "--k",
        type=float,
        nargs="*",
        default=[1.0],
        help="Smart Entry ATR multipliers to test (default: 1.0)",
    )
    ap.add_argument("--timeout", type=int, nargs="*", default=[2, 3, 5])
    ap.add_argument("--pre-pad-minutes", type=int, default=5)
    ap.add_argument("--cache-dir", default="data/cache/ohlcv")
    ap.add_argument("--max-signal-lookback-s", type=int, default=90)
    ap.add_argument("--max-price-delta", type=float, default=200.0)

    args = ap.parse_args()

    log_path = Path(args.log)
    cache_dir = Path(args.cache_dir)

    print(
        " | ".join(
            [
                "trade_id",
                "symbol",
                "side",
                "entry_time",
                "entry_price",
                "vol_atr_bps",
                "atr_price",
                "k",
                "limit_price",
                "timeout",
                "touched",
                "first_touch_time",
                "win_low",
                "win_high",
            ]
        )
    )
    print("-" * 140)

    for tid in args.trade_ids:
        for k in args.k:
            for t in args.timeout:
                d = await diagnose_one(
                    log_path=log_path,
                    trade_id=tid,
                    k=float(k),
                    timeout_minutes=int(t),
                    pre_pad_minutes=int(args.pre_pad_minutes),
                    cache_dir=cache_dir,
                    max_signal_lookback_s=int(args.max_signal_lookback_s),
                    max_price_delta=float(args.max_price_delta),
                )
                print(
                    " | ".join(
                        [
                            d.trade_id,
                            d.symbol,
                            str(d.side),
                            d.entry_time,
                            _p(d.entry_price),
                            _p(d.vol_atr_bps),
                            _p(d.atr_price),
                            _p(d.k),
                            _p(d.limit_price),
                            str(d.timeout_min),
                            str(d.touched),
                            d.first_touch_time or "",
                            _p(d.window_low),
                            _p(d.window_high),
                        ]
                    )
                )

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
