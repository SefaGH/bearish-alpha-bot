"""Sweep Smart Entry parameters on a fixed set of trade_ids.

This is an analysis helper around `scripts/simulate_min_stop_effect.py`.

It runs multiple combinations of:
- smart ATR multiplier (k)
- limit fill timeout (minutes)
- optional volatility threshold (vol_atr_bps)

and summarizes:
- fill rate
- TP/SL outcomes (based on OHLCV path simulation)
- mean simulated PnL%

NOTE: This does NOT change production trading logic.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path


# Ensure repo root is on sys.path so `scripts.*` imports work when running `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class Summary:
    smart_atr_mult: float
    smart_timeout_minutes: int
    smart_only_when_atr_bps_gte: float | None
    scale_target_to_required_rr: bool
    trades: int
    filled: int
    no_fill: int
    tp: int
    sl: int
    time_exit: int
    other_exit: int
    mean_sim_pnl_pct: float | None


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def summarize(results) -> tuple[int, int, int, int, int, int, int, float | None]:
    smart_applied = 0
    filled = 0
    no_fill = 0
    market = 0
    tp = 0
    sl = 0
    time_exit = 0
    other = 0
    pnls: list[float] = []

    for r in results:
        if r.entry_mode == "smart_limit":
            smart_applied += 1
            filled += 1
        elif r.entry_mode == "smart_limit_no_fill":
            smart_applied += 1
            no_fill += 1
        else:
            market += 1

        reason = (r.simulated_exit_reason or "").lower()
        if reason == "take_profit":
            tp += 1
        elif reason == "stop_loss":
            sl += 1
        elif reason == "time_exit":
            time_exit += 1
        elif reason:
            other += 1

        if r.simulated_pnl_pct is not None:
            pnls.append(float(r.simulated_pnl_pct))

    return smart_applied, filled, no_fill, market, tp, sl, time_exit, other, len(pnls), _mean(pnls)


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


async def run_one(
    *,
    log_path: Path,
    trade_ids: list[str],
    hard_floor_bps: float,
    atr_mult: float,
    max_sim_minutes: int,
    ignore_rr: bool,
    tie_break: str,
    cache_dir: Path,
    smart_atr_mult: float,
    smart_timeout_minutes: int,
    smart_only_when_atr_bps_gte: float | None,
    scale_target_to_required_rr: bool,
):
    # Import lazily so this script can be run from repo root.
    import scripts.simulate_min_stop_effect as sim

    return await sim.simulate(
        log_path=log_path,
        trade_ids=set(trade_ids),
        hard_floor_bps=hard_floor_bps,
        atr_mult=atr_mult,
        max_signal_lookback_s=90,
        max_price_delta=200.0,
        warmup_minutes=60,
        pre_pad_minutes=5,
        post_pad_minutes=5,
        max_sim_minutes=max_sim_minutes,
        fetch_ohlcv=True,
        tie_break=tie_break,
        cache_dir=cache_dir,
        ignore_rr=ignore_rr,
        scale_target_to_required_rr=scale_target_to_required_rr,
        smart_entry=True,
        smart_atr_mult=smart_atr_mult,
        smart_timeout_minutes=smart_timeout_minutes,
        smart_only_when_atr_bps_gte=smart_only_when_atr_bps_gte,
    )


async def run_baseline(
    *,
    log_path: Path,
    trade_ids: list[str],
    hard_floor_bps: float,
    atr_mult: float,
    max_sim_minutes: int,
    ignore_rr: bool,
    tie_break: str,
    cache_dir: Path,
    scale_target_to_required_rr: bool,
):
    import scripts.simulate_min_stop_effect as sim

    return await sim.simulate(
        log_path=log_path,
        trade_ids=set(trade_ids),
        hard_floor_bps=hard_floor_bps,
        atr_mult=atr_mult,
        max_signal_lookback_s=90,
        max_price_delta=200.0,
        warmup_minutes=60,
        pre_pad_minutes=5,
        post_pad_minutes=5,
        max_sim_minutes=max_sim_minutes,
        fetch_ohlcv=True,
        tie_break=tie_break,
        cache_dir=cache_dir,
        ignore_rr=ignore_rr,
        scale_target_to_required_rr=scale_target_to_required_rr,
        smart_entry=False,
        smart_atr_mult=0.5,
        smart_timeout_minutes=15,
        smart_only_when_atr_bps_gte=None,
    )


def _index_by_trade_id(results) -> dict[str, object]:
    out: dict[str, object] = {}
    for r in results:
        tid = getattr(getattr(r, "trade", None), "trade_id", None)
        if isinstance(tid, str):
            out[tid] = r
    return out


def _pnl(r) -> float | None:
    v = getattr(r, "simulated_pnl_pct", None)
    return None if v is None else float(v)


def _reason(r) -> str:
    return str(getattr(r, "simulated_exit_reason", "") or "")


_WORSE_EPS_PNL_PCT = 1e-4  # 0.0001%: ignore float-noise deltas


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep Smart Entry params using OHLCV simulation")
    p.add_argument("--log", required=True, help="Path to the live trading log")
    p.add_argument("trade_ids", nargs="+", help="Trade IDs to analyze")

    p.add_argument("--hard-floor-bps", type=float, default=15.0)
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--max-sim-minutes", type=int, default=90)
    p.add_argument("--ignore-rr", action="store_true")
    p.add_argument("--tie-break", choices=["stop", "tp"], default="stop")
    p.add_argument("--cache-dir", default="data/cache/ohlcv")

    p.add_argument(
        "--policy",
        choices=["directional_hybrid_safe"],
        default=None,
        help=(
            "Run a side-aware Smart Entry policy instead of a full grid sweep. "
            "directional_hybrid_safe: LONG k=0.90 timeout=4m, SHORT k=0.85 timeout=3m, vol_thr=5.0 bps"
        ),
    )

    p.add_argument("--k", type=float, nargs="*", default=[0.25, 0.5, 1.0], help="smart_atr_mult values")
    p.add_argument("--timeout", type=int, nargs="*", default=[3, 5, 10, 15], help="timeout minutes")
    p.add_argument(
        "--vol-threshold",
        type=float,
        nargs="*",
        default=[None],
        help="Only apply Smart Entry when vol_atr_bps >= threshold. Use e.g. 10 12 15.",
    )
    p.add_argument(
        "--scale-target-to-required-rr",
        action="store_true",
        help="Preserve required R/R by moving target after widening stop",
    )

    p.add_argument(
        "--details",
        action="store_true",
        help="Print per-trade deltas vs baseline for each combo (no_fill and worse-than-baseline lists).",
    )
    p.add_argument(
        "--details-max-combos",
        type=int,
        default=50,
        help="Safety limit for --details output (default: 50 combos).",
    )

    args = p.parse_args()

    log_path = Path(args.log)
    cache_dir = Path(args.cache_dir)

    trade_ids = list(args.trade_ids)

    baseline_results = asyncio.run(
        run_baseline(
            log_path=log_path,
            trade_ids=trade_ids,
            hard_floor_bps=float(args.hard_floor_bps),
            atr_mult=float(args.atr_mult),
            max_sim_minutes=int(args.max_sim_minutes),
            ignore_rr=bool(args.ignore_rr),
            tie_break=str(args.tie_break),
            cache_dir=cache_dir,
            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
        )
    )
    baseline_by_id = _index_by_trade_id(baseline_results)

    if args.policy is not None:
        b_smart_applied, b_filled, b_no_fill, b_market, b_tp, b_sl, b_time_exit, b_other, b_pnl_n, b_mean_pnl = summarize(
            baseline_results
        )
        _ = (b_other, b_pnl_n)

        # Side-aware policy execution: split trade_ids by side, run per-side config.
        import scripts.simulate_min_stop_effect as sim

        trades, _signals = sim.parse_log(log_path, set(trade_ids))
        long_trade_ids: list[str] = []
        short_trade_ids: list[str] = []
        unknown_trade_ids: list[str] = []

        for tid in trade_ids:
            t = trades.get(tid)
            if t is None:
                unknown_trade_ids.append(tid)
                continue
            side = str(getattr(t, "side", "") or "").upper()
            if side == "LONG":
                long_trade_ids.append(tid)
            elif side == "SHORT":
                short_trade_ids.append(tid)
            else:
                unknown_trade_ids.append(tid)

        if args.policy == "directional_hybrid_safe":
            vol_thr_val = 5.0
            long_k = 0.90
            long_timeout = 4
            short_k = 0.85
            short_timeout = 3

            policy_results = []
            if long_trade_ids:
                policy_results.extend(
                    asyncio.run(
                        run_one(
                            log_path=log_path,
                            trade_ids=long_trade_ids,
                            hard_floor_bps=float(args.hard_floor_bps),
                            atr_mult=float(args.atr_mult),
                            max_sim_minutes=int(args.max_sim_minutes),
                            ignore_rr=bool(args.ignore_rr),
                            tie_break=str(args.tie_break),
                            cache_dir=cache_dir,
                            smart_atr_mult=float(long_k),
                            smart_timeout_minutes=int(long_timeout),
                            smart_only_when_atr_bps_gte=float(vol_thr_val),
                            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                        )
                    )
                )
            if short_trade_ids:
                policy_results.extend(
                    asyncio.run(
                        run_one(
                            log_path=log_path,
                            trade_ids=short_trade_ids,
                            hard_floor_bps=float(args.hard_floor_bps),
                            atr_mult=float(args.atr_mult),
                            max_sim_minutes=int(args.max_sim_minutes),
                            ignore_rr=bool(args.ignore_rr),
                            tie_break=str(args.tie_break),
                            cache_dir=cache_dir,
                            smart_atr_mult=float(short_k),
                            smart_timeout_minutes=int(short_timeout),
                            smart_only_when_atr_bps_gte=float(vol_thr_val),
                            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                        )
                    )
                )
            if unknown_trade_ids:
                # If we can't determine side, avoid applying Smart Entry rather than guessing.
                policy_results.extend(
                    asyncio.run(
                        run_baseline(
                            log_path=log_path,
                            trade_ids=unknown_trade_ids,
                            hard_floor_bps=float(args.hard_floor_bps),
                            atr_mult=float(args.atr_mult),
                            max_sim_minutes=int(args.max_sim_minutes),
                            ignore_rr=bool(args.ignore_rr),
                            tie_break=str(args.tie_break),
                            cache_dir=cache_dir,
                            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                        )
                    )
                )

            smart_applied, filled, no_fill, market, tp, sl, time_exit, other, pnl_n, mean_pnl = summarize(policy_results)
            _ = (other, pnl_n)

            smart_fill_rate = None
            if smart_applied > 0:
                smart_fill_rate = filled / smart_applied

            print(
                " | ".join(
                    [
                        "policy",
                        "vol_thr",
                        "LONG(k,timeout)",
                        "SHORT(k,timeout)",
                        "scale_target",
                        "trades",
                        "baseline_mean_sim_pnl_pct",
                        "smart_applied",
                        "filled",
                        "no_fill",
                        "market",
                        "smart_fill_rate",
                        "tp",
                        "sl",
                        "time_exit",
                        "mean_sim_pnl_pct",
                    ]
                )
            )
            print("-" * 120)
            print(
                " | ".join(
                    [
                        str(args.policy),
                        str(vol_thr_val),
                        f"{long_k:.2f},{long_timeout}",
                        f"{short_k:.2f},{short_timeout}",
                        str(bool(args.scale_target_to_required_rr)),
                        str(len(policy_results)),
                        _fmt(b_mean_pnl),
                        str(smart_applied),
                        str(filled),
                        str(no_fill),
                        str(market),
                        ("" if smart_fill_rate is None else f"{smart_fill_rate:.3f}"),
                        str(tp),
                        str(sl),
                        str(time_exit),
                        _fmt(mean_pnl),
                    ]
                )
            )

            if args.details:
                no_fill_ids: list[str] = []
                worse_ids: list[str] = []
                tp_to_sl: list[str] = []

                for r in policy_results:
                    tid = r.trade.trade_id
                    if r.entry_mode == "smart_limit_no_fill":
                        no_fill_ids.append(tid)
                        continue

                    b = baseline_by_id.get(tid)
                    if b is None:
                        continue

                    b_reason = _reason(b).lower()
                    r_reason = _reason(r).lower()
                    if b_reason == "take_profit" and r_reason == "stop_loss":
                        tp_to_sl.append(tid)

                    bp = _pnl(b)
                    rp = _pnl(r)
                    if bp is not None and rp is not None:
                        if rp < (bp - _WORSE_EPS_PNL_PCT):
                            worse_ids.append(tid)

                if no_fill_ids or tp_to_sl or worse_ids:
                    print(f"  details: policy={args.policy} scale_target={bool(args.scale_target_to_required_rr)}")
                    if unknown_trade_ids:
                        print(f"    unknown_side_market: {' '.join(unknown_trade_ids)}")
                    if no_fill_ids:
                        print(f"    no_fill: {' '.join(no_fill_ids)}")
                    if tp_to_sl:
                        print(f"    tp_to_sl: {' '.join(tp_to_sl)}")
                    if worse_ids:
                        print(f"    worse_pnl: {' '.join(worse_ids)}")

        return 0

    print(
        " | ".join(
            [
                "k",
                "timeout",
                "vol_thr",
                "scale_target",
                "trades",
                "smart_applied",
                "filled",
                "no_fill",
                "market",
                "smart_fill_rate",
                "tp",
                "sl",
                "time_exit",
                "mean_sim_pnl_pct",
            ]
        )
    )
    print("-" * 120)

    for vol_thr in args.vol_threshold:
        vol_thr_val = None if vol_thr in (None, "None") else float(vol_thr)
        details_printed = 0
        for k in args.k:
            for timeout in args.timeout:
                results = asyncio.run(
                    run_one(
                        log_path=log_path,
                        trade_ids=trade_ids,
                        hard_floor_bps=float(args.hard_floor_bps),
                        atr_mult=float(args.atr_mult),
                        max_sim_minutes=int(args.max_sim_minutes),
                        ignore_rr=bool(args.ignore_rr),
                        tie_break=str(args.tie_break),
                        cache_dir=cache_dir,
                        smart_atr_mult=float(k),
                        smart_timeout_minutes=int(timeout),
                        smart_only_when_atr_bps_gte=vol_thr_val,
                        scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                    )
                )

                smart_applied, filled, no_fill, market, tp, sl, time_exit, other, pnl_n, mean_pnl = summarize(results)
                _ = (other, pnl_n)  # keep columns compact

                smart_fill_rate = None
                if smart_applied > 0:
                    smart_fill_rate = filled / smart_applied
                print(
                    " | ".join(
                        [
                            f"{float(k):.2f}",
                            str(int(timeout)),
                            ("" if vol_thr_val is None else str(vol_thr_val)),
                            str(bool(args.scale_target_to_required_rr)),
                            str(len(results)),
                            str(smart_applied),
                            str(filled),
                            str(no_fill),
                            str(market),
                            ("" if smart_fill_rate is None else f"{smart_fill_rate:.3f}"),
                            str(tp),
                            str(sl),
                            str(time_exit),
                            _fmt(mean_pnl),
                        ]
                    )
                )

                if args.details:
                    if details_printed >= int(args.details_max_combos):
                        continue

                    # Identify trade_ids that would be skipped (no fill)
                    no_fill_ids: list[str] = []
                    worse_ids: list[str] = []
                    tp_to_sl: list[str] = []

                    for r in results:
                        tid = r.trade.trade_id
                        if r.entry_mode == "smart_limit_no_fill":
                            no_fill_ids.append(tid)
                            continue

                        b = baseline_by_id.get(tid)
                        if b is None:
                            continue

                        b_reason = _reason(b).lower()
                        r_reason = _reason(r).lower()
                        if b_reason == "take_profit" and r_reason == "stop_loss":
                            tp_to_sl.append(tid)

                        bp = _pnl(b)
                        rp = _pnl(r)
                        if bp is not None and rp is not None:
                            # Only flag meaningfully worse deltas; float noise is common.
                            if rp < (bp - _WORSE_EPS_PNL_PCT):
                                worse_ids.append(tid)

                    if no_fill_ids or tp_to_sl or worse_ids:
                        details_printed += 1
                        print(
                            f"  details: vol_thr={vol_thr_val} k={float(k):.2f} timeout={int(timeout)} scale_target={bool(args.scale_target_to_required_rr)}"
                        )
                        if no_fill_ids:
                            print(f"    no_fill: {' '.join(no_fill_ids)}")
                        if tp_to_sl:
                            print(f"    tp_to_sl: {' '.join(tp_to_sl)}")
                        if worse_ids:
                            print(f"    worse_pnl: {' '.join(worse_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
