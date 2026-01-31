"""Evaluate a Smart Entry policy against a baseline on all trades in a log.

This is an analysis-only helper.

It:
- extracts all TRADE_CLOSED trade_ids from a log
- runs the min-stop simulation baseline (Smart Entry disabled)
- runs a side-aware Smart Entry policy (LONG and SHORT params differ)
- prints:
  - overall baseline vs policy summary
  - breakdown for vol_atr_bps < vol_threshold vs >= vol_threshold
  - per-side breakdown

Notes:
- Smart Entry is a LIMIT-fill simulation using 1m OHLCV.
- This does NOT change production trading logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import scripts.simulate_min_stop_effect as sim


@dataclass(frozen=True)
class Policy:
    vol_threshold_bps: float
    long_k: float
    long_timeout_min: int
    short_k: float
    short_timeout_min: int


@dataclass(frozen=True)
class Summary:
    trades: int
    smart_applied: int
    filled: int
    no_fill: int
    market: int
    low_vol_market: int
    smart_fill_rate: float | None
    tp: int
    sl: int
    time_exit: int
    other: int
    rr_reject: int
    mean_pnl_taken: float | None
    mean_pnl_all: float | None


@dataclass(frozen=True)
class FillThreshold:
    k_max_fill: float | None
    missed_by_price: float | None
    missed_by_bps: float | None
    timeout_price: float | None
    timeout_chase_bps: float | None


def _extract_trade_ids(log_path: Path) -> list[str]:
    trade_ids: list[str] = []
    seen: set[str] = set()

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if sim.TRADE_CLOSED_MARKER not in line:
                continue
            payload = sim._extract_json_after_marker(line, sim.TRADE_CLOSED_MARKER)
            if not payload or payload.get("event") != "TRADE_CLOSED":
                continue
            tid = payload.get("trade_id")
            if isinstance(tid, str) and tid and tid not in seen:
                seen.add(tid)
                trade_ids.append(tid)

    return trade_ids


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


def _side_of(result) -> str:
    return str(getattr(getattr(result, "trade", None), "side", "") or "").upper()


def _vol_bps_of(result) -> float | None:
    s = getattr(result, "signal", None)
    v = getattr(s, "vol_atr_bps", None)
    return None if v is None else float(v)


def _reason(result) -> str:
    return str(getattr(result, "simulated_exit_reason", "") or "")


def _pnl(result) -> float | None:
    v = getattr(result, "simulated_pnl_pct", None)
    return None if v is None else float(v)


def _pnl_net(pnl_pct: float | None, *, cost_bps: float) -> float | None:
    if pnl_pct is None:
        return None
    # pnl_pct is already in percent units (e.g. 0.1740 means +0.174%).
    # 1 bp = 0.01% => cost_bps / 100 = cost_pct
    return float(pnl_pct) - (float(cost_bps) / 100.0)


def _entry_price(result) -> float | None:
    t = getattr(result, "trade", None)
    v = getattr(t, "entry_price", None)
    return None if v is None else float(v)


def _entry_time(result):
    t = getattr(result, "trade", None)
    return getattr(t, "entry_time", None)


def _trade_symbol(result) -> str:
    t = getattr(result, "trade", None)
    return str(getattr(t, "symbol", "") or "")


def _would_pass_rr(result) -> bool | None:
    v = getattr(result, "would_pass_rr", None)
    return None if v is None else bool(v)


def _index_by_trade_id(results) -> dict[str, object]:
    out: dict[str, object] = {}
    for r in results:
        tid = getattr(getattr(r, "trade", None), "trade_id", None)
        if isinstance(tid, str):
            out[tid] = r
    return out


async def _compute_k_threshold(
    *,
    result: sim.SimulationResult,
    k_policy: float,
    timeout_minutes: int,
    cache_dir: Path,
) -> FillThreshold:
    """Compute k_max_fill and missed-by distance from OHLCV extrema.

    For a LONG entry:
      limit = entry - ATR * k
      fill if min_low <= limit
      k_max_fill = (entry - min_low) / ATR
      missed_by = min_low - limit (if > 0)

    For a SHORT entry:
      limit = entry + ATR * k
      fill if max_high >= limit
      k_max_fill = (max_high - entry) / ATR
      missed_by = limit - max_high (if > 0)
    """

    entry = _entry_price(result)
    entry_time = _entry_time(result)
    if entry is None or entry_time is None:
        return FillThreshold(k_max_fill=None, missed_by_price=None, missed_by_bps=None, timeout_price=None, timeout_chase_bps=None)

    signal = getattr(result, "signal", None)
    vol_bps = getattr(signal, "vol_atr_bps", None) if signal is not None else None
    if vol_bps is None:
        return FillThreshold(k_max_fill=None, missed_by_price=None, missed_by_bps=None, timeout_price=None, timeout_chase_bps=None)

    atr_price = float(entry) * (float(vol_bps) / 10000.0)
    if atr_price <= 0:
        return FillThreshold(k_max_fill=None, missed_by_price=None, missed_by_bps=None, timeout_price=None, timeout_chase_bps=None)

    start = entry_time
    end = entry_time + timedelta(minutes=max(int(timeout_minutes), 1))
    df_1m = await sim._fetch_ohlcv_1m(symbol=_trade_symbol(result), start=start, end=end, cache_dir=cache_dir)
    if df_1m is None or len(df_1m) == 0:
        return FillThreshold(k_max_fill=None, missed_by_price=None, missed_by_bps=None, timeout_price=None, timeout_chase_bps=None)

    timeout_ts = entry_time + timedelta(minutes=max(int(timeout_minutes), 1))
    timeout_price = sim._market_price_at_or_after(df_1m, timeout_ts)

    sig_entry = float(getattr(signal, "entry_price", entry))
    timeout_chase_bps = None
    if timeout_price is not None and sig_entry > 0:
        if _side_of(result) == "LONG":
            timeout_chase_bps = (float(timeout_price) / sig_entry - 1.0) * 10000.0
        elif _side_of(result) == "SHORT":
            timeout_chase_bps = (1.0 - float(timeout_price) / sig_entry) * 10000.0

    side = _side_of(result)
    if side == "LONG":
        min_low = float(df_1m["low"].min())
        k_max_fill = (entry - min_low) / atr_price
        limit = entry - (atr_price * float(k_policy))
        missed_by = max(min_low - limit, 0.0)
        missed_by_bps = (missed_by / entry) * 10000.0
        return FillThreshold(
            k_max_fill=float(k_max_fill),
            missed_by_price=float(missed_by),
            missed_by_bps=float(missed_by_bps),
            timeout_price=(None if timeout_price is None else float(timeout_price)),
            timeout_chase_bps=(None if timeout_chase_bps is None else float(timeout_chase_bps)),
        )

    if side == "SHORT":
        max_high = float(df_1m["high"].max())
        k_max_fill = (max_high - entry) / atr_price
        limit = entry + (atr_price * float(k_policy))
        missed_by = max(limit - max_high, 0.0)
        missed_by_bps = (missed_by / entry) * 10000.0
        return FillThreshold(
            k_max_fill=float(k_max_fill),
            missed_by_price=float(missed_by),
            missed_by_bps=float(missed_by_bps),
            timeout_price=(None if timeout_price is None else float(timeout_price)),
            timeout_chase_bps=(None if timeout_chase_bps is None else float(timeout_chase_bps)),
        )

    return FillThreshold(k_max_fill=None, missed_by_price=None, missed_by_bps=None, timeout_price=None, timeout_chase_bps=None)


def _summarize(results, *, vol_threshold_bps: float | None = None, side: str | None = None) -> Summary:
    selected = []
    for r in results:
        if side is not None and _side_of(r) != side:
            continue
        if vol_threshold_bps is not None:
            vb = _vol_bps_of(r)
            if vb is None:
                continue
        selected.append(r)

    smart_applied = 0
    filled = 0
    no_fill = 0
    market = 0
    tp = 0
    sl = 0
    time_exit = 0
    other = 0
    rr_reject = 0
    pnls_taken: list[float] = []
    pnls_all: list[float] = []
    low_vol_market = 0

    for r in selected:
        entry_mode = str(getattr(r, "entry_mode", "") or "")
        if entry_mode == "smart_limit":
            smart_applied += 1
            filled += 1
        elif entry_mode == "smart_limit_no_fill":
            smart_applied += 1
            no_fill += 1
        else:
            market += 1

        vb = _vol_bps_of(r)
        if vol_threshold_bps is not None and entry_mode == "market" and vb is not None and vb < vol_threshold_bps:
            low_vol_market += 1

        reason = _reason(r).lower()
        if reason == "take_profit":
            tp += 1
        elif reason == "stop_loss":
            sl += 1
        elif reason == "time_exit":
            time_exit += 1
        elif reason:
            other += 1

        p = _pnl(r)
        if p is not None:
            pnls_taken.append(p)
            pnls_all.append(p)
        else:
            # Treat "not taken" outcomes as 0% for an overall effectiveness view.
            # - smart_limit_no_fill => no position opened
            # - RR rejected (would_pass_rr==False) => trade filtered out by risk gate
            wp = _would_pass_rr(r)
            if entry_mode == "smart_limit_no_fill":
                pnls_all.append(0.0)
            elif wp is False:
                rr_reject += 1
                pnls_all.append(0.0)

    smart_fill_rate = None
    if smart_applied > 0:
        smart_fill_rate = filled / smart_applied

    return Summary(
        trades=len(selected),
        smart_applied=smart_applied,
        filled=filled,
        no_fill=no_fill,
        market=market,
        low_vol_market=low_vol_market,
        smart_fill_rate=smart_fill_rate,
        tp=tp,
        sl=sl,
        time_exit=time_exit,
        other=other,
        rr_reject=rr_reject,
        mean_pnl_taken=_mean(pnls_taken),
        mean_pnl_all=_mean(pnls_all),
    )


async def _run_baseline(*, log_path: Path, trade_ids: list[str], args) -> list[sim.SimulationResult]:
    return await sim.simulate(
        log_path=log_path,
        trade_ids=list(trade_ids),
        hard_floor_bps=float(args.hard_floor_bps),
        atr_mult=float(args.atr_mult),
        max_signal_lookback_s=90,
        max_price_delta=200.0,
        warmup_minutes=60,
        pre_pad_minutes=5,
        post_pad_minutes=5,
        max_sim_minutes=int(args.max_sim_minutes),
        fetch_ohlcv=not bool(args.no_ohlcv),
        tie_break=str(args.tie_break),
        cache_dir=Path(args.cache_dir),
        ignore_rr=bool(args.ignore_rr),
        scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
        smart_entry=False,
        smart_atr_mult=0.5,
        smart_timeout_minutes=15,
        smart_only_when_atr_bps_gte=None,
    )


async def _run_policy(*, log_path: Path, trade_ids: list[str], policy: Policy, args) -> list[sim.SimulationResult]:
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

    out: list[sim.SimulationResult] = []

    if long_trade_ids:
        out.extend(
            await sim.simulate(
                log_path=log_path,
                trade_ids=list(long_trade_ids),
                hard_floor_bps=float(args.hard_floor_bps),
                atr_mult=float(args.atr_mult),
                max_signal_lookback_s=90,
                max_price_delta=200.0,
                warmup_minutes=60,
                pre_pad_minutes=5,
                post_pad_minutes=5,
                max_sim_minutes=int(args.max_sim_minutes),
                fetch_ohlcv=not bool(args.no_ohlcv),
                tie_break=str(args.tie_break),
                cache_dir=Path(args.cache_dir),
                ignore_rr=bool(args.ignore_rr),
                scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                smart_entry=True,
                smart_atr_mult=float(policy.long_k),
                smart_timeout_minutes=int(policy.long_timeout_min),
                smart_only_when_atr_bps_gte=float(policy.vol_threshold_bps),
            )
        )

    if short_trade_ids:
        out.extend(
            await sim.simulate(
                log_path=log_path,
                trade_ids=list(short_trade_ids),
                hard_floor_bps=float(args.hard_floor_bps),
                atr_mult=float(args.atr_mult),
                max_signal_lookback_s=90,
                max_price_delta=200.0,
                warmup_minutes=60,
                pre_pad_minutes=5,
                post_pad_minutes=5,
                max_sim_minutes=int(args.max_sim_minutes),
                fetch_ohlcv=not bool(args.no_ohlcv),
                tie_break=str(args.tie_break),
                cache_dir=Path(args.cache_dir),
                ignore_rr=bool(args.ignore_rr),
                scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                smart_entry=True,
                smart_atr_mult=float(policy.short_k),
                smart_timeout_minutes=int(policy.short_timeout_min),
                smart_only_when_atr_bps_gte=float(policy.vol_threshold_bps),
            )
        )

    if unknown_trade_ids:
        # Side unknown: avoid guessing; keep baseline behavior.
        out.extend(
            await sim.simulate(
                log_path=log_path,
                trade_ids=list(unknown_trade_ids),
                hard_floor_bps=float(args.hard_floor_bps),
                atr_mult=float(args.atr_mult),
                max_signal_lookback_s=90,
                max_price_delta=200.0,
                warmup_minutes=60,
                pre_pad_minutes=5,
                post_pad_minutes=5,
                max_sim_minutes=int(args.max_sim_minutes),
                fetch_ohlcv=not bool(args.no_ohlcv),
                tie_break=str(args.tie_break),
                cache_dir=Path(args.cache_dir),
                ignore_rr=bool(args.ignore_rr),
                scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                smart_entry=False,
                smart_atr_mult=0.5,
                smart_timeout_minutes=15,
                smart_only_when_atr_bps_gte=None,
            )
        )

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a Smart Entry policy on all trades in a log")
    p.add_argument("--log", required=True, help="Path to the live trading log")

    p.add_argument("--hard-floor-bps", type=float, default=15.0)
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--max-sim-minutes", type=int, default=90)
    p.add_argument("--ignore-rr", action="store_true")
    p.add_argument("--scale-target-to-required-rr", action="store_true")
    p.add_argument("--tie-break", choices=["stop", "tp"], default="stop")
    p.add_argument("--cache-dir", default="data/cache/ohlcv")
    p.add_argument("--no-ohlcv", action="store_true", help="Skip OHLCV fetch; mostly for fast sanity checks")

    p.add_argument("--vol-threshold", type=float, default=5.0)
    p.add_argument("--long-k", type=float, default=0.90)
    p.add_argument("--long-timeout", type=int, default=4)
    p.add_argument("--short-k", type=float, default=0.85)
    p.add_argument("--short-timeout", type=int, default=3)

    p.add_argument(
        "--max-chase-bps",
        type=float,
        default=None,
        help="Max chase distance (bps) for timeout-market fallback. Example: 5 means 0.05%.",
    )
    p.add_argument(
        "--long-max-chase-bps",
        type=float,
        default=None,
        help="Max chase distance (bps) for LONG timeout-market fallback.",
    )
    p.add_argument(
        "--short-max-chase-bps",
        type=float,
        default=None,
        help="Max chase distance (bps) for SHORT timeout-market fallback.",
    )

    p.add_argument(
        "--detailed-report",
        action="store_true",
        help="Print no-fill and worse-trade diagnostics (baseline outcome + k_max_fill/missed-by).",
    )
    p.add_argument(
        "--include-baseline-outcome",
        action="store_true",
        help="Include baseline (market) simulated outcome columns in detailed report.",
    )
    p.add_argument(
        "--report-market-on-timeout",
        action="store_true",
        help=(
            "For NO-FILL trades, also report a hypothetical MARKET entry at timeout (if stop wasn't hit before timeout). "
            "This models 'Smart Entry with Market Fallback'."
        ),
    )

    p.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Approx total round-trip costs (fees+slippage) in bps to subtract from taken-trade PnL in detailed report.",
    )

    args = p.parse_args()

    log_path = Path(args.log)

    trade_ids = _extract_trade_ids(log_path)
    if not trade_ids:
        print("No TRADE_CLOSED trades found.")
        return 2

    policy = Policy(
        vol_threshold_bps=float(args.vol_threshold),
        long_k=float(args.long_k),
        long_timeout_min=int(args.long_timeout),
        short_k=float(args.short_k),
        short_timeout_min=int(args.short_timeout),
    )

    baseline_results = asyncio.run(_run_baseline(log_path=log_path, trade_ids=trade_ids, args=args))
    policy_results = asyncio.run(_run_policy(log_path=log_path, trade_ids=trade_ids, policy=policy, args=args))

    baseline_by_id = _index_by_trade_id(baseline_results)
    policy_by_id = _index_by_trade_id(policy_results)

    # Compare only intersecting trade_ids
    common_ids = [tid for tid in trade_ids if tid in baseline_by_id and tid in policy_by_id]

    print(
        " | ".join(
            [
                "set",
                "trades",
                "smart_applied",
                "filled",
                "no_fill",
                "market",
                "low_vol_market",
                "smart_fill_rate",
                "tp",
                "sl",
                "time_exit",
                "rr_reject",
                "mean_pnl_taken_pct",
                "mean_pnl_all_pct",
            ]
        )
    )
    print("-" * 120)

    b_all = _summarize([baseline_by_id[tid] for tid in common_ids])
    p_all = _summarize([policy_by_id[tid] for tid in common_ids], vol_threshold_bps=policy.vol_threshold_bps)

    def _print_row(name: str, s: Summary):
        print(
            " | ".join(
                [
                    name,
                    str(s.trades),
                    str(s.smart_applied),
                    str(s.filled),
                    str(s.no_fill),
                    str(s.market),
                    str(s.low_vol_market),
                    ("" if s.smart_fill_rate is None else f"{float(s.smart_fill_rate):.3f}"),
                    str(s.tp),
                    str(s.sl),
                    str(s.time_exit),
                    str(s.rr_reject),
                    _fmt(s.mean_pnl_taken),
                    _fmt(s.mean_pnl_all),
                ]
            )
        )

    _print_row("baseline", b_all)
    _print_row("policy", p_all)

    # Breakdown: low vol vs high vol (based on matched signal volatility)
    low_ids: list[str] = []
    high_ids: list[str] = []
    unknown_vol: list[str] = []

    for tid in common_ids:
        vb = _vol_bps_of(policy_by_id[tid])
        if vb is None:
            unknown_vol.append(tid)
        elif vb < policy.vol_threshold_bps:
            low_ids.append(tid)
        else:
            high_ids.append(tid)

    print("\nBreakdown by volatility (using matched SIGNAL_BREAKDOWN.vol_atr_bps)")
    print(
        " | ".join(
            [
                "group",
                "trades",
                "baseline_mean_taken",
                "policy_mean_taken",
                "baseline_mean_all",
                "policy_mean_all",
                "policy_smart_applied",
                "policy_no_fill",
                "policy_tp",
                "policy_sl",
            ]
        )
    )
    print("-" * 120)

    def _group_line(name: str, ids: list[str]):
        if not ids:
            return
        b = _summarize([baseline_by_id[tid] for tid in ids])
        p = _summarize([policy_by_id[tid] for tid in ids], vol_threshold_bps=policy.vol_threshold_bps)
        print(
            " | ".join(
                [
                    name,
                    str(len(ids)),
                    _fmt(b.mean_pnl_taken),
                    _fmt(p.mean_pnl_taken),
                    _fmt(b.mean_pnl_all),
                    _fmt(p.mean_pnl_all),
                    str(p.smart_applied),
                    str(p.no_fill),
                    str(p.tp),
                    str(p.sl),
                ]
            )
        )

    _group_line(f"vol<{policy.vol_threshold_bps}", low_ids)
    _group_line(f"vol>={policy.vol_threshold_bps}", high_ids)
    if unknown_vol:
        _group_line("vol=unknown", unknown_vol)

    # Breakdown by side
    print("\nBreakdown by side")
    print(
        "group | trades | baseline_mean_taken | policy_mean_taken | baseline_mean_all | policy_mean_all | policy_smart_applied | policy_no_fill | policy_tp | policy_sl"
    )
    print("-" * 120)
    for side in ("LONG", "SHORT"):
        ids = [tid for tid in common_ids if _side_of(policy_by_id[tid]) == side]
        if not ids:
            continue
        b = _summarize([baseline_by_id[tid] for tid in ids])
        p = _summarize([policy_by_id[tid] for tid in ids], vol_threshold_bps=policy.vol_threshold_bps)
        print(
            " | ".join(
                [
                    side,
                    str(len(ids)),
                    _fmt(b.mean_pnl_taken),
                    _fmt(p.mean_pnl_taken),
                    _fmt(b.mean_pnl_all),
                    _fmt(p.mean_pnl_all),
                    str(p.smart_applied),
                    str(p.no_fill),
                    str(p.tp),
                    str(p.sl),
                ]
            )
        )

    # Per-trade deltas list (only interesting changes)
    improved: list[str] = []
    worse: list[str] = []
    no_fill: list[str] = []

    for tid in common_ids:
        pr = policy_by_id[tid]
        if str(getattr(pr, "entry_mode", "") or "") == "smart_limit_no_fill":
            no_fill.append(tid)
            continue
        bp = _pnl(baseline_by_id[tid])
        pp = _pnl(pr)
        if bp is None or pp is None:
            continue
        if pp > bp:
            improved.append(tid)
        elif pp < bp:
            worse.append(tid)

    if improved or worse or no_fill:
        print("\nPer-trade comparison (policy vs baseline)")
        if no_fill:
            print("no_fill: " + " ".join(no_fill))
        if improved:
            print("improved: " + " ".join(improved))
        if worse:
            print("worse: " + " ".join(worse))

    if args.detailed_report:
        cache_dir = Path(args.cache_dir)
        cost_bps = float(args.cost_bps or 0.0)

        # Detailed NO-FILL diagnostics
        if no_fill:
            print("\nDetailed report: NO-FILL")
            headers = [
                "trade_id",
                "side",
                "vol_atr_bps",
                "k_policy",
                "timeout_min",
                "k_max_fill",
                "missed_by_price",
                "missed_by_bps",
                "timeout_price",
                "timeout_chase_bps",
            ]
            if args.include_baseline_outcome:
                headers.extend(["baseline_result", "baseline_pnl_pct"])
                if cost_bps > 0:
                    headers.extend(["baseline_pnl_net_pct"])
            if args.report_market_on_timeout:
                headers.extend(["timeout_market_entry_mode", "timeout_market_result", "timeout_market_pnl_pct"])
                if cost_bps > 0:
                    headers.extend(["timeout_market_pnl_net_pct"])
            headers.extend(["policy_result", "policy_entry_mode"])
            print(" | ".join(headers))
            print("-" * 140)

            # Compute thresholds in a single event loop to avoid repeated loop creation.
            async def _compute_all_thresholds():
                out: dict[str, FillThreshold] = {}
                for tid in no_fill:
                    pr = policy_by_id[tid]
                    side = _side_of(pr)
                    k_pol = policy.long_k if side == "LONG" else policy.short_k
                    timeout = policy.long_timeout_min if side == "LONG" else policy.short_timeout_min
                    out[tid] = await _compute_k_threshold(
                        result=pr,
                        k_policy=float(k_pol),
                        timeout_minutes=int(timeout),
                        cache_dir=cache_dir,
                    )
                return out

            thresholds = asyncio.run(_compute_all_thresholds())

            timeout_market_by_id: dict[str, sim.SimulationResult] = {}
            if args.report_market_on_timeout:
                async def _compute_timeout_market():
                    out: dict[str, sim.SimulationResult] = {}
                    for tid in no_fill:
                        pr = policy_by_id[tid]
                        side = _side_of(pr)
                        k_pol = policy.long_k if side == "LONG" else policy.short_k
                        timeout = policy.long_timeout_min if side == "LONG" else policy.short_timeout_min
                        rs = await sim.simulate(
                            log_path=log_path,
                            trade_ids=[tid],
                            hard_floor_bps=float(args.hard_floor_bps),
                            atr_mult=float(args.atr_mult),
                            max_signal_lookback_s=90,
                            max_price_delta=200.0,
                            warmup_minutes=60,
                            pre_pad_minutes=5,
                            post_pad_minutes=5,
                            max_sim_minutes=int(args.max_sim_minutes),
                            fetch_ohlcv=not bool(args.no_ohlcv),
                            tie_break=str(args.tie_break),
                            cache_dir=cache_dir,
                            ignore_rr=bool(args.ignore_rr),
                            scale_target_to_required_rr=bool(args.scale_target_to_required_rr),
                            smart_entry=True,
                            smart_atr_mult=float(k_pol),
                            smart_timeout_minutes=int(timeout),
                            smart_only_when_atr_bps_gte=float(policy.vol_threshold_bps),
                            smart_market_fallback=True,
                            smart_fallback_block_if_stop_hit=True,
                            smart_fallback_stop_reference="signal",
                            smart_fallback_max_chase_bps=(None if args.max_chase_bps is None else float(args.max_chase_bps)),
                            smart_fallback_max_chase_bps_long=(
                                None if args.long_max_chase_bps is None else float(args.long_max_chase_bps)
                            ),
                            smart_fallback_max_chase_bps_short=(
                                None if args.short_max_chase_bps is None else float(args.short_max_chase_bps)
                            ),
                        )
                        if rs:
                            out[tid] = rs[0]
                    return out

                timeout_market_by_id = asyncio.run(_compute_timeout_market())

            for tid in no_fill:
                pr = policy_by_id[tid]
                br = baseline_by_id.get(tid)
                side = _side_of(pr)
                vb = _vol_bps_of(pr)
                k_pol = policy.long_k if side == "LONG" else policy.short_k
                timeout = policy.long_timeout_min if side == "LONG" else policy.short_timeout_min
                thr = thresholds.get(tid) or FillThreshold(None, None, None, None, None)

                row = [
                    tid,
                    side,
                    ("" if vb is None else f"{float(vb):.4f}"),
                    f"{float(k_pol):.2f}",
                    str(int(timeout)),
                    ("" if thr.k_max_fill is None else f"{float(thr.k_max_fill):.4f}"),
                    ("" if thr.missed_by_price is None else f"{float(thr.missed_by_price):.4f}"),
                    ("" if thr.missed_by_bps is None else f"{float(thr.missed_by_bps):.2f}"),
                    ("" if thr.timeout_price is None else f"{float(thr.timeout_price):.4f}"),
                    ("" if thr.timeout_chase_bps is None else f"{float(thr.timeout_chase_bps):.2f}"),
                ]
                if args.include_baseline_outcome:
                    row.append(_reason(br).lower() if br is not None else "")
                    bp = (_pnl(br) if br is not None else None)
                    row.append(_fmt(bp) if bp is not None else "")
                    if cost_bps > 0:
                        row.append(_fmt(_pnl_net(bp, cost_bps=cost_bps)) if bp is not None else "")
                if args.report_market_on_timeout:
                    fr = timeout_market_by_id.get(tid)
                    if fr is None:
                        row.extend(["", "", ""])
                        if cost_bps > 0:
                            row.append("")
                    else:
                        row.append(str(getattr(fr, "entry_mode", "") or ""))
                        row.append(_reason(fr).lower())
                        fp = _pnl(fr)
                        row.append(_fmt(fp))
                        if cost_bps > 0:
                            row.append(_fmt(_pnl_net(fp, cost_bps=cost_bps)) if fp is not None else "")
                row.append(_reason(pr).lower())
                row.append(str(getattr(pr, "entry_mode", "") or ""))
                print(" | ".join(row))

        # Detailed WORSE diagnostics
        if worse:
            print("\nDetailed report: WORSE")
            headers = [
                "trade_id",
                "side",
                "vol_atr_bps",
                "baseline_result",
                "baseline_pnl_pct",
                "policy_result",
                "policy_entry_mode",
                "policy_pnl_pct",
                "delta_pct(policy-baseline)",
            ]
            if float(args.cost_bps or 0.0) > 0:
                headers.extend(["baseline_pnl_net_pct", "policy_pnl_net_pct", "delta_net_pct(policy-baseline)"])
            print(" | ".join(headers))
            print("-" * 140)
            for tid in worse:
                pr = policy_by_id[tid]
                br = baseline_by_id.get(tid)
                side = _side_of(pr)
                vb = _vol_bps_of(pr)
                bp = _pnl(br) if br is not None else None
                pp = _pnl(pr)
                delta = None
                if bp is not None and pp is not None:
                    delta = pp - bp

                bp_net = _pnl_net(bp, cost_bps=float(args.cost_bps or 0.0)) if bp is not None else None
                pp_net = _pnl_net(pp, cost_bps=float(args.cost_bps or 0.0)) if pp is not None else None
                delta_net = None
                if bp_net is not None and pp_net is not None:
                    delta_net = pp_net - bp_net

                extra = []
                if float(args.cost_bps or 0.0) > 0:
                    extra = [
                        ("" if bp_net is None else f"{float(bp_net):.4f}"),
                        ("" if pp_net is None else f"{float(pp_net):.4f}"),
                        ("" if delta_net is None else f"{float(delta_net):.4f}"),
                    ]
                print(
                    " | ".join(
                        [
                            tid,
                            side,
                            ("" if vb is None else f"{float(vb):.4f}"),
                            (_reason(br).lower() if br is not None else ""),
                            (_fmt(bp) if bp is not None else ""),
                            _reason(pr).lower(),
                            str(getattr(pr, "entry_mode", "") or ""),
                            (_fmt(pp) if pp is not None else ""),
                            ("" if delta is None else f"{float(delta):.4f}"),
                        ]
                        + extra
                    )
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
