#!/usr/bin/env python3
"""Analyze MR PROMOTE min_z_score sweep from runtime telemetry logs.

This tool joins:
- strategy_recheck_request (source for ADX / touch / fast_watch dist)
- mr_recheck_eval         (source for z / near / dist fallback / action)
- TRADE_CLOSED            (optional realized outcome if promotion metadata exists)

Then it computes how many recheck cases would pass PROMOTE's z gate for a
threshold sweep (for example 1.8 / 2.0 / 2.2) while keeping fixed gates:
- abs(dist_bps) <= max_dist_bps
- adx <= max_adx
- touch policy (configurable)

Notes:
- This is an observability/tuning helper. It does not modify execution.
- Trend/volume/shock gates are not reconstructed from historical logs here.
- Trade-labeled sweep requires `TRADE_CLOSED.entry_metadata.promotion_override`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUEST_MARKER = "strategy_recheck_request "
EVAL_MARKER = "mr_recheck_eval "
TRADE_CLOSED_MARKER = "TRADE_CLOSED "


@dataclass(frozen=True)
class RecheckCase:
    file: str
    run_id: Optional[str]
    pending_id: str
    symbol: Optional[str]
    side: Optional[str]
    near: Optional[str]
    touch_confirmed: Optional[bool]
    dist_bps: Optional[float]
    adx: Optional[float]
    z: Optional[float]
    action: Optional[str]
    primary_gate_reason: Optional[str]


@dataclass(frozen=True)
class TradeClosedCase:
    file: str
    run_id: Optional[str]
    symbol: Optional[str]
    side: Optional[str]
    strategy: Optional[str]
    signal_id: Optional[str]
    pnl_usd: Optional[float]
    rr_achieved: Optional[float]
    exit_reason: Optional[str]
    has_promotion_meta: bool
    promote_candidate: Optional[bool]
    promote_applied: Optional[bool]
    touch_confirmed: Optional[bool]
    near: Optional[str]
    dist_bps: Optional[float]
    adx: Optional[float]
    z: Optional[float]


def _parse_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _parse_json_after_marker(line: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = line.find(marker)
    if idx < 0:
        return None
    payload_raw = line[idx + len(marker) :].strip()
    try:
        payload = json.loads(payload_raw)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _iter_log_files(logs: Iterable[str], log_glob: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for raw in logs:
        p = Path(raw)
        if p.exists() and p.is_file():
            out.append(p)
    if log_glob:
        out.extend(sorted(Path().glob(log_glob)))

    unique: List[Path] = []
    seen = set()
    for p in out:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _touch_gate_ok(touch_confirmed: Optional[bool], policy: str) -> bool:
    if policy == "required":
        return touch_confirmed is True
    if policy == "missing_as_false":
        return bool(touch_confirmed)
    if policy == "missing_as_true":
        return touch_confirmed is not False
    return True  # ignore


def _load_cases(paths: List[Path]) -> List[RecheckCase]:
    requests: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cases: List[RecheckCase] = []

    for path in paths:
        path_key = str(path)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if REQUEST_MARKER in line:
                    payload = _parse_json_after_marker(line, REQUEST_MARKER)
                    if not payload or payload.get("event") != "strategy_recheck_request":
                        continue
                    pending_id = payload.get("pending_id")
                    if not pending_id:
                        continue
                    requests[(path_key, str(pending_id))] = payload
                    continue

                if EVAL_MARKER in line:
                    payload = _parse_json_after_marker(line, EVAL_MARKER)
                    if not payload or payload.get("event") != "mr_recheck_eval":
                        continue
                    pending_id = payload.get("pending_id")
                    if not pending_id:
                        continue
                    req = requests.get((path_key, str(pending_id)), {})

                    check_detail = req.get("check_detail") if isinstance(req.get("check_detail"), dict) else {}
                    fast_watch = check_detail.get("fast_watch") if isinstance(check_detail.get("fast_watch"), dict) else {}
                    condition_data = req.get("condition_data") if isinstance(req.get("condition_data"), dict) else {}

                    near_raw = payload.get("near")
                    if near_raw is None:
                        near_raw = condition_data.get("near")
                    near_norm = str(near_raw).strip().lower() if near_raw is not None else None
                    if near_norm not in ("lower", "upper"):
                        near_norm = None

                    dist_bps = _parse_float(fast_watch.get("dist_to_band_bps"))
                    if dist_bps is None:
                        dist_bps = _parse_float(payload.get("dist_to_trigger_bps"))

                    touch_confirmed: Optional[bool]
                    if isinstance(fast_watch.get("touch_confirmed"), bool):
                        touch_confirmed = bool(fast_watch.get("touch_confirmed"))
                    else:
                        touch_confirmed = None

                    cases.append(
                        RecheckCase(
                            file=path_key,
                            run_id=str(payload.get("run_id")) if payload.get("run_id") is not None else None,
                            pending_id=str(pending_id),
                            symbol=str(payload.get("symbol")) if payload.get("symbol") is not None else None,
                            side=str(payload.get("side")) if payload.get("side") is not None else None,
                            near=near_norm,
                            touch_confirmed=touch_confirmed,
                            dist_bps=dist_bps,
                            adx=_parse_float(condition_data.get("adx")),
                            z=_parse_float(payload.get("z")),
                            action=str(payload.get("action")) if payload.get("action") is not None else None,
                            primary_gate_reason=(
                                str(payload.get("primary_gate_reason"))
                                if payload.get("primary_gate_reason") is not None
                                else None
                            ),
                        )
                    )
    return cases


def _load_trade_closed_cases(paths: List[Path], strategy_name: str) -> List[TradeClosedCase]:
    out: List[TradeClosedCase] = []
    strategy_norm = str(strategy_name or "").strip().lower()

    for path in paths:
        path_key = str(path)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                payload = _parse_json_after_marker(line, TRADE_CLOSED_MARKER)
                if not payload or str(payload.get("event", "")).upper() != "TRADE_CLOSED":
                    continue

                payload_strategy = payload.get("strategy_name") or payload.get("strategy")
                payload_strategy_norm = str(payload_strategy or "").strip().lower()
                if strategy_norm and payload_strategy_norm != strategy_norm:
                    continue

                entry_meta = payload.get("entry_metadata")
                if not isinstance(entry_meta, dict):
                    entry_meta = {}
                promote = entry_meta.get("promotion_override")
                if not isinstance(promote, dict):
                    promote = None

                raw_near = promote.get("near") if isinstance(promote, dict) else None
                near_norm = str(raw_near).strip().lower() if raw_near is not None else None
                if near_norm not in ("lower", "upper"):
                    near_norm = None

                touch_confirmed: Optional[bool]
                if isinstance(promote, dict) and promote.get("touch_confirmed") is not None:
                    touch_confirmed = bool(promote.get("touch_confirmed"))
                else:
                    touch_confirmed = None

                signal_id = entry_meta.get("signal_id")
                if signal_id is None:
                    signal_id = payload.get("signal_id")
                signal_id = str(signal_id).strip() if signal_id is not None else None
                if signal_id == "":
                    signal_id = None

                out.append(
                    TradeClosedCase(
                        file=path_key,
                        run_id=str(payload.get("run_id")) if payload.get("run_id") is not None else None,
                        symbol=str(payload.get("symbol")) if payload.get("symbol") is not None else None,
                        side=str(payload.get("side")) if payload.get("side") is not None else None,
                        strategy=str(payload_strategy) if payload_strategy is not None else None,
                        signal_id=signal_id,
                        pnl_usd=_parse_float(payload.get("pnl_usd")),
                        rr_achieved=_parse_float(payload.get("rr_achieved")),
                        exit_reason=str(payload.get("exit_reason")) if payload.get("exit_reason") is not None else None,
                        has_promotion_meta=isinstance(promote, dict),
                        promote_candidate=(
                            bool(promote.get("candidate")) if isinstance(promote, dict) and promote.get("candidate") is not None else None
                        ),
                        promote_applied=(
                            bool(promote.get("applied")) if isinstance(promote, dict) and promote.get("applied") is not None else None
                        ),
                        touch_confirmed=touch_confirmed,
                        near=near_norm,
                        dist_bps=_parse_float(promote.get("dist_bps")) if isinstance(promote, dict) else None,
                        adx=_parse_float(promote.get("adx")) if isinstance(promote, dict) else None,
                        z=_parse_float(promote.get("z")) if isinstance(promote, dict) else None,
                    )
                )
    return out


def _parse_thresholds(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except Exception:
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        out.append(value)
    return sorted(set(out))


def _fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.0%"
    return f"{(100.0 * num / den):.1f}%"


def _recommendation(
    *,
    base_count: int,
    by_threshold: Dict[str, Dict[str, Any]],
    default_threshold: float,
    min_sample_size: int,
) -> str:
    if base_count < min_sample_size:
        return (
            "insufficient_sample: keep current threshold "
            f"({default_threshold:.2f}) and collect more observe data"
        )

    default_key = f"{default_threshold:.2f}"
    default_row = by_threshold.get(default_key)
    if not default_row:
        return f"no_default_in_sweep: keep current threshold ({default_threshold:.2f})"

    # Without PnL labels, choose by opportunity compression only.
    # Prefer the most conservative threshold that still keeps >=50% of default pass count.
    default_pass = int(default_row["pass_count"])
    if default_pass <= 0:
        return f"default_pass_zero: keep current threshold ({default_threshold:.2f})"

    sorted_rows = sorted(
        ((float(k), v) for k, v in by_threshold.items()),
        key=lambda kv: kv[0],
    )
    candidate = default_threshold
    for thr, row in sorted_rows:
        pass_count = int(row["pass_count"])
        if thr >= default_threshold and pass_count >= int(math.ceil(default_pass * 0.5)):
            candidate = thr
    if abs(candidate - default_threshold) < 1e-9:
        return f"keep {default_threshold:.2f} (no safer threshold with acceptable opportunity retention)"
    return f"try {candidate:.2f} (conservative move with >=50% opportunity retention vs {default_threshold:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep MR PROMOTE min_z_score from runtime logs.")
    ap.add_argument("--log", action="append", default=[], help="Explicit log path (repeatable).")
    ap.add_argument(
        "--glob",
        default="logs/live_trading_*.log",
        help="Glob for logs (default: logs/live_trading_*.log).",
    )
    ap.add_argument(
        "--thresholds",
        default="1.8,2.0,2.2",
        help="Comma separated min_z_score sweep values.",
    )
    ap.add_argument("--default-threshold", type=float, default=2.0)
    ap.add_argument("--max-dist-bps", type=float, default=2.0)
    ap.add_argument("--max-adx", type=float, default=20.0)
    ap.add_argument("--strategy-name", default="mean_reversion")
    ap.add_argument(
        "--touch-policy",
        choices=("required", "missing_as_true", "missing_as_false", "ignore"),
        default="missing_as_true",
        help="How to treat missing touch_confirmed in historical logs.",
    )
    ap.add_argument("--min-sample-size", type=int, default=30)
    ap.add_argument("--out-json", default=None, help="Optional output JSON path.")
    ap.add_argument("--out-md", default=None, help="Optional output markdown path.")
    args = ap.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    if not thresholds:
        raise SystemExit("No valid thresholds parsed.")

    paths = _iter_log_files(args.log, args.glob)
    if not paths:
        raise SystemExit("No log files found.")

    cases = _load_cases(paths)
    if not cases:
        raise SystemExit("No mr_recheck_eval cases found.")

    missing_touch = sum(1 for c in cases if c.touch_confirmed is None)
    missing_dist = sum(1 for c in cases if c.dist_bps is None)
    missing_adx = sum(1 for c in cases if c.adx is None)
    missing_z = sum(1 for c in cases if c.z is None)
    missing_near = sum(1 for c in cases if c.near is None)

    base_pool: List[RecheckCase] = []
    rejected_reason = Counter()
    for c in cases:
        if not _touch_gate_ok(c.touch_confirmed, args.touch_policy):
            rejected_reason["touch_policy"] += 1
            continue
        if c.near not in ("lower", "upper"):
            rejected_reason["near_missing"] += 1
            continue
        if c.dist_bps is None:
            rejected_reason["dist_missing"] += 1
            continue
        if abs(float(c.dist_bps)) > float(args.max_dist_bps):
            rejected_reason["dist_gate"] += 1
            continue
        if c.adx is None:
            rejected_reason["adx_missing"] += 1
            continue
        if float(c.adx) > float(args.max_adx):
            rejected_reason["adx_gate"] += 1
            continue
        if c.z is None:
            rejected_reason["z_missing"] += 1
            continue
        base_pool.append(c)

    by_threshold: Dict[str, Dict[str, Any]] = {}
    for thr in thresholds:
        selected = [c for c in base_pool if abs(float(c.z)) >= float(thr)]
        by_side = Counter((c.side or "unknown") for c in selected)
        by_action = Counter((c.action or "unknown") for c in selected)
        by_threshold[f"{thr:.2f}"] = {
            "threshold": thr,
            "pass_count": len(selected),
            "pass_rate_vs_base": (len(selected) / len(base_pool)) if base_pool else 0.0,
            "pass_rate_vs_all_eval": (len(selected) / len(cases)) if cases else 0.0,
            "by_side": dict(by_side),
            "by_action": dict(by_action),
        }

    trade_cases = _load_trade_closed_cases(paths, strategy_name=str(args.strategy_name))
    trade_missing_promote_meta = sum(1 for t in trade_cases if not t.has_promotion_meta)
    trade_pool: List[TradeClosedCase] = []
    trade_rejected_reason = Counter()
    for t in trade_cases:
        if not t.has_promotion_meta:
            trade_rejected_reason["promotion_meta_missing"] += 1
            continue
        if not (t.promote_candidate is True or t.promote_applied is True):
            trade_rejected_reason["not_promote_candidate"] += 1
            continue
        if not _touch_gate_ok(t.touch_confirmed, args.touch_policy):
            trade_rejected_reason["touch_policy"] += 1
            continue
        if t.near not in ("lower", "upper"):
            trade_rejected_reason["near_missing"] += 1
            continue
        if t.dist_bps is None:
            trade_rejected_reason["dist_missing"] += 1
            continue
        if abs(float(t.dist_bps)) > float(args.max_dist_bps):
            trade_rejected_reason["dist_gate"] += 1
            continue
        if t.adx is None:
            trade_rejected_reason["adx_missing"] += 1
            continue
        if float(t.adx) > float(args.max_adx):
            trade_rejected_reason["adx_gate"] += 1
            continue
        if t.z is None:
            trade_rejected_reason["z_missing"] += 1
            continue
        trade_pool.append(t)

    trade_sweep: Dict[str, Dict[str, Any]] = {}
    for thr in thresholds:
        selected_trades = [t for t in trade_pool if abs(float(t.z)) >= float(thr)]
        pnl_vals = [float(t.pnl_usd) for t in selected_trades if t.pnl_usd is not None]
        rr_vals = [float(t.rr_achieved) for t in selected_trades if t.rr_achieved is not None]
        wins = sum(1 for v in pnl_vals if v > 0)
        losses = sum(1 for v in pnl_vals if v < 0)
        exit_counts = Counter((t.exit_reason or "unknown") for t in selected_trades)
        trade_sweep[f"{thr:.2f}"] = {
            "threshold": thr,
            "pass_count": len(selected_trades),
            "pass_rate_vs_trade_base": (len(selected_trades) / len(trade_pool)) if trade_pool else 0.0,
            "labeled_trade_count": len(pnl_vals),
            "win_count": wins,
            "loss_count": losses,
            "win_rate": (wins / len(pnl_vals)) if pnl_vals else None,
            "pnl_sum_usd": sum(pnl_vals) if pnl_vals else None,
            "avg_pnl_usd": (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else None,
            "avg_rr_achieved": (sum(rr_vals) / len(rr_vals)) if rr_vals else None,
            "exit_reasons": dict(exit_counts),
        }

    recommendation = _recommendation(
        base_count=len(base_pool),
        by_threshold=by_threshold,
        default_threshold=float(args.default_threshold),
        min_sample_size=int(args.min_sample_size),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "file_count": len(paths),
            "files": [str(p) for p in paths],
            "thresholds": thresholds,
            "default_threshold": float(args.default_threshold),
            "max_dist_bps": float(args.max_dist_bps),
            "max_adx": float(args.max_adx),
            "strategy_name": str(args.strategy_name),
            "touch_policy": str(args.touch_policy),
        },
        "coverage": {
            "total_eval_cases": len(cases),
            "missing_touch_confirmed": missing_touch,
            "missing_dist_bps": missing_dist,
            "missing_adx": missing_adx,
            "missing_z": missing_z,
            "missing_near": missing_near,
        },
        "base_gate": {
            "eligible_count": len(base_pool),
            "eligible_rate_vs_all_eval": (len(base_pool) / len(cases)) if cases else 0.0,
            "rejected_breakdown": dict(rejected_reason),
        },
        "trade_closed_coverage": {
            "total_trade_closed_in_scope": len(trade_cases),
            "missing_promotion_meta": trade_missing_promote_meta,
            "trade_base_eligible_count": len(trade_pool),
            "trade_base_eligible_rate_vs_scope": (len(trade_pool) / len(trade_cases)) if trade_cases else 0.0,
            "trade_rejected_breakdown": dict(trade_rejected_reason),
        },
        "sweep": by_threshold,
        "trade_sweep": trade_sweep,
        "recommendation": recommendation,
        "telemetry_gaps": [
            "trend_veto / ema_stack data not reconstructed in this sweep",
            "volume_strength gate not reconstructed in this sweep",
            "shock_state gate not reconstructed in this sweep",
            "no direct PnL label in mr_recheck_eval; result is opportunity analysis, not win-rate optimization",
        ],
    }
    if not trade_cases:
        summary["telemetry_gaps"].append("no TRADE_CLOSED events for selected strategy in selected logs")
    elif trade_missing_promote_meta > 0:
        summary["telemetry_gaps"].append("TRADE_CLOSED.entry_metadata.promotion_override missing on part of historical trades")

    # Console summary
    print(f"Files: {len(paths)}")
    print(f"Eval cases: {len(cases)}")
    print(
        "Coverage gaps: "
        f"touch_missing={missing_touch} dist_missing={missing_dist} adx_missing={missing_adx} "
        f"z_missing={missing_z} near_missing={missing_near}"
    )
    print(
        "Base gate pool: "
        f"{len(base_pool)}/{len(cases)} ({_fmt_pct(len(base_pool), len(cases))}) "
        f"[|dist|<={args.max_dist_bps}, adx<={args.max_adx}, touch_policy={args.touch_policy}]"
    )
    for key in sorted(by_threshold.keys(), key=lambda k: float(k)):
        row = by_threshold[key]
        pass_count = int(row["pass_count"])
        print(
            f"  z>={key}: pass={pass_count} "
            f"vs_base={_fmt_pct(pass_count, len(base_pool))} "
            f"vs_all={_fmt_pct(pass_count, len(cases))}"
        )
    print(
        "Trade coverage: "
        f"scope={len(trade_cases)} with_promote_meta={len(trade_cases) - trade_missing_promote_meta} "
        f"trade_base={len(trade_pool)}"
    )
    if trade_pool:
        for key in sorted(trade_sweep.keys(), key=lambda k: float(k)):
            row = trade_sweep[key]
            pass_count = int(row["pass_count"])
            pnl_sum = row.get("pnl_sum_usd")
            win_rate = row.get("win_rate")
            pnl_str = f"{float(pnl_sum):.4f}" if pnl_sum is not None else "na"
            win_str = f"{(float(win_rate) * 100.0):.1f}%" if win_rate is not None else "na"
            print(
                f"  trade z>={key}: pass={pass_count} "
                f"vs_trade_base={_fmt_pct(pass_count, len(trade_pool))} "
                f"win_rate={win_str} pnl_sum_usd={pnl_str}"
            )
    print(f"Recommendation: {recommendation}")

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json_path}")

    if args.out_md:
        out_md_path = Path(args.out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("# MR PROMOTE min_z_score Tuning Report")
        lines.append("")
        lines.append(f"- Generated at (UTC): `{summary['generated_at_utc']}`")
        lines.append(f"- Files: `{len(paths)}`")
        lines.append(f"- Eval cases: `{len(cases)}`")
        lines.append(
            f"- Base gate pool: `{len(base_pool)}/{len(cases)}` "
            f"(`{_fmt_pct(len(base_pool), len(cases))}`)"
        )
        lines.append("")
        lines.append("## Sweep")
        lines.append("")
        lines.append("| min_z_score | pass_count | pass_rate_vs_base | pass_rate_vs_all_eval |")
        lines.append("| --- | --- | --- | --- |")
        for key in sorted(by_threshold.keys(), key=lambda k: float(k)):
            row = by_threshold[key]
            lines.append(
                f"| {key} | {row['pass_count']} | "
                f"{(row['pass_rate_vs_base'] * 100.0):.1f}% | {(row['pass_rate_vs_all_eval'] * 100.0):.1f}% |"
            )
        lines.append("")
        lines.append("## Trade-Labeled Sweep (TRADE_CLOSED)")
        lines.append("")
        lines.append(
            f"- Trade coverage: scope=`{len(trade_cases)}`, "
            f"with_promotion_meta=`{len(trade_cases) - trade_missing_promote_meta}`, "
            f"base=`{len(trade_pool)}`"
        )
        if trade_pool:
            lines.append("")
            lines.append(
                "| min_z_score | pass_count | pass_rate_vs_trade_base | "
                "labeled_trade_count | win_rate | pnl_sum_usd | avg_pnl_usd | avg_rr_achieved |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for key in sorted(trade_sweep.keys(), key=lambda k: float(k)):
                row = trade_sweep[key]
                win_rate = row.get("win_rate")
                pnl_sum = row.get("pnl_sum_usd")
                avg_pnl = row.get("avg_pnl_usd")
                avg_rr = row.get("avg_rr_achieved")
                win_txt = f"{(float(win_rate) * 100.0):.1f}%" if win_rate is not None else "na"
                pnl_sum_txt = f"{float(pnl_sum):.4f}" if pnl_sum is not None else "na"
                avg_pnl_txt = f"{float(avg_pnl):.4f}" if avg_pnl is not None else "na"
                avg_rr_txt = f"{float(avg_rr):.3f}" if avg_rr is not None else "na"
                lines.append(
                    f"| {key} | {row['pass_count']} | {(row['pass_rate_vs_trade_base'] * 100.0):.1f}% | "
                    f"{row['labeled_trade_count']} | {win_txt} | {pnl_sum_txt} | {avg_pnl_txt} | {avg_rr_txt} |"
                )
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"- {recommendation}")
        lines.append("")
        lines.append("## Telemetry Gaps")
        lines.append("")
        for gap in summary["telemetry_gaps"]:
            lines.append(f"- {gap}")
        out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote MD: {out_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
