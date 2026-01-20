"""Offline MR signal-universe analyzer.

Goal: Build a single dataset of Mean Reversion candidate/emit events (ingress, waiting room,
volume checks, soft deferral/recheck) from a single run log and produce:
- reports/mr_signal_universe_<run_id>.csv
- reports/mr_signal_universe_<run_id>.md

This script is read-only: it only parses logs and writes reports.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


MR_STRATEGY = "mean_reversion"


@dataclasses.dataclass
class MRControllerSnapshot:
    ts_ms: int
    run_id: Optional[str]
    symbol: Optional[str]
    timeframe: Optional[str]
    px: Optional[float]
    vwap: Optional[float]
    vwap_std: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    z: Optional[float]
    adx: Optional[float]
    atr: Optional[float]
    atr_pct: Optional[float]


@dataclasses.dataclass
class VolumeSnapshot:
    ts_ms: int
    run_id: Optional[str]
    symbol: Optional[str]
    timeframe: Optional[str]
    volume_bucket: Optional[str]
    volume_strength: Optional[float]
    central_bucket_decision: Optional[str]


@dataclasses.dataclass
class IngressLine:
    ts_ms: int
    run_id: Optional[str]
    symbol: str
    side: str
    timeframe: Optional[str]
    intent_hint: Optional[str]
    entry_price: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    adx: Optional[float]


@dataclasses.dataclass
class SignalRecord:
    run_id: str
    signal_id: str
    pending_id: Optional[str]
    parent_pending_id: Optional[str]
    dedupe_key: Optional[str]
    symbol: Optional[str]
    side: Optional[str]
    timeframe: Optional[str]
    ts_ms: Optional[int]

    final_outcome: str
    final_reason_code: Optional[str]
    drop_reason: Optional[str]

    volume_bucket: Optional[str]
    volume_strength: Optional[float]

    adx: Optional[float]
    vwap_std: Optional[float]
    band_width_bps: Optional[float]
    dist_outside_bps: Optional[float]
    z: Optional[float]

    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]

    stop_pct_bps_observed: Optional[float]
    stop_pct_bps_model: Optional[float]
    stop_pct_bps: Optional[float]
    target_bps: Optional[float]
    std_bps: Optional[float]
    atr_bps: Optional[float]
    k_implied: Optional[float]

    gate_threshold_bps: Optional[float]
    gate_margin_bps: Optional[float]

    cost_bps_assumed: float
    rr_gross: Optional[float]
    rr_net: Optional[float]
    rr_net_clamped0: Optional[float]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def parse_iso_ts_ms(value: str) -> Optional[int]:
    # Handles: 2026-01-18T20:55:44.469974+00:00
    try:
        # Python 3.11 supports fromisoformat with timezone
        ts = dt.datetime.fromisoformat(value)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return int(ts.timestamp() * 1000)
    except Exception:
        return None


LOG_PREFIX_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+-\s+\[(?P<logger>[^\]]+)\]\s+-\s+(?P<level>\w+)\s+-\s+(?P<body>.*)$"
)


def parse_prefix_ts_ms(line: str, tz: dt.tzinfo = dt.timezone.utc) -> Optional[int]:
    m = LOG_PREFIX_RE.match(line)
    if not m:
        return None
    try:
        d = m.group("date")
        t = m.group("time")
        ts = dt.datetime.fromisoformat(f"{d}T{t}").replace(tzinfo=tz)
        return int(ts.timestamp() * 1000)
    except Exception:
        return None


def extract_json_from_line(line: str) -> Optional[Dict[str, Any]]:
    # We expect logs like: ... - INFO - event_name {"event": "...", ...}
    # Find first '{' and attempt json.loads from there.
    idx = line.find("{")
    if idx < 0:
        return None
    payload = line[idx:]
    try:
        return json.loads(payload)
    except Exception:
        return None


INGRESS_RE = re.compile(
    r"\[MEAN_REVERSION/(?P<symbol>[^\]]+)\]\s+Signal ingress\s+\|\s+side=(?P<side>long|short)\s+\|\s+intent_hint=(?P<intent>\w+)\s+\|\s+reason=.*?\(px\s+(?P<px>[0-9.]+)\s+[^\s]+\s+(?P<band>[0-9.]+),\s+ADX\s+(?P<adx>[0-9.]+)\s+<\s+(?P<adx_th>[0-9.]+)\)"
)


SC_SIGNAL_RECEIVED_RE = re.compile(
    r"\[MEAN_REVERSION/(?P<symbol>[^\]]+)\]\s+Signal Received\.\s+Side:\s+(?P<side>sell|buy),\s+Reason:\s+'(?P<reason>.*)'\s*$"
)


SC_DEFERRING_RE = re.compile(
    r"\[MEAN_REVERSION/(?P<symbol>[^\]]+)\]\s+Deferring signal\s+\((?P<reason>[^\)]+)\)\s*$"
)


SC_RISK_REJECTED_RE = re.compile(
    r"\[MEAN_REVERSION/(?P<symbol>[^\]]+)\]\s+REJECTED\s+\(Risk Check\):\s*(?P<reason>[A-Za-z0-9_\-\.]+)\s*$"
)


SC_ENQUEUED_RE = re.compile(
    r"\[MEAN_REVERSION/(?P<symbol>[^\]]+)\]\s+ENQUEUED\s+\|\s+.*?\bside=(?P<side>buy|sell)\b.*?\bsignal_id=(?P<signal_id>[A-Za-z0-9_\-\.]+)\b"
)


RISK_RR_PRICES_RE = re.compile(
    r"Prices:\s+Entry=\$(?P<entry>[0-9.]+),\s+Stop=\$(?P<stop>[0-9.]+).*?Target=\$(?P<target>[0-9.]+)"
)


MR_REASON_PX_BAND_RE = re.compile(
    r"\(px\s+(?P<px>[0-9.]+)\s+(?P<cmp><|>)\s+(?P<band_name>lower|upper)\s+(?P<band>[0-9.]+),\s+ADX\s+(?P<adx>[0-9.]+)\s+<\s+(?P<adx_th>[0-9.]+)\)"
)


def parse_ingress_line(line: str, fallback_run_id: str) -> Optional[IngressLine]:
    m = INGRESS_RE.search(line)
    if not m:
        return None
    ts_ms = parse_prefix_ts_ms(line)
    if ts_ms is None:
        return None
    symbol = m.group("symbol")
    side = m.group("side")
    intent = m.group("intent")
    px = _safe_float(m.group("px"))
    band_value = _safe_float(m.group("band"))
    adx = _safe_float(m.group("adx"))

    # Heuristic: if text had '< lower' it's a long signal; if '> upper' it's short.
    lower = None
    upper = None
    if "< lower" in line:
        lower = band_value
    elif "> upper" in line:
        upper = band_value

    return IngressLine(
        ts_ms=ts_ms,
        run_id=fallback_run_id,
        symbol=symbol,
        side=side,
        timeframe="5m",
        intent_hint=intent,
        entry_price=px,
        lower=lower,
        upper=upper,
        adx=adx,
    )


def parse_sc_signal_received_line(line: str, fallback_run_id: str) -> Optional[IngressLine]:
    m = SC_SIGNAL_RECEIVED_RE.search(line)
    if not m:
        return None
    ts_ms = parse_prefix_ts_ms(line)
    if ts_ms is None:
        return None

    symbol = m.group("symbol")
    side_raw = m.group("side")
    reason = m.group("reason")

    # Convert coordinator side to long/short
    side = "short" if side_raw == "sell" else "long"

    px = None
    lower = None
    upper = None
    adx = None

    m2 = MR_REASON_PX_BAND_RE.search(reason)
    if m2:
        px = _safe_float(m2.group("px"))
        band_val = _safe_float(m2.group("band"))
        adx = _safe_float(m2.group("adx"))
        if m2.group("cmp") == "<" and m2.group("band_name") == "lower":
            lower = band_val
        elif m2.group("cmp") == ">" and m2.group("band_name") == "upper":
            upper = band_val

    return IngressLine(
        ts_ms=ts_ms,
        run_id=fallback_run_id,
        symbol=symbol,
        side=side,
        timeframe="5m",
        intent_hint=None,
        entry_price=px,
        lower=lower,
        upper=upper,
        adx=adx,
    )


def compute_bps(distance: float, denom: float) -> Optional[float]:
    if denom is None or denom == 0:
        return None
    try:
        return distance / denom * 1e4
    except Exception:
        return None


def compute_target_bps(entry: Optional[float], target: Optional[float], side: Optional[str]) -> Optional[float]:
    if entry is None or target is None or side not in ("long", "short"):
        return None
    if entry == 0:
        return None
    if side == "long":
        return (target - entry) / entry * 1e4
    return (entry - target) / entry * 1e4


def compute_stop_bps(entry: Optional[float], stop: Optional[float], side: Optional[str]) -> Optional[float]:
    if entry is None or stop is None or side not in ("long", "short"):
        return None
    if entry == 0:
        return None
    if side == "long":
        return (entry - stop) / entry * 1e4
    return (stop - entry) / entry * 1e4


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{nd}f}"


def nearest_by_time(items: List[Any], ts_ms: int, max_window_ms: int) -> Optional[Any]:
    # items must have .ts_ms
    if not items:
        return None
    best = None
    best_dt = None
    for it in items:
        dt_ms = abs(it.ts_ms - ts_ms)
        if dt_ms <= max_window_ms and (best_dt is None or dt_ms < best_dt):
            best = it
            best_dt = dt_ms
    return best


def read_log_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def guess_run_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"live_trading_(\d{8}_\d{6}_\d{6})", base)
    if m:
        return m.group(1)
    # fallback
    stem, _ = os.path.splitext(base)
    return stem


def get_or_create_signal(
    signals_by_key: Dict[str, SignalRecord],
    run_id: str,
    signal_id: Optional[str],
    dedupe_key: Optional[str],
    defaults: Dict[str, Any],
    cost_bps_assumed: float,
) -> SignalRecord:
    sid = signal_id or ""
    dkey = dedupe_key or ""
    key = sid if sid else dkey
    if not key:
        # As a last resort, force a unique key.
        key = f"synthetic:run={run_id}:ts={defaults.get('ts_ms')}:sym={defaults.get('symbol')}:side={defaults.get('side')}"

    if key in signals_by_key:
        rec = signals_by_key[key]
    else:
        rec = SignalRecord(
            run_id=run_id,
            signal_id=sid,
            pending_id=None,
            parent_pending_id=None,
            dedupe_key=None,
            symbol=None,
            side=None,
            timeframe=None,
            ts_ms=None,
            final_outcome="unknown",
            final_reason_code=None,
            drop_reason=None,
            volume_bucket=None,
            volume_strength=None,
            adx=None,
            vwap_std=None,
            band_width_bps=None,
            dist_outside_bps=None,
            z=None,
            entry_price=None,
            stop_price=None,
            target_price=None,
            stop_pct_bps_observed=None,
            stop_pct_bps_model=None,
            stop_pct_bps=None,
            target_bps=None,
            std_bps=None,
            atr_bps=None,
            k_implied=None,
            gate_threshold_bps=None,
            gate_margin_bps=None,
            cost_bps_assumed=cost_bps_assumed,
            rr_gross=None,
            rr_net=None,
            rr_net_clamped0=None,
        )
        signals_by_key[key] = rec

    # Apply defaults only if field is None/empty
    for k, v in defaults.items():
        if not hasattr(rec, k):
            continue
        cur = getattr(rec, k)
        if cur in (None, "") and v not in (None, ""):
            setattr(rec, k, v)

    return rec


def assign_outcome_from_waiting_room_drop(rec: SignalRecord, payload: Dict[str, Any]) -> None:
    drop_reason = payload.get("drop_reason")
    reason_code = payload.get("reason_code")

    rec.drop_reason = rec.drop_reason or drop_reason
    rec.final_reason_code = rec.final_reason_code or reason_code

    # Gate drops
    if drop_reason == "gate_far_from_pass" or (reason_code and str(reason_code).startswith("volume.")):
        rec.final_outcome = "dropped_gate"
        return

    # Soft deferral expires / max checks
    if drop_reason in ("expired", "max_checks"):
        rec.final_outcome = "deferred"
        return

    # Soft deferral emitted a signal (not a trade acceptance)
    if drop_reason == "signal_emitted":
        rec.final_outcome = "deferred"
        return

    # Default
    rec.final_outcome = rec.final_outcome or "unknown"


def finalize_derived_metrics(rec: SignalRecord) -> None:
    # Derived bps
    if rec.stop_pct_bps is None:
        rec.stop_pct_bps = compute_stop_bps(rec.entry_price, rec.stop_price, rec.side)

    if rec.target_bps is None:
        rec.target_bps = compute_target_bps(rec.entry_price, rec.target_price, rec.side)

    if rec.entry_price is not None:
        if rec.vwap_std is not None:
            rec.std_bps = compute_bps(rec.vwap_std, rec.entry_price)

    if rec.stop_pct_bps is not None and rec.std_bps not in (None, 0):
        rec.k_implied = rec.stop_pct_bps / rec.std_bps

    # RR
    if rec.target_bps is not None and rec.stop_pct_bps not in (None, 0):
        rec.rr_gross = rec.target_bps / rec.stop_pct_bps

    if rec.target_bps is not None and rec.stop_pct_bps is not None:
        cost = rec.cost_bps_assumed
        denom = rec.stop_pct_bps + cost
        if denom != 0:
            rec.rr_net = (rec.target_bps - cost) / denom
            rec.rr_net_clamped0 = max(0.0, rec.rr_net)


def render_md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to live_trading_*.log")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--join-window-ms", type=int, default=90_000)
    ap.add_argument("--carry-forward-ms", type=int, default=15 * 60_000)
    ap.add_argument("--cost-bps", type=float, default=6.0)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    log_path = args.log
    fallback_run_id = guess_run_id_from_path(log_path)

    controllers: List[MRControllerSnapshot] = []
    volumes: List[VolumeSnapshot] = []
    ingresses: List[IngressLine] = []

    mr_stop_loss_std_delta: Optional[float] = None

    signals_by_key: Dict[str, SignalRecord] = {}
    signals_by_pending: Dict[str, SignalRecord] = {}

    # Best-effort state to attach non-JSON StrategyCoordinator/Risk lines to the most recent MR candidate.
    last_candidate_by_symbol: Dict[str, Tuple[int, SignalRecord]] = {}

    # Parse pass
    for line in read_log_lines(log_path):
        # Parse MR config (python-dict repr)
        if mr_stop_loss_std_delta is None and "- MR Config:" in line:
            try:
                # Example: "- MR Config: {'...','stop_loss_std_delta': 1.0, ...}"
                idx = line.index("- MR Config:")
                cfg_str = line[idx + len("- MR Config:") :].strip()
                cfg = ast.literal_eval(cfg_str)
                if isinstance(cfg, dict):
                    mr_stop_loss_std_delta = _safe_float(cfg.get("stop_loss_std_delta"))
            except Exception:
                mr_stop_loss_std_delta = mr_stop_loss_std_delta

        # Ingress (non-JSON)
        if "[MEAN_REVERSION/" in line and "Signal ingress" in line:
            ing = parse_ingress_line(line, fallback_run_id)
            if ing:
                ingresses.append(ing)
            continue

        # StrategyCoordinator ingress (non-JSON, alternate schema)
        if "[MEAN_REVERSION/" in line and "Signal Received." in line and "Side:" in line:
            ing = parse_sc_signal_received_line(line, fallback_run_id)
            if ing:
                ingresses.append(ing)

                dkey = f"{MR_STRATEGY}:{ing.symbol}:{ing.side}:sc_signal_received:{ing.ts_ms}"
                defaults = {
                    "symbol": ing.symbol,
                    "side": ing.side,
                    "timeframe": ing.timeframe,
                    "dedupe_key": dkey,
                    "ts_ms": ing.ts_ms,
                    "entry_price": ing.entry_price,
                    "adx": ing.adx,
                }
                rec = get_or_create_signal(
                    signals_by_key,
                    ing.run_id or fallback_run_id,
                    signal_id=None,
                    dedupe_key=dkey,
                    defaults=defaults,
                    cost_bps_assumed=args.cost_bps,
                )
                last_candidate_by_symbol[ing.symbol] = (ing.ts_ms, rec)
            continue

        # StrategyCoordinator deferral (non-JSON)
        if "[MEAN_REVERSION/" in line and "Deferring signal" in line:
            m = SC_DEFERRING_RE.search(line)
            if m:
                ts_ms = parse_prefix_ts_ms(line)
                symbol = m.group("symbol")
                reason = m.group("reason")
                if ts_ms is not None and symbol in last_candidate_by_symbol:
                    prev_ts, rec = last_candidate_by_symbol[symbol]
                    if 0 <= ts_ms - prev_ts <= 10_000:
                        rec.final_outcome = "deferred"
                        rec.final_reason_code = rec.final_reason_code or "volume.low_vol_tight_stop"
                        rec.drop_reason = rec.drop_reason or reason
                        # The deferral condition explicitly references the 15 bps (=0.15%) gate.
                        rec.gate_threshold_bps = rec.gate_threshold_bps or 15.0
            continue

        # StrategyCoordinator risk rejection (non-JSON)
        if "[MEAN_REVERSION/" in line and "REJECTED (Risk Check):" in line:
            m = SC_RISK_REJECTED_RE.search(line)
            if m:
                ts_ms = parse_prefix_ts_ms(line)
                symbol = m.group("symbol")
                reason = m.group("reason")
                if ts_ms is not None and symbol in last_candidate_by_symbol:
                    prev_ts, rec = last_candidate_by_symbol[symbol]
                    if 0 <= ts_ms - prev_ts <= 30_000:
                        rec.final_outcome = "dropped_risk"
                        rec.final_reason_code = rec.final_reason_code or str(reason)
                        rec.drop_reason = rec.drop_reason or "risk_rejected"
            continue

        # StrategyCoordinator enqueue (non-JSON)
        if "[MEAN_REVERSION/" in line and "ENQUEUED" in line and "signal_id=" in line:
            m = SC_ENQUEUED_RE.search(line)
            if m:
                ts_ms = parse_prefix_ts_ms(line)
                symbol = m.group("symbol")
                side_raw = m.group("side")
                signal_id = m.group("signal_id")

                side = "short" if side_raw == "sell" else "long"
                # Attach to most recent candidate if available, else create record.
                rec = None
                if ts_ms is not None and symbol in last_candidate_by_symbol:
                    prev_ts, prev_rec = last_candidate_by_symbol[symbol]
                    if 0 <= ts_ms - prev_ts <= 10_000:
                        rec = prev_rec

                if rec is None:
                    dkey = f"{MR_STRATEGY}:{symbol}:{side}:sc_enqueued:{ts_ms}"
                    rec = get_or_create_signal(
                        signals_by_key,
                        fallback_run_id,
                        signal_id=None,
                        dedupe_key=dkey,
                        defaults={
                            "symbol": symbol,
                            "side": side,
                            "timeframe": "5m",
                            "ts_ms": ts_ms,
                            "dedupe_key": dkey,
                        },
                        cost_bps_assumed=args.cost_bps,
                    )

                if rec.signal_id in (None, ""):
                    rec.signal_id = signal_id
                rec.symbol = rec.symbol or symbol
                rec.side = rec.side or side
                rec.final_outcome = "accepted"
            continue

        # Risk rules R/R prices line (non-JSON)
        if "[core.risk_rules]" in line and "Prices:" in line:
            m = RISK_RR_PRICES_RE.search(line)
            if m:
                ts_ms = parse_prefix_ts_ms(line)
                entry = _safe_float(m.group("entry"))
                stop = _safe_float(m.group("stop"))
                target = _safe_float(m.group("target"))

                # Best-effort: attach to most recent candidate regardless of symbol (log doesn't include symbol here).
                # We pick the freshest MR candidate within a short window.
                best: Optional[Tuple[int, SignalRecord]] = None
                if ts_ms is not None:
                    for _sym, (prev_ts, rec0) in last_candidate_by_symbol.items():
                        if 0 <= ts_ms - prev_ts <= 10_000:
                            if best is None or prev_ts > best[0]:
                                best = (prev_ts, rec0)
                if best:
                    rec = best[1]
                    if entry is not None:
                        rec.entry_price = rec.entry_price if rec.entry_price is not None else entry
                    rec.stop_price = rec.stop_price if rec.stop_price is not None else stop
                    rec.target_price = rec.target_price if rec.target_price is not None else target
                    rec.stop_pct_bps_observed = compute_stop_bps(rec.entry_price, rec.stop_price, rec.side)
                    if rec.stop_pct_bps_observed is not None:
                        rec.stop_pct_bps = rec.stop_pct_bps_observed
            continue

        payload = extract_json_from_line(line)
        if not payload:
            continue

        event = payload.get("event")
        run_id = payload.get("run_id") or fallback_run_id

        # Controller
        if event == "mr_controller_decision":
            # Schema in log is nested:
            # {"event":"mr_controller_decision","symbol":...,"ts_utc":...,"inputs":{...},"derived":{...}}
            ts_ms = (
                _safe_int(payload.get("ts_ms"))
                or parse_iso_ts_ms(payload.get("timestamp", ""))
                or parse_iso_ts_ms(payload.get("ts_utc", ""))
            )
            if ts_ms is None:
                continue

            raw_inputs = payload.get("inputs")
            raw_derived = payload.get("derived")
            inputs: Dict[str, Any] = raw_inputs if isinstance(raw_inputs, dict) else {}
            derived: Dict[str, Any] = raw_derived if isinstance(raw_derived, dict) else {}
            controllers.append(
                MRControllerSnapshot(
                    ts_ms=ts_ms,
                    run_id=run_id,
                    symbol=payload.get("symbol"),
                    timeframe=payload.get("timeframe") or "5m",
                    px=_safe_float(inputs.get("px")),
                    vwap=_safe_float(inputs.get("vwap")),
                    vwap_std=_safe_float(inputs.get("vwap_std")),
                    lower=_safe_float(derived.get("lower")),
                    upper=_safe_float(derived.get("upper")),
                    z=_safe_float(derived.get("z")),
                    adx=_safe_float(inputs.get("adx")),
                    atr=_safe_float(inputs.get("atr")),
                    atr_pct=_safe_float(inputs.get("atr_pct")),
                )
            )
            continue

        # Volume check snapshot
        if event == "volume_decision_check" and payload.get("strategy_name") == MR_STRATEGY:
            ts_ms = _safe_int(payload.get("ts_ms")) or parse_iso_ts_ms(payload.get("timestamp", ""))
            if ts_ms is None:
                continue
            volumes.append(
                VolumeSnapshot(
                    ts_ms=ts_ms,
                    run_id=run_id,
                    symbol=payload.get("symbol"),
                    timeframe=payload.get("timeframe"),
                    volume_bucket=payload.get("volume_bucket"),
                    volume_strength=_safe_float(payload.get("volume_strength")),
                    central_bucket_decision=payload.get("central_bucket_decision"),
                )
            )
            continue

        # Signal breakdown provides a canonical signal_id and timestamp (when present)
        if event == "signal_breakdown" and payload.get("strategy") == MR_STRATEGY:
            ts_ms = _safe_int(payload.get("ts_ms")) or parse_iso_ts_ms(payload.get("timestamp", ""))
            symbol = payload.get("symbol")
            signal_id = payload.get("signal_id")
            side_raw = payload.get("side")
            side = "short" if str(side_raw).lower() == "sell" else "long" if str(side_raw).lower() == "buy" else None

            if ts_ms is None or not symbol:
                continue

            # Prefer attaching to last candidate if close in time; else create new.
            rec = None
            if symbol in last_candidate_by_symbol:
                prev_ts, prev_rec = last_candidate_by_symbol[symbol]
                if 0 <= ts_ms - prev_ts <= 10_000:
                    rec = prev_rec

            if rec is None:
                dkey = f"{MR_STRATEGY}:{symbol}:{side or 'unknown'}:signal_breakdown:{ts_ms}"
                rec = get_or_create_signal(
                    signals_by_key,
                    run_id,
                    signal_id=None,
                    dedupe_key=dkey,
                    defaults={
                        "symbol": symbol,
                        "side": side,
                        "timeframe": payload.get("timeframe") or "5m",
                        "ts_ms": ts_ms,
                        "dedupe_key": dkey,
                    },
                    cost_bps_assumed=args.cost_bps,
                )
                last_candidate_by_symbol[symbol] = (ts_ms, rec)

            if signal_id and rec.signal_id in (None, ""):
                rec.signal_id = signal_id
            if side and rec.side is None:
                rec.side = side
            if rec.ts_ms is None:
                rec.ts_ms = ts_ms
            continue

        # trade_execution_size_debug has entry price + volume bucket at entry
        if event == "trade_execution_size_debug" and payload.get("strategy_name") == MR_STRATEGY:
            ts_ms = _safe_int(payload.get("ts_ms")) or parse_iso_ts_ms(payload.get("timestamp", ""))
            symbol = payload.get("symbol")
            if ts_ms is None or not symbol:
                continue
            if symbol in last_candidate_by_symbol:
                prev_ts, rec = last_candidate_by_symbol[symbol]
                if 0 <= ts_ms - prev_ts <= 30_000:
                    rec.entry_price = rec.entry_price if rec.entry_price is not None else _safe_float(payload.get("entry_price"))
                    vb = payload.get("volume_bucket_at_entry")
                    if vb:
                        rec.volume_bucket = rec.volume_bucket or vb
            continue

        # Waiting room add/drop
        if event in ("waiting_room_add", "waiting_room_drop") and payload.get("strategy") == MR_STRATEGY:
            sid = payload.get("signal_id") or ""
            dkey = payload.get("dedupe_key")
            defaults = {
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "timeframe": payload.get("timeframe"),
                "dedupe_key": payload.get("dedupe_key"),
                "pending_id": payload.get("pending_id"),
                "parent_pending_id": payload.get("parent_pending_id"),
                "ts_ms": _safe_int(payload.get("ts_ms")),
                "gate_threshold_bps": _safe_float(payload.get("gate_threshold_bps")),
                "gate_margin_bps": _safe_float(payload.get("gate_margin_bps")),
            }
            rec = get_or_create_signal(signals_by_key, run_id, sid, dkey, defaults, cost_bps_assumed=args.cost_bps)

            # Index by pending_id if present
            pid = payload.get("pending_id")
            if pid:
                signals_by_pending[pid] = rec

            # Attach pricing info from gate drops
            if event == "waiting_room_drop":
                assign_outcome_from_waiting_room_drop(rec, payload)

                # Gate-typed drop carries stop and entry
                if payload.get("px_used") is not None and rec.entry_price is None:
                    rec.entry_price = _safe_float(payload.get("px_used"))

                stop_pct = _safe_float(payload.get("stop_pct"))
                stop_dist = _safe_float(payload.get("stop_distance"))
                if stop_pct is not None:
                    rec.stop_pct_bps_observed = stop_pct * 1e4
                    rec.stop_pct_bps = rec.stop_pct_bps_observed
                elif stop_dist is not None and rec.entry_price is not None:
                    rec.stop_pct_bps_observed = compute_bps(stop_dist, rec.entry_price)
                    rec.stop_pct_bps = rec.stop_pct_bps_observed

                if rec.entry_price is not None and rec.stop_pct_bps is not None and rec.side in ("long", "short"):
                    if rec.side == "long":
                        rec.stop_price = rec.entry_price * (1.0 - rec.stop_pct_bps / 1e4)
                    else:
                        rec.stop_price = rec.entry_price * (1.0 + rec.stop_pct_bps / 1e4)

            continue

        # Recheck evaluation provides rich context keyed by pending_id/signal_id
        if event in ("strategy_recheck_request", "mr_recheck_eval", "soft_deferral_recheck_outcome", "fast_watch_rearm", "soft_deferral_rearm", "fast_watch_outcome"):
            sid = payload.get("signal_id")
            pid = payload.get("pending_id")
            rec = None
            if sid:
                rec = get_or_create_signal(signals_by_key, run_id, sid, payload.get("dedupe_key"), {}, cost_bps_assumed=args.cost_bps)
            elif pid and pid in signals_by_pending:
                rec = signals_by_pending[pid]

            if rec is None:
                continue

            # Apply identity fields
            if rec.pending_id is None and pid:
                rec.pending_id = pid
            if rec.parent_pending_id is None and payload.get("parent_pending_id"):
                rec.parent_pending_id = payload.get("parent_pending_id")
            if rec.dedupe_key is None and payload.get("dedupe_key"):
                rec.dedupe_key = payload.get("dedupe_key")
            if rec.ts_ms is None and payload.get("ts_ms"):
                rec.ts_ms = _safe_int(payload.get("ts_ms"))
            if rec.symbol is None and payload.get("symbol"):
                rec.symbol = payload.get("symbol")
            if rec.side is None and payload.get("side"):
                rec.side = payload.get("side")
            if rec.timeframe is None and payload.get("timeframe"):
                rec.timeframe = payload.get("timeframe")

            # For deferral flows, we treat as deferred unless it later becomes a gate drop.
            if rec.final_outcome == "unknown" and payload.get("intent") == "soft_deferral":
                rec.final_outcome = "deferred"

            # Some events carry band context.
            cond = payload.get("condition_data") if isinstance(payload.get("condition_data"), dict) else None
            if cond:
                rec.adx = rec.adx if rec.adx is not None else _safe_float(cond.get("adx"))
                rec.target_price = rec.target_price or _safe_float(cond.get("vwap"))
                rec.vwap_std = rec.vwap_std if rec.vwap_std is not None else _safe_float(cond.get("vwap_std"))
                rec.z = rec.z if rec.z is not None else _safe_float(cond.get("z"))
                lower = _safe_float(cond.get("lower"))
                upper = _safe_float(cond.get("upper"))
                px_used = _safe_float(cond.get("px_used")) or _safe_float(cond.get("px"))
                if rec.entry_price is None and px_used is not None:
                    rec.entry_price = px_used
                if rec.entry_price is not None and lower is not None and upper is not None:
                    rec.band_width_bps = compute_bps(upper - lower, rec.entry_price)
                    if rec.side == "long":
                        # distance below lower
                        rec.dist_outside_bps = compute_bps(max(0.0, (lower - rec.entry_price)), rec.entry_price)
                    elif rec.side == "short":
                        rec.dist_outside_bps = compute_bps(max(0.0, (rec.entry_price - upper)), rec.entry_price)

            continue

    # Determine run_id (prefer actual)
    run_id = fallback_run_id
    for snap in controllers:
        if snap.run_id:
            run_id = snap.run_id
            break
    for snap in volumes:
        if snap.run_id:
            run_id = snap.run_id
            break

    # Create records for ingress-only lines (candidate universe).
    for ing in ingresses:
        # Avoid duplicating records for the alternate StrategyCoordinator schema.
        # Those are already materialized above as ":sc_signal_received:" candidates.
        sc_key = f"{MR_STRATEGY}:{ing.symbol}:{ing.side}:sc_signal_received:{ing.ts_ms}"
        if sc_key in signals_by_key:
            continue
        dkey = f"{MR_STRATEGY}:{ing.symbol}:{ing.side}:ingress:{ing.ts_ms}"
        defaults = {
            "symbol": ing.symbol,
            "side": ing.side,
            "timeframe": ing.timeframe,
            "dedupe_key": dkey,
            "ts_ms": ing.ts_ms,
            "entry_price": ing.entry_price,
            "adx": ing.adx,
        }
        get_or_create_signal(signals_by_key, run_id, signal_id=None, dedupe_key=dkey, defaults=defaults, cost_bps_assumed=args.cost_bps)

    # Link ingress lines to signal records (best-effort)
    ingresses_by_side = defaultdict(list)
    for ing in ingresses:
        ingresses_by_side[(ing.symbol, ing.side)].append(ing)

    # Attach controller/volume snapshots to each signal.
    for rec in list(signals_by_key.values()):
        if rec.ts_ms is None:
            continue

        # Prefer explicit entry price, else from controller
        controller = nearest_by_time(
            [c for c in controllers if c.symbol == rec.symbol and c.timeframe == (rec.timeframe or c.timeframe)],
            rec.ts_ms,
            args.join_window_ms,
        )
        if controller is None:
            # carry-forward
            prior = [
                c
                for c in controllers
                if c.symbol == rec.symbol
                and (rec.timeframe is None or c.timeframe == rec.timeframe)
                and 0 <= rec.ts_ms - c.ts_ms <= args.carry_forward_ms
            ]
            if prior:
                controller = max(prior, key=lambda x: x.ts_ms)

        if controller:
            if rec.entry_price is None:
                rec.entry_price = controller.px
            rec.target_price = rec.target_price or controller.vwap
            rec.vwap_std = rec.vwap_std if rec.vwap_std is not None else controller.vwap_std
            rec.adx = rec.adx if rec.adx is not None else controller.adx
            rec.z = rec.z if rec.z is not None else controller.z

            if rec.entry_price is not None and controller.lower is not None and controller.upper is not None:
                rec.band_width_bps = rec.band_width_bps if rec.band_width_bps is not None else compute_bps(
                    controller.upper - controller.lower, rec.entry_price
                )
                if rec.dist_outside_bps is None:
                    if rec.side == "long":
                        rec.dist_outside_bps = compute_bps(max(0.0, controller.lower - rec.entry_price), rec.entry_price)
                    elif rec.side == "short":
                        rec.dist_outside_bps = compute_bps(max(0.0, rec.entry_price - controller.upper), rec.entry_price)

            # atr_bps
            if rec.entry_price is not None and controller.atr is not None and rec.atr_bps is None:
                rec.atr_bps = compute_bps(controller.atr, rec.entry_price)

        vol = nearest_by_time(
            [v for v in volumes if v.symbol == rec.symbol and v.timeframe == (rec.timeframe or v.timeframe)],
            rec.ts_ms,
            args.join_window_ms,
        )
        if vol is None:
            prior_v = [
                v
                for v in volumes
                if v.symbol == rec.symbol
                and (rec.timeframe is None or v.timeframe == rec.timeframe)
                and 0 <= rec.ts_ms - v.ts_ms <= args.carry_forward_ms
            ]
            if prior_v:
                vol = max(prior_v, key=lambda x: x.ts_ms)
        if vol:
            rec.volume_bucket = rec.volume_bucket or vol.volume_bucket
            rec.volume_strength = rec.volume_strength if rec.volume_strength is not None else vol.volume_strength

        # Best-effort ingress attachment
        if rec.symbol and rec.side:
            candidates = ingresses_by_side.get((rec.symbol, rec.side), [])
            ing = nearest_by_time(candidates, rec.ts_ms, max_window_ms=5_000)
            if ing:
                rec.adx = rec.adx if rec.adx is not None else ing.adx
                rec.entry_price = rec.entry_price if rec.entry_price is not None else ing.entry_price

        # Offline model stop (used when stop is missing in logs)
        if rec.stop_pct_bps is None and rec.entry_price is not None:
            # Prefer std-based sizing when vwap_std exists.
            if rec.std_bps is None and rec.vwap_std is not None:
                rec.std_bps = compute_bps(rec.vwap_std, rec.entry_price)
            if rec.std_bps is not None and mr_stop_loss_std_delta is not None:
                rec.stop_pct_bps_model = mr_stop_loss_std_delta * rec.std_bps
            elif rec.atr_bps is not None:
                # Fallback heuristic (only if std is unavailable)
                rec.stop_pct_bps_model = 1.5 * rec.atr_bps

            if rec.stop_pct_bps_model is not None:
                rec.stop_pct_bps = rec.stop_pct_bps_model
                if rec.side in ("long", "short"):
                    if rec.side == "long":
                        rec.stop_price = rec.entry_price * (1.0 - rec.stop_pct_bps / 1e4)
                    else:
                        rec.stop_price = rec.entry_price * (1.0 + rec.stop_pct_bps / 1e4)

        finalize_derived_metrics(rec)

    # Output paths
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_csv = args.out_csv or os.path.join(reports_dir, f"mr_signal_universe_{run_id}.csv")
    out_md = args.out_md or os.path.join(reports_dir, f"mr_signal_universe_{run_id}.md")

    # Write CSV
    rows = list(signals_by_key.values())
    rows.sort(key=lambda r: (r.ts_ms or 0, r.signal_id))

    fieldnames = [f.name for f in dataclasses.fields(SignalRecord)]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(dataclasses.asdict(r))

    # Stats
    total = len(rows)
    outcomes = Counter([r.final_outcome or "unknown" for r in rows])
    buckets = Counter([r.volume_bucket or "(missing)" for r in rows])

    micro = [r for r in rows if r.stop_pct_bps is not None and r.stop_pct_bps < 15]
    micro_by_bucket = defaultdict(lambda: [0, 0])
    for r in rows:
        b = r.volume_bucket or "(missing)"
        if r.stop_pct_bps is not None:
            micro_by_bucket[b][1] += 1
            if r.stop_pct_bps < 15:
                micro_by_bucket[b][0] += 1

    missing_target = sum(1 for r in rows if r.target_bps is None)

    # Quantiles per bucket
    metrics = [
        "stop_pct_bps",
        "target_bps",
        "std_bps",
        "atr_bps",
        "rr_gross",
        "rr_net",
        "band_width_bps",
        "dist_outside_bps",
    ]

    def bucket_values(bucket: str, metric: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            if (r.volume_bucket or "(missing)") != bucket:
                continue
            v = getattr(r, metric)
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            out.append(float(v))
        return out

    # >=15 bps stop cases
    big_stop = [r for r in rows if r.stop_pct_bps is not None and r.stop_pct_bps >= 15]
    big_stop.sort(key=lambda r: (r.ts_ms or 0))

    # Accepted cases (if any)
    accepted = [r for r in rows if r.final_outcome == "accepted"]

    # Render MD
    md: List[str] = []
    md.append(f"# MR Signal Universe Report ({run_id})")
    md.append("")

    # (1) Executive summary
    md.append("## (1) Executive summary")
    md.append("")
    md.append(f"- Total MR signals (universe): {total}")
    md.append("- Outcome distribution: " + ", ".join([f"{k}={v}" for k, v in outcomes.most_common()]))
    md.append("- Volume bucket distribution: " + ", ".join([f"{k}={v}" for k, v in buckets.most_common()]))
    md.append(
        "- stop_pct_bps source: observed(from risk_rules Prices or gate drop) else std-model (stop_loss_std_delta) else atr-fallback"
    )
    md.append(
        f"- Missing target ratio: {missing_target}/{total} ({(missing_target/total*100 if total else 0):.1f}%)"
    )
    md.append(
        f"- Micro-stop rate (<15 bps): {len(micro)}/{sum(1 for r in rows if r.stop_pct_bps is not None)} "
        f"({(len(micro)/max(1, sum(1 for r in rows if r.stop_pct_bps is not None))*100):.1f}%)"
    )
    for b, (m_cnt, denom) in micro_by_bucket.items():
        if denom == 0:
            continue
        md.append(f"  - {b}: {m_cnt}/{denom} ({(m_cnt/denom*100):.1f}%)")
    md.append("")

    # (2) Bucket quantiles
    md.append("## (2) Bucket bazlı quantile tabloları")
    md.append("")
    for b in buckets.keys():
        md.append(f"### Bucket: {b}")
        md.append("")
        headers = ["metric", "p10", "p50", "p90", "n"]
        q_rows: List[List[str]] = []
        for mname in metrics:
            vals = bucket_values(b, mname)
            q_rows.append(
                [
                    mname,
                    fmt_num(percentile(vals, 10)),
                    fmt_num(percentile(vals, 50)),
                    fmt_num(percentile(vals, 90)),
                    str(len(vals)),
                ]
            )
        md.append(render_md_table(headers, q_rows))
        md.append("")

    # (3) >=15 bps stop cases
    md.append('## (3) ">=15 bps stop üreten MR" varsa: karakteristikleri')
    md.append("")
    if not big_stop:
        md.append("- Count: 0")
        md.append("")
    else:
        md.append(f"- Count: {len(big_stop)}")
        md.append("- Bucket breakdown: " + ", ".join([f"{k}={v}" for k, v in Counter([r.volume_bucket or '(missing)' for r in big_stop]).most_common()]))
        md.append("")
        headers = ["ts_ms", "volume_bucket", "volume_strength", "adx", "std_bps", "target_bps", "stop_pct_bps", "rr_net", "outcome"]
        rws: List[List[str]] = []
        for r in big_stop[: args.top_n]:
            rws.append(
                [
                    str(r.ts_ms or ""),
                    r.volume_bucket or "",
                    fmt_num(r.volume_strength, 3),
                    fmt_num(r.adx, 2),
                    fmt_num(r.std_bps, 2),
                    fmt_num(r.target_bps, 2),
                    fmt_num(r.stop_pct_bps, 2),
                    fmt_num(r.rr_net, 2),
                    r.final_outcome,
                ]
            )
        md.append(render_md_table(headers, rws))
        md.append("")

    # (4) Accepted profile
    md.append('## (4) "Accepted" MR işlemleri varsa: profil')
    md.append("")
    if not accepted:
        md.append("- Accepted count: 0")
        md.append("")
    else:
        md.append(f"- Accepted count: {len(accepted)}")
        md.append("- Bucket distribution: " + ", ".join([f"{k}={v}" for k, v in Counter([r.volume_bucket or '(missing)' for r in accepted]).most_common()]))
        rr_vals = [r.rr_net for r in accepted if r.rr_net is not None]
        md.append(
            "- rr_net p10/p50/p90: "
            + "/".join([fmt_num(percentile(rr_vals, 10)), fmt_num(percentile(rr_vals, 50)), fmt_num(percentile(rr_vals, 90))])
        )
        md.append("")

    # (5) Root-cause decision sentence
    md.append("## (5) Root-cause karar cümlesi")
    md.append("")

    # Q1: micro-stop only LOW?
    if set([b for b in buckets.keys() if b not in ("LOW", "(missing)")]):
        md.append("- Mikro-stop patolojisi yalnız LOW bucket’ta mı yoğun? **Kıyas mümkün** (bu run’da LOW dışı bucket var).")
    else:
        md.append("- Mikro-stop patolojisi yalnız LOW bucket’ta mı yoğun? **Bu run ile kıyaslanamaz** (volume_bucket fiilen LOW).")

    low_rows = [r for r in rows if (r.volume_bucket or "(missing)") == "LOW"]
    low_rr = [r.rr_net for r in low_rows if r.rr_net is not None]
    low_target = [r.target_bps for r in low_rows if r.target_bps is not None]

    low_rr_median = percentile([float(x) for x in low_rr], 50) if low_rr else None
    low_target_median = percentile([float(x) for x in low_target], 50) if low_target else None

    md.append(
        "- LOW’da target_bps ölçeği net-of-cost tradeable olmaya yetiyor mu? "
        f"median(target_bps)={fmt_num(low_target_median, 2)}, median(rr_net)={fmt_num(low_rr_median, 2)}"
    )
    md.append("")

    # Decision criteria block
    md.append("## (6) Karar kriterleri (kısa öneri)")
    md.append("")
    md.append("- Kriter A (Rejim filtresi):")
    if low_rr_median is not None and low_target_median is not None and low_rr_median < 1.0 and 10.0 <= low_target_median <= 12.0:
        md.append("  - LOW bucket: median(rr_net)<1 ve median(target_bps)≈10–12 → LOW’da MR tradeability yok; stop şişirmek yerine min reward/mesafe filtresi veya LOW’da trade etmeme.")
    else:
        md.append("  - LOW bucket: koşul net değil (bu run’daki medyanlara göre karar verilecek).")

    md.append("- Kriter B (Stop modeli):")
    non_low = [b for b in buckets.keys() if b not in ("LOW", "(missing)")]
    if non_low:
        md.append("  - NORMAL/HIGH/EXTREME bucket da var → mikro-stop oranlarını bucket bazında kontrol et; yüksekse stop ölçek problemi genel.")
    else:
        md.append("  - Bu run’da LOW dışı bucket yok → B kriteri için karşılaştırmalı veri yok.")

    md.append("- Kriter C (İstisna vakalar):")
    high_reward = [r for r in rows if r.target_bps is not None and r.target_bps > 30 and r.stop_pct_bps is not None and r.stop_pct_bps < 15]
    if high_reward:
        md.append(f"  - target_bps>30 ve stop mikro olan {len(high_reward)} vaka var → reward-consistent stop yalnız bu istisnalara uygulanabilir.")
    else:
        md.append("  - target_bps>30 & mikro-stop kombinasyonu bulunmadı (bu run’da).")

    md.append("")
    md.append("---")
    md.append(f"Generated with cost_bps_assumed={args.cost_bps:.1f}.")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD:  {out_md}")
    print(f"Signals: {total} | outcomes={dict(outcomes)} | buckets={dict(buckets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
