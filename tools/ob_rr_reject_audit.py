#!/usr/bin/env python3
"""
Offline audit for RiskRewardRatioRule-rejected AdaptiveOversoldBounce (adaptive_ob) signals.

This tool does NOT modify production behavior. It parses a live trading log plus an
existing rejected-cases JSONL, enriches each case with nearby log context (spread,
volume/momentum, PPO decision, trigger source), then proposes offline SL/TP levels
and recomputes the implied R/R.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
    r"\[(?P<logger>[^\]]+)\] - "
    r"(?P<level>[A-Z]+) - "
    r"(?P<msg>.*)$"
)

REJECT_RE = re.compile(r"\[RiskRewardRatioRule\] REJECTED (?P<symbol>.+?): Risk/reward ratio")
OB_RR_RE = re.compile(
    r"\[OB R/R\] Entry=\$(?P<entry>\d+(?:\.\d+)?), "
    r"Stop=\$(?P<stop>\d+(?:\.\d+)?).+?"
    r"Target=\$(?P<target>\d+(?:\.\d+)?).+?"
    r"R/R=(?P<rr>\d+(?:\.\d+)?)"
)
TRIGGER_DIAG_RE = re.compile(
    r"\[TRIGGER-DIAG\] exchange=(?P<exchange>\S+) symbol=(?P<symbol>\S+) "
    r"requested_source=(?P<requested_source>\S+) resolved_source=(?P<resolved_source>\S+) "
    r"mark=(?P<mark>\S+) bid=(?P<bid>\S+) ask=(?P<ask>\S+) last=(?P<last>\S+) "
    r"ticker_age_ms=(?P<ticker_age_ms>\S+)"
)
PPO_DECISION_RE = re.compile(
    r"\[PPO-DECISION\] (?P<symbol>\S+) \| Action: (?P<action>\w+) \| Score: (?P<score>-?\d+(?:\.\d+)?) \| Conf: (?P<conf>-?\d+(?:\.\d+)?)"
)
SIGNAL_INGRESS_RE = re.compile(
    r"\[ADAPTIVE_OB/(?P<symbol>[^\]]+)\] Signal ingress \| side=(?P<side>\w+).+?reason=(?P<reason>.+)$"
)


def _parse_dt(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _clamp01(value: Optional[float], default: float = 0.5) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return max(0.0, min(1.0, float(value)))


def read_log(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n")
            m = LOG_LINE_RE.match(raw)
            if not m:
                continue
            dt = _parse_dt(m.group("ts"))
            if not dt:
                continue
            out.append(
                {
                    "line_no": line_no,
                    "ts": m.group("ts"),
                    "dt": dt,
                    "logger": m.group("logger"),
                    "level": m.group("level"),
                    "msg": m.group("msg"),
                    "raw": raw,
                }
            )
    return out


def _find_prev(lines: List[Dict[str, Any]], start_idx: int, *, window_s: int, pred) -> Optional[Dict[str, Any]]:
    start_dt = lines[start_idx]["dt"]
    min_dt = start_dt - timedelta(seconds=window_s)
    for j in range(start_idx, -1, -1):
        if lines[j]["dt"] < min_dt:
            return None
        if pred(lines[j]):
            return lines[j]
    return None


def _json_from_msg(msg: str) -> Optional[Dict[str, Any]]:
    brace = msg.find("{")
    if brace < 0:
        return None
    try:
        return json.loads(msg[brace:])
    except Exception:
        return None


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                cases.append(json.loads(raw))
    return cases


def extract_ob_config(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    for ln in lines[:600]:
        if "OB Config:" not in ln["msg"]:
            continue
        _, _, tail = ln["msg"].partition("OB Config:")
        try:
            cfg = ast.literal_eval(tail.strip())
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}
    return {}


def index_rejects(lines: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for i, ln in enumerate(lines):
        if "[RiskRewardRatioRule] REJECTED" not in ln["msg"]:
            continue
        m = REJECT_RE.search(ln["msg"])
        if not m:
            continue
        out[(ln["ts"], m.group("symbol"))] = i
    return out


def _parse_trigger_diag(msg: str) -> Dict[str, Any]:
    m = TRIGGER_DIAG_RE.search(msg)
    if not m:
        return {}
    bid = _to_float(m.group("bid"))
    ask = _to_float(m.group("ask"))
    spread = (ask - bid) if (bid is not None and ask is not None) else None
    return {
        "requested_source": m.group("requested_source"),
        "resolved_source": m.group("resolved_source"),
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "ticker_age_ms": int(m.group("ticker_age_ms")) if m.group("ticker_age_ms").isdigit() else None,
    }


def _parse_signal_enriched(msg: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m_vol = re.search(r"Vol=(?P<vol>\d+(?:\.\d+)?)\s+\[(?P<bucket>[A-Z_]+)\]", msg)
    if m_vol:
        out["volume_strength"] = float(m_vol.group("vol"))
        out["volume_bucket"] = m_vol.group("bucket")
    m_mom = re.search(r"Mom=(?P<mom>\d+(?:\.\d+)?)", msg)
    if m_mom:
        out["momentum_strength"] = float(m_mom.group("mom"))
    m_ppo = re.search(r"PPO_RR=(?P<ppo>\d+(?:\.\d+)?)", msg)
    if m_ppo:
        out["ppo_rr_multiplier"] = float(m_ppo.group("ppo"))
    return out


def _parse_hybrid_meta(msg: str) -> Dict[str, Any]:
    pairs = dict(re.findall(r"(\w+)=([^\s]+)", msg))
    out: Dict[str, Any] = {}
    if "trigger_price_source" in pairs:
        out["trigger_price_source"] = pairs["trigger_price_source"]
    if "trigger_price" in pairs:
        out["trigger_price"] = _to_float(pairs["trigger_price"])
    return out


def _calc_rr(entry: float, stop: float, target: float, side: str) -> Tuple[float, float, float]:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0, 0.0, reward
    return reward / risk, risk, reward


def model_tp_only(entry: float, stop: float, required_rr: float, side: str) -> Tuple[float, float]:
    risk = abs(entry - stop)
    return stop, (entry - risk * required_rr) if side in {"short", "sell"} else (entry + risk * required_rr)


def model_sl_only(entry: float, target: float, required_rr: float, side: str) -> Tuple[float, float]:
    reward = abs(target - entry)
    needed_risk = (reward / required_rr) if required_rr > 0 else 0.0
    stop = (entry + needed_risk) if side in {"short", "sell"} else (entry - needed_risk)
    return stop, target


def model_hybrid(
    *,
    entry: float,
    base_risk: float,
    required_rr: float,
    side: str,
    volume_strength: Optional[float],
    momentum_strength: Optional[float],
    spread: Optional[float],
    k_strength: float = 0.6,
    k_spread: float = 2.0,
) -> Tuple[float, float, Dict[str, Any]]:
    v = _clamp01(volume_strength, 0.5)
    m = _clamp01(momentum_strength, 0.5)
    strength = 0.5 * (v + m)
    stop_scale = 1.0 + k_strength * (0.5 - strength)
    stop_dist = max(0.0, base_risk * stop_scale) + (k_spread * (spread or 0.0))
    stop = (entry + stop_dist) if side in {"short", "sell"} else (entry - stop_dist)
    target = (entry - stop_dist * required_rr) if side in {"short", "sell"} else (entry + stop_dist * required_rr)
    return stop, target, {"strength": strength, "stop_scale": stop_scale, "stop_dist": stop_dist}


def enrich_cases(
    *,
    lines: List[Dict[str, Any]],
    base_cases: List[Dict[str, Any]],
    strategy: str,
) -> List[Dict[str, Any]]:
    rejects = index_rejects(lines)
    out: List[Dict[str, Any]] = []
    for base in base_cases:
        if str(base.get("strategy")) != strategy:
            continue
        ts = str(base.get("ts"))
        symbol = str(base.get("symbol"))
        idx = rejects.get((ts, symbol))
        rec = dict(base)
        rec["log_reject_line_no"] = lines[idx]["line_no"] if idx is not None else None

        if idx is not None:
            ob_rr_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("adaptive_ob") and "[OB R/R]" in l["msg"])
            if ob_rr_ln:
                rec["ln_ob_rr"] = ob_rr_ln["line_no"]
                rec.update(_parse_ob_rr(ob_rr_ln["msg"]))

            trig_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("market_data_pipeline") and "[TRIGGER-DIAG]" in l["msg"] and f"symbol={symbol}" in l["msg"])
            if trig_ln:
                rec["ln_trigger_diag"] = trig_ln["line_no"]
                rec.update(_parse_trigger_diag(trig_ln["msg"]))

            vol_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("strategy_coordinator") and l["msg"].startswith("volume_decision_check") and symbol in l["msg"])
            if vol_ln:
                rec["ln_volume_decision_check"] = vol_ln["line_no"]
                payload = _json_from_msg(vol_ln["msg"]) or {}
                for k in (
                    "volume_ratio_short",
                    "volume_ratio_medium",
                    "volume_ratio_combined",
                    "current_window_volume",
                    "short_baseline_volume",
                    "medium_baseline_volume",
                    "central_bucket_decision",
                    "timeframe",
                ):
                    if k in payload:
                        rec[k] = payload.get(k)

            enriched_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("strategy_coordinator") and "[Signal Enriched]" in l["msg"] and symbol in l["msg"])
            if enriched_ln:
                rec["ln_signal_enriched"] = enriched_ln["line_no"]
                rec.update(_parse_signal_enriched(enriched_ln["msg"]))

            ppo_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("strategy_coordinator") and "[PPO-DECISION]" in l["msg"] and symbol in l["msg"])
            if ppo_ln:
                rec["ln_ppo_decision"] = ppo_ln["line_no"]
                m = PPO_DECISION_RE.search(ppo_ln["msg"])
                if m:
                    rec["ppo_action"] = m.group("action")
                    rec["ppo_score"] = float(m.group("score"))
                    rec["ppo_confidence"] = float(m.group("conf"))

            meta_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("adaptive_ob") and "Hybrid meta" in l["msg"] and symbol in l["msg"])
            if meta_ln:
                rec["ln_hybrid_meta"] = meta_ln["line_no"]
                rec.update(_parse_hybrid_meta(meta_ln["msg"]))

            ingress_ln = _find_prev(lines, idx, window_s=30, pred=lambda l: l["logger"].endswith("strategy_coordinator") and "Signal ingress" in l["msg"] and symbol in l["msg"])
            if ingress_ln:
                rec["ln_signal_ingress"] = ingress_ln["line_no"]
                m = SIGNAL_INGRESS_RE.search(ingress_ln["msg"])
                if m:
                    rec["side"] = m.group("side")
                    rec["reason"] = m.group("reason")
                    m2 = re.search(r"RSI\\s+(?P<rsi>\\d+(?:\\.\\d+)?)\\s+<=\\s+(?P<th>\\d+(?:\\.\\d+)?)", rec["reason"])
                    if m2:
                        rec["rsi"] = float(m2.group("rsi"))
                        rec["rsi_threshold"] = float(m2.group("th"))

        # Normalize naming
        rec["side"] = "long" if str(rec.get("side", "long")).lower() in {"buy", "long"} else "short"
        rec["required_rr"] = _to_float(rec.get("required_rr_v1")) or _to_float(rec.get("dyn_final")) or _to_float(rec.get("required_rr"))
        rec["actual_rr"] = _to_float(rec.get("actual_rr")) or _to_float(rec.get("rr_actual"))

        out.append(rec)
    return out


def _parse_ob_rr(msg: str) -> Dict[str, Any]:
    m = OB_RR_RE.search(msg)
    if not m:
        return {}
    return {
        "ob_entry": float(m.group("entry")),
        "ob_stop": float(m.group("stop")),
        "ob_target": float(m.group("target")),
        "ob_rr": float(m.group("rr")),
    }


def compute_models(cases: List[Dict[str, Any]]) -> None:
    eps = 1e-9
    for c in cases:
        entry = float(c["entry"])
        stop = float(c["stop"])
        target = float(c["target"])
        side = str(c.get("side") or "long").lower()
        required = float(c.get("required_rr") or 0.0)

        rr_cur, risk, reward = _calc_rr(entry, stop, target, side)
        c["rr_current_calc"] = rr_cur
        c["risk_current"] = risk
        c["reward_current"] = reward

        s, t = model_tp_only(entry, stop, required, side)
        rr, _, _ = _calc_rr(entry, s, t, side)
        c.update({"m_tp_only_stop": s, "m_tp_only_target": t, "m_tp_only_rr": rr, "m_tp_only_pass": (rr + eps) >= required})

        s, t = model_sl_only(entry, target, required, side)
        rr, _, _ = _calc_rr(entry, s, t, side)
        c.update({"m_sl_only_stop": s, "m_sl_only_target": t, "m_sl_only_rr": rr, "m_sl_only_pass": (rr + eps) >= required})

        s, t, meta = model_hybrid(
            entry=entry,
            base_risk=risk,
            required_rr=required,
            side=side,
            volume_strength=_to_float(c.get("volume_strength")),
            momentum_strength=_to_float(c.get("momentum_strength")),
            spread=_to_float(c.get("spread")),
        )
        rr, _, _ = _calc_rr(entry, s, t, side)
        c.update({"m_hybrid_stop": s, "m_hybrid_target": t, "m_hybrid_rr": rr, "m_hybrid_pass": (rr + eps) >= required, "m_hybrid_meta": meta})

        c["recommended_model"] = "hybrid"
        c["recommended_stop"] = s
        c["recommended_target"] = t
        c["recommended_rr"] = rr
        c["recommended_pass"] = (rr + eps) >= required

        # Simple deltas for reporting/CSV
        try:
            c["delta_tp_only_target_abs"] = abs(float(c["m_tp_only_target"]) - target)
            c["delta_sl_only_stop_abs"] = abs(float(c["m_sl_only_stop"]) - stop)
            c["delta_hybrid_target_abs"] = abs(float(c["m_hybrid_target"]) - target)
            c["delta_hybrid_stop_abs"] = abs(float(c["m_hybrid_stop"]) - stop)
        except Exception:
            pass

        meta = c.get("m_hybrid_meta") if isinstance(c.get("m_hybrid_meta"), dict) else {}
        strength = meta.get("strength")
        stop_scale = meta.get("stop_scale")
        stop_dist = meta.get("stop_dist")
        spread = _to_float(c.get("spread")) or 0.0
        if isinstance(strength, (int, float)) and isinstance(stop_scale, (int, float)) and isinstance(stop_dist, (int, float)):
            c["recommended_rationale"] = (
                "hybrid: strength=(volume_strength+momentum_strength)/2="
                f"{float(strength):.2f}; stop_scale=1+0.6*(0.5-strength)={float(stop_scale):.3f}; "
                f"stop_dist=risk*stop_scale + 2*spread={float(stop_dist):.2f} (spread={spread:.2f}); "
                "TP=entry + stop_dist*required_rr; SL=entry - stop_dist"
            )
        else:
            c["recommended_rationale"] = "hybrid: insufficient context; fell back to basic stop/target re-derivation"


def write_csv(path: Path, cases: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for c in cases:
        row = dict(c)
        meta = row.pop("m_hybrid_meta", None) or {}
        if isinstance(meta, dict):
            for k, v in meta.items():
                row[f"m_hybrid_{k}"] = v
        rows.append(row)

    preferred = [
        "ts",
        "symbol",
        "strategy",
        "side",
        "entry",
        "stop",
        "target",
        "rr_current_calc",
        "required_rr",
        "volume_bucket",
        "volume_strength",
        "momentum_strength",
        "bid",
        "ask",
        "spread",
        "requested_source",
        "resolved_source",
        "trigger_price_source",
        "ticker_age_ms",
        "rsi",
        "rsi_threshold",
        "volume_ratio_short",
        "volume_ratio_medium",
        "volume_ratio_combined",
        "m_tp_only_target",
        "m_tp_only_pass",
        "m_sl_only_stop",
        "m_sl_only_pass",
        "m_hybrid_stop",
        "m_hybrid_target",
        "m_hybrid_pass",
        "recommended_model",
        "recommended_stop",
        "recommended_target",
        "recommended_rr",
        "recommended_pass",
        "delta_tp_only_target_abs",
        "delta_sl_only_stop_abs",
        "delta_hybrid_target_abs",
        "delta_hybrid_stop_abs",
        "log_reject_line_no",
        "ln_ob_rr",
        "ln_trigger_diag",
        "ln_volume_decision_check",
        "ln_signal_enriched",
    ]
    all_keys: List[str] = []
    seen = set()
    for k in preferred:
        seen.add(k)
        all_keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path: Path, cases: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            payload = {
                "ts": c.get("ts"),
                "symbol": c.get("symbol"),
                "strategy": c.get("strategy"),
                "side": c.get("side"),
                "entry": c.get("entry"),
                "current": {"stop": c.get("stop"), "target": c.get("target"), "actual_rr": c.get("rr_current_calc"), "required_rr": c.get("required_rr")},
                "recommended": {
                    "model": c.get("recommended_model"),
                    "stop": c.get("recommended_stop"),
                    "target": c.get("recommended_target"),
                    "rr": c.get("recommended_rr"),
                    "passes_required": bool(c.get("recommended_pass")),
                    "rationale": c.get("recommended_rationale"),
                },
                "alternatives": {
                    "tp_only": {"stop": c.get("m_tp_only_stop"), "target": c.get("m_tp_only_target"), "rr": c.get("m_tp_only_rr")},
                    "sl_only": {"stop": c.get("m_sl_only_stop"), "target": c.get("m_sl_only_target"), "rr": c.get("m_sl_only_rr")},
                    "hybrid": {"stop": c.get("m_hybrid_stop"), "target": c.get("m_hybrid_target"), "rr": c.get("m_hybrid_rr"), "meta": c.get("m_hybrid_meta")},
                },
                "inputs_used": {"volume_strength": c.get("volume_strength"), "momentum_strength": c.get("momentum_strength"), "spread": c.get("spread")},
                "evidence": {
                    "log_reject_line_no": c.get("log_reject_line_no"),
                    "ln_ob_rr": c.get("ln_ob_rr"),
                    "ln_trigger_diag": c.get("ln_trigger_diag"),
                    "ln_volume_decision_check": c.get("ln_volume_decision_check"),
                    "ln_signal_enriched": c.get("ln_signal_enriched"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_md(path: Path, *, cases: List[Dict[str, Any]], ob_config: Dict[str, Any], log_path: Path, cases_path: Path) -> None:
    def _nums(key: str) -> List[float]:
        out: List[float] = []
        for c in cases:
            v = c.get(key)
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                out.append(fv)
        return out

    def _mean(vals: List[float]) -> float:
        return (sum(vals) / len(vals)) if vals else float("nan")

    def _median(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        mid = len(s) // 2
        return s[mid] if (len(s) % 2 == 1) else (s[mid - 1] + s[mid]) / 2.0

    tp_atr = ob_config.get("tp_atr_mult")
    sl_atr = ob_config.get("sl_atr_mult")
    intended_rr = None
    if isinstance(tp_atr, (int, float)) and isinstance(sl_atr, (int, float)) and sl_atr:
        intended_rr = float(tp_atr) / float(sl_atr)

    matched = sum(1 for c in cases if c.get("log_reject_line_no") is not None)
    rr_cur_vals = _nums("rr_current_calc")
    rr_req_vals = _nums("required_rr")
    spread_vals = _nums("spread")

    tp_pass = sum(1 for c in cases if c.get("m_tp_only_pass"))
    sl_pass = sum(1 for c in cases if c.get("m_sl_only_pass"))
    hy_pass = sum(1 for c in cases if c.get("m_hybrid_pass"))

    strength_vals: List[float] = []
    stop_scale_vals: List[float] = []
    for c in cases:
        meta = c.get("m_hybrid_meta")
        if not isinstance(meta, dict):
            continue
        st = _to_float(meta.get("strength"))
        sc = _to_float(meta.get("stop_scale"))
        if st is not None:
            strength_vals.append(st)
        if sc is not None:
            stop_scale_vals.append(sc)

    lines: List[str] = []
    lines.append("# OB RR-Reject Audit (2026-01-20)")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Log: `{log_path}`")
    lines.append(f"- Input cases: `{cases_path}`")
    lines.append(f"- Cases: **{len(cases)}** (matched to log rejects: **{matched}/{len(cases)}**)")
    lines.append(
        f"- RR(cur) mean={_mean(rr_cur_vals):.4f} median={_median(rr_cur_vals):.4f} | "
        f"RR(required) mean={_mean(rr_req_vals):.4f} median={_median(rr_req_vals):.4f}"
    )
    if intended_rr is not None:
        lines.append(f"- Constant RR driver: `tp_atr_mult/sl_atr_mult` = {tp_atr}/{sl_atr} = **{intended_rr:.4f}**")
    lines.append(f"- Offline models (pass counts): TP-only **{tp_pass}/{len(cases)}**, SL-only **{sl_pass}/{len(cases)}**, Hybrid **{hy_pass}/{len(cases)}**")
    lines.append("")

    lines.append("## Evidence Anchors (Log Line Numbers)")
    lines.append("| ts | reject | ob_rr | trigger_diag | volume_decision | signal_enriched | ppo_decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        lines.append(
            f"| {c.get('ts')} | {c.get('log_reject_line_no','')} | {c.get('ln_ob_rr','')} | {c.get('ln_trigger_diag','')} | {c.get('ln_volume_decision_check','')} | {c.get('ln_signal_enriched','')} | {c.get('ln_ppo_decision','')} |"
        )
    lines.append("")

    lines.append("## Constant RR Phenomenon (Code + Config)")
    if intended_rr is not None:
        lines.append(f"- Startup log prints OB config: `tp_atr_mult={tp_atr}`, `sl_atr_mult={sl_atr}` -> intended RR ~ {intended_rr:.4f}.")
    else:
        lines.append("- Startup log prints OB config, but tp/sl ATR multipliers could not be parsed.")
    lines.append("- Strategy SL/TP derivation uses ATR multipliers and realigns TP to preserve intended RR (`src/strategies/adaptive_ob.py:1245`).")
    lines.append("- RiskRewardRatioRule 'Actual RR' uses only `entry/stop/target` (no spread/fees) (`src/core/risk_rules.py:633`).")
    lines.append("")

    lines.append("## Unused / Under-used Parameters Inventory (Log -> Usage)")
    lines.append("| Parameter | Seen in log | Used in OB SL/TP? | Used in Actual RR? | Notes / Evidence |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append("| volume_strength | Yes | No | No | Logged in `[Signal Enriched]` and `volume_decision_check`; used for volume gating/telemetry (`src/core/strategy_coordinator.py:5465`). |")
    lines.append("| momentum_strength | Yes | No | No | Logged in `[Signal Enriched]`; not used by OB SL/TP. |")
    lines.append("| bid/ask spread | Yes | No | No | Logged in `[TRIGGER-DIAG]` (`src/core/market_data_pipeline.py:1574`); not used in RR computation. |")
    lines.append("| volume_ratio_short/medium/combined | Yes | No | No | Logged via `volume_decision_check`; not consumed by RR/SLTP. |")
    lines.append("")
    lines.append("Note: `Vol=... [BUCKET]` in these logs refers to **volume strength**, not volatility.")
    lines.append("")

    lines.append("## Hybrid Model Sensitivity (This Tool)")
    if strength_vals:
        lines.append(f"- strength range: min={min(strength_vals):.2f} max={max(strength_vals):.2f} mean={_mean(strength_vals):.2f}")
    if stop_scale_vals:
        lines.append(f"- stop_scale range: min={min(stop_scale_vals):.3f} max={max(stop_scale_vals):.3f} mean={_mean(stop_scale_vals):.3f}")
    if spread_vals:
        lines.append(f"- spread range: min={min(spread_vals):.2f} max={max(spread_vals):.2f} mean={_mean(spread_vals):.2f}")
    lines.append("- Formula: `stop_scale = 1 + 0.6*(0.5 - strength)` and `stop_dist = risk*stop_scale + 2*spread`")
    lines.append("")

    lines.append("## Aggregate Deltas (Absolute)")
    tp_d = _nums("delta_tp_only_target_abs")
    sl_d = _nums("delta_sl_only_stop_abs")
    hy_tp_d = _nums("delta_hybrid_target_abs")
    hy_sl_d = _nums("delta_hybrid_stop_abs")
    lines.append("| metric | tp_only ΔTP | sl_only ΔSL | hybrid ΔTP / ΔSL |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| mean | {(_mean(tp_d) if tp_d else float('nan')):.2f} | {(_mean(sl_d) if sl_d else float('nan')):.2f} | {(_mean(hy_tp_d) if hy_tp_d else float('nan')):.2f} / {(_mean(hy_sl_d) if hy_sl_d else float('nan')):.2f} |")
    lines.append(f"| median | {(_median(tp_d) if tp_d else float('nan')):.2f} | {(_median(sl_d) if sl_d else float('nan')):.2f} | {(_median(hy_tp_d) if hy_tp_d else float('nan')):.2f} / {(_median(hy_sl_d) if hy_sl_d else float('nan')):.2f} |")
    lines.append("")

    lines.append("## Cases (Current vs Proposed)")
    lines.append("| ts | entry | SL(cur) | TP(cur) | RR(cur) | RR(req) | vol_str | mom_str | spread | TP-only TP | SL-only SL | Hybrid TP | Hybrid SL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        v = _to_float(c.get("volume_strength"))
        m = _to_float(c.get("momentum_strength"))
        sp = _to_float(c.get("spread"))
        lines.append(
            f"| {c['ts']} | {c['entry']:.2f} | {c['stop']:.2f} | {c['target']:.2f} | {c['rr_current_calc']:.4f} | {c['required_rr']:.4f} | "
            f"{(v if v is not None else float('nan')):.2f} | {(m if m is not None else float('nan')):.2f} | {(sp if sp is not None else float('nan')):.2f} | "
            f"{c['m_tp_only_target']:.2f} | {c['m_sl_only_stop']:.2f} | {c['m_hybrid_target']:.2f} | {c['m_hybrid_stop']:.2f} |"
        )
    lines.append("")

    lines.append("## Deep Dives (3 Cases)")
    sample = [cases[0], cases[len(cases) // 2], cases[-1]] if len(cases) >= 3 else cases
    for c in sample:
        lines.append(f"### {c.get('ts')} {c.get('symbol')}")
        lines.append(f"- Entry={c.get('entry'):.2f} SL(cur)={c.get('stop'):.2f} TP(cur)={c.get('target'):.2f} RR(cur)={c.get('rr_current_calc'):.4f} RR(req)={c.get('required_rr'):.4f}")
        lines.append(f"- VolStrength={c.get('volume_strength')} MomStrength={c.get('momentum_strength')} Spread={c.get('spread')} TriggerSrc={c.get('trigger_price_source')}")
        lines.append(f"- Recommended ({c.get('recommended_model')}): SL={c.get('recommended_stop'):.2f} TP={c.get('recommended_target'):.2f} RR={c.get('recommended_rr'):.4f}")
        if c.get("recommended_rationale"):
            lines.append(f"- Rationale: {c.get('recommended_rationale')}")
        lines.append("")

    lines.append("## Notes / Limitations")
    lines.append("- No forward-window backtest included (OHLCV source not derived from this log alone).")
    lines.append("- `[TRIGGER-DIAG]` logging is throttled; bid/ask spread is unavailable for some cases in this run.")
    lines.append("- Offline levels do not apply exchange tick-size rounding or fees/slippage unless explicitly modeled.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--strategy", default="adaptive_ob")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    args = p.parse_args(argv)

    lines = read_log(args.log)
    cases = load_cases(args.cases)
    ob_config = extract_ob_config(lines)

    enriched = enrich_cases(lines=lines, base_cases=cases, strategy=args.strategy)
    compute_models(enriched)

    write_csv(args.out_csv, enriched)
    write_jsonl(args.out_jsonl, enriched)
    write_md(args.out_md, cases=enriched, ob_config=ob_config, log_path=args.log, cases_path=args.cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
