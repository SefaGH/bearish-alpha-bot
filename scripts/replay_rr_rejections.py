"""
Offline replay tool for RiskRewardRatioRule rejections.

Goal:
- Extract AdaptiveOversoldBounce (adaptive_ob) signals rejected due to
  "risk/reward ratio ... below dynamic target ..." from a live trading log.
- Compare required R/R between model v1 vs v2 (v2 includes volume/momentum).

Usage example:
  python scripts/replay_rr_rejections.py \
    --log logs/live_trading_20260120_220231_509146.log \
    --strategy adaptive_ob \
    --only-rejected-by-rr \
    --out rr_compare_20260120.csv \
    --md reports/rr_replay_20260120.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TS_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \[(?P<logger>[^\]]+)\] - (?P<level>[A-Z]+) - (?P<msg>.*)$"
)

# Strategy anchors
OB_RR_RE = re.compile(
    r"\[(?P<prefix>ADAPTIVE_OB)/(?P<symbol>[^\]]+)\]\s+\[OB R/R\]\s+Entry=\$(?P<entry>[\d,]+(?:\.\d+)?)"
    r",\s+Stop=\$(?P<stop>[\d,]+(?:\.\d+)?)\s+\([^\)]*\),\s+Target=\$(?P<target>[\d,]+(?:\.\d+)?)"
    r"\s+\([^\)]*\),\s+R/R=(?P<rr>[\d.]+)"
)

OB_MIN_RR_INIT_RE = re.compile(
    r"\[ADAPTIVE_OB\]\s+Minimum R/R Ratio initialized to:\s+(?P<min_rr>[\d.]+)"
)

SIGNAL_INGRESS_RE = re.compile(
    r"\[(?P<prefix>ADAPTIVE_OB)/(?P<symbol>[^\]]+)\]\s+Signal ingress\s+\|\s+side=(?P<side>\w+)\s+\|\s+intent_hint=(?P<intent>\w+)"
)

# Enrichment anchors
SIGNAL_ENRICHED_RE = re.compile(
    r"\[Signal Enriched\]\s+(?P<symbol>.+?):\s+ML=(?P<ml>[\d.]+),\s+RL_agree=(?P<rl_agree>True|False),\s+Regime=(?P<regime>.+?)\s+\((?P<regime_conf>[\d.]+)\),\s+Vol=(?P<vol>[\d.]+)\s+\[(?P<vol_bucket>[^\]]+)\],\s+Mom=(?P<mom>[\d.]+),\s+PPO_RR=(?P<ppo_rr>[\d.]+)"
)

DYNAMIC_RR_RE = re.compile(
    r"\[Dynamic R/R Calc\]\s+Base=(?P<base>[\d.]+)\s+-\s+Relax=(?P<relax>[\d.]+)\s+\+\s+Tight=(?P<tight>[\d.]+)"
    r".*?mult=(?P<regime_mult>[\d.]+),\s+weight=(?P<regime_weight>[\d.]+)\)=(?P<regime_adj>[\d.]+)"
    r".*?=\s+(?P<dynamic>[\d.]+)\s+.*?PPO\((?P<ppo>[\d.]+)\).*?Final=(?P<final>[\d.]+)"
)

# R/R analysis block (multi-line)
RR_ANALYSIS_HDR_RE = re.compile(r"\[R/R Analysis\]\s+(?P<symbol>.+?):\s*$")
RR_PRICES_RE = re.compile(
    r"^\s*Prices:\s+Entry=\$(?P<entry>[\d,]+(?:\.\d+)?),\s+Stop=\$(?P<stop>[\d,]+(?:\.\d+)?)\s+\((?P<risk_pct>[-\d.]+)%\),\s+Target=\$(?P<target>[\d,]+(?:\.\d+)?)\s+\((?P<reward_pct>[-\d.]+)%\)\s*$"
)
RR_VALUES_RE = re.compile(r"^\s*R/R:\s+Actual=(?P<actual>[\d.]+),\s+Required=(?P<required>[\d.]+)\s*$")
RR_INTEL_RE = re.compile(
    r"^\s*Intelligence:\s+ML=(?P<ml>[\d.]+),\s+RL_agree=(?P<rl_agree>True|False),\s+RL_prob=(?P<rl_prob>[\d.]+),\s+Regime=(?P<regime>.+?)\s+\((?P<regime_conf>[\d.]+)\),\s+Vol=(?P<vol>[\d.]+),\s+Mom=(?P<mom>[\d.]+)\s*$"
)

RR_REJECT_RE = re.compile(
    r"\[RiskRewardRatioRule\]\s+REJECTED\s+(?P<symbol>\S+):\s+Risk/reward ratio\s+(?P<actual>[\d.]+)\s+is below dynamic target\s+(?P<required>[\d.]+)\s+\(Risk:\s+(?P<risk_pct>[-\d.]+)%,\s+Reward:\s+(?P<reward_pct>[-\d.]+)%\)"
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def _import_pyyaml():
    """
    This repo contains a top-level `yaml/` package that shadows PyYAML.
    Temporarily remove CWD from sys.path to import site-packages PyYAML.
    """

    import importlib

    cwd = os.getcwd()
    removed: List[str] = []
    for entry in ("", cwd):
        if entry in sys.path:
            sys.path.remove(entry)
            removed.append(entry)
    try:
        module = importlib.import_module("yaml")
        if hasattr(module, "safe_load"):
            return module
        return None
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)


def _load_config_risk_section(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    yaml_mod = _import_pyyaml()
    if yaml_mod is None:
        raise RuntimeError(
            "PyYAML is not available (or import is shadowed). Try running from repo root and ensure PyYAML is installed."
        )

    data = yaml_mod.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}

    risk_section = data.get("risk")
    if isinstance(risk_section, dict):
        return risk_section

    # Allow passing a file that is already the risk section.
    return data


def _compute_rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0
    return reward / risk


@dataclass
class ParsedCase:
    ts: datetime
    symbol: str
    strategy: str
    side: Optional[str] = None
    intent: Optional[str] = None

    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    actual_rr: Optional[float] = None
    required_rr_v1: Optional[float] = None

    ppo_rr_multiplier: Optional[float] = None
    ml_confidence: Optional[float] = None
    rl_is_agree: Optional[bool] = None
    rl_action_prob: Optional[float] = None
    regime_name: Optional[str] = None
    regime_confidence: Optional[float] = None
    regime_weight: Optional[float] = None
    volume_strength: Optional[float] = None
    volume_bucket: Optional[str] = None
    momentum_strength: Optional[float] = None

    # From Dynamic R/R Calc line (v1 runtime)
    dyn_base_rr: Optional[float] = None
    dyn_relax: Optional[float] = None
    dyn_tight: Optional[float] = None
    dyn_regime_mult: Optional[float] = None
    dyn_regime_weight: Optional[float] = None
    dyn_regime_adjustment: Optional[float] = None
    dyn_pre_ppo: Optional[float] = None
    dyn_final: Optional[float] = None

    reject_reason: Optional[str] = None
    match_confidence: str = "time_window"


def _derive_default_jsonl_path(log_path: Path) -> Path:
    m = re.search(r"(\d{8})", log_path.name)
    if m:
        return Path(f"rr_rejected_ob_cases_{m.group(1)}.jsonl")
    return Path("rr_rejected_ob_cases.jsonl")


def _iter_log_records(path: Path) -> Iterable[Tuple[datetime, str, str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            m = TS_LINE_RE.match(line)
            if not m:
                continue
            ts = _parse_ts(m.group("ts"))
            yield ts, m.group("logger"), m.group("level"), m.group("msg")


def extract_rr_rejected_cases(
    log_path: Path,
    strategy: str,
    only_rejected_by_rr: bool,
    window_sec: int = 30,
) -> Tuple[List[ParsedCase], Dict[str, Any]]:
    window = timedelta(seconds=max(1, int(window_sec)))
    strategy = str(strategy or "").strip().lower() or "adaptive_ob"

    # Rolling caches keyed by symbol
    last_ob_rr: Dict[str, Dict[str, Any]] = {}
    last_ingress: Dict[str, Dict[str, Any]] = {}
    last_enriched: Dict[str, Dict[str, Any]] = {}
    last_dynamic_rr: Dict[str, Dict[str, Any]] = {}

    # Per-symbol last R/R analysis block (filled progressively)
    rr_block: Dict[str, Dict[str, Any]] = {}

    # Strategy-level constants (parsed from init logs)
    strategy_min_rr: Dict[str, float] = {}

    cases: List[ParsedCase] = []
    stats: Dict[str, Any] = {
        "log_path": str(log_path),
        "strategy": strategy,
        "reject_lines_seen": 0,
        "cases_emitted": 0,
        "missing_ob_rr_within_window": 0,
    }

    for ts, logger_name, level, msg in _iter_log_records(log_path):
        # 1) Strategy init min rr
        m = OB_MIN_RR_INIT_RE.search(msg)
        if m:
            val = _safe_float(m.group("min_rr"))
            if val is not None:
                strategy_min_rr["adaptive_ob"] = val

        # 2) Strategy OB R/R anchor
        m = OB_RR_RE.search(msg)
        if m:
            sym = m.group("symbol")
            last_ob_rr[sym] = {
                "ts": ts,
                "entry": _safe_float(m.group("entry")),
                "stop": _safe_float(m.group("stop")),
                "target": _safe_float(m.group("target")),
                "rr": _safe_float(m.group("rr")),
            }

        # 3) Signal ingress (side/intent)
        m = SIGNAL_INGRESS_RE.search(msg)
        if m:
            sym = m.group("symbol")
            last_ingress[sym] = {
                "ts": ts,
                "side": m.group("side"),
                "intent": m.group("intent"),
            }

        # 4) Signal Enriched anchor
        m = SIGNAL_ENRICHED_RE.search(msg)
        if m:
            sym = m.group("symbol")
            last_enriched[sym] = {
                "ts": ts,
                "ml_confidence": _safe_float(m.group("ml")),
                "rl_is_agree": m.group("rl_agree") == "True",
                "regime_name": m.group("regime"),
                "regime_confidence": _safe_float(m.group("regime_conf")),
                "volume_strength": _safe_float(m.group("vol")),
                "volume_bucket": m.group("vol_bucket"),
                "momentum_strength": _safe_float(m.group("mom")),
                "ppo_rr_multiplier": _safe_float(m.group("ppo_rr")),
            }

        # 5) Dynamic R/R calc anchor (runtime v1)
        m = DYNAMIC_RR_RE.search(msg)
        if m:
            # Symbol is not in this line; use best-effort: parse later from rr_block or caches.
            # We still cache by a "current symbol" heuristic using the most recent enriched symbol within window.
            inferred_sym = None
            # Pick the most recent enriched record (any symbol) as inference.
            if last_enriched:
                inferred_sym = max(last_enriched.items(), key=lambda item: item[1].get("ts", datetime.min))[0]
            if inferred_sym is None:
                inferred_sym = "UNKNOWN"

            last_dynamic_rr[inferred_sym] = {
                "ts": ts,
                "base": _safe_float(m.group("base")),
                "relax": _safe_float(m.group("relax")),
                "tight": _safe_float(m.group("tight")),
                "regime_mult": _safe_float(m.group("regime_mult")),
                "regime_weight": _safe_float(m.group("regime_weight")),
                "regime_adj": _safe_float(m.group("regime_adj")),
                "pre_ppo": _safe_float(m.group("dynamic")),
                "ppo": _safe_float(m.group("ppo")),
                "final": _safe_float(m.group("final")),
            }

        # 6) R/R Analysis block start
        m = RR_ANALYSIS_HDR_RE.search(msg)
        if m:
            sym = m.group("symbol")
            rr_block[sym] = {"ts": ts}

        # 7) R/R Analysis details
        if rr_block:
            # Heuristic: apply detail lines to the newest active block within small window.
            active_sym = max(rr_block.items(), key=lambda item: item[1].get("ts", datetime.min))[0]
            active = rr_block.get(active_sym) or {}
            if active and (ts - active.get("ts", ts)) <= timedelta(seconds=2):
                m_prices = RR_PRICES_RE.match(msg)
                if m_prices:
                    active.update(
                        {
                            "entry": _safe_float(m_prices.group("entry")),
                            "stop": _safe_float(m_prices.group("stop")),
                            "target": _safe_float(m_prices.group("target")),
                            "risk_pct": _safe_float(m_prices.group("risk_pct")),
                            "reward_pct": _safe_float(m_prices.group("reward_pct")),
                        }
                    )
                m_vals = RR_VALUES_RE.match(msg)
                if m_vals:
                    active.update(
                        {
                            "actual_rr": _safe_float(m_vals.group("actual")),
                            "required_rr": _safe_float(m_vals.group("required")),
                        }
                    )
                m_int = RR_INTEL_RE.match(msg)
                if m_int:
                    active.update(
                        {
                            "ml_confidence": _safe_float(m_int.group("ml")),
                            "rl_is_agree": m_int.group("rl_agree") == "True",
                            "rl_action_prob": _safe_float(m_int.group("rl_prob")),
                            "regime_name": m_int.group("regime"),
                            "regime_confidence": _safe_float(m_int.group("regime_conf")),
                            "volume_strength": _safe_float(m_int.group("vol")),
                            "momentum_strength": _safe_float(m_int.group("mom")),
                        }
                    )
                rr_block[active_sym] = active

        # 8) Reject line
        m = RR_REJECT_RE.search(msg)
        if not m:
            continue

        stats["reject_lines_seen"] += 1
        sym = m.group("symbol")

        if only_rejected_by_rr and "below dynamic target" not in msg:
            continue

        # Strategy filter: require a matching adaptive_ob OB_RR anchor within window.
        ob = last_ob_rr.get(sym)
        if strategy == "adaptive_ob":
            if not ob or (ts - ob.get("ts", ts)) > window:
                stats["missing_ob_rr_within_window"] += 1
                continue

        # Build case
        case = ParsedCase(
            ts=ts,
            symbol=sym,
            strategy=strategy,
        )
        case.reject_reason = msg

        # Ingress hints
        ingress = last_ingress.get(sym)
        if ingress and (ts - ingress.get("ts", ts)) <= window:
            case.side = ingress.get("side")
            case.intent = ingress.get("intent")

        # Prefer prices from the R/R analysis block (what RiskRule actually used)
        block = rr_block.get(sym)
        if block and (ts - block.get("ts", ts)) <= window:
            case.entry = block.get("entry")
            case.stop = block.get("stop")
            case.target = block.get("target")
            case.actual_rr = block.get("actual_rr")
            case.required_rr_v1 = block.get("required_rr")
            case.ml_confidence = block.get("ml_confidence")
            case.rl_is_agree = block.get("rl_is_agree")
            case.rl_action_prob = block.get("rl_action_prob")
            case.regime_name = block.get("regime_name")
            case.regime_confidence = block.get("regime_confidence")
            case.volume_strength = block.get("volume_strength")
            case.momentum_strength = block.get("momentum_strength")

        # Enriched signal (captures volume_bucket + PPO_RR)
        enriched = last_enriched.get(sym)
        if enriched and (ts - enriched.get("ts", ts)) <= window:
            case.volume_bucket = enriched.get("volume_bucket")
            case.ppo_rr_multiplier = enriched.get("ppo_rr_multiplier")
            # Fill missing values if R/R Analysis block was missing
            case.ml_confidence = case.ml_confidence if case.ml_confidence is not None else enriched.get("ml_confidence")
            case.rl_is_agree = case.rl_is_agree if case.rl_is_agree is not None else enriched.get("rl_is_agree")
            case.regime_name = case.regime_name if case.regime_name is not None else enriched.get("regime_name")
            case.regime_confidence = (
                case.regime_confidence if case.regime_confidence is not None else enriched.get("regime_confidence")
            )
            case.volume_strength = (
                case.volume_strength if case.volume_strength is not None else enriched.get("volume_strength")
            )
            case.momentum_strength = (
                case.momentum_strength if case.momentum_strength is not None else enriched.get("momentum_strength")
            )

        # Dynamic RR calc (v1 runtime)
        dyn = last_dynamic_rr.get(sym) or last_dynamic_rr.get("UNKNOWN")
        if dyn and (ts - dyn.get("ts", ts)) <= window:
            case.dyn_base_rr = dyn.get("base")
            case.dyn_relax = dyn.get("relax")
            case.dyn_tight = dyn.get("tight")
            case.dyn_regime_mult = dyn.get("regime_mult")
            case.dyn_regime_weight = dyn.get("regime_weight")
            case.dyn_regime_adjustment = dyn.get("regime_adj")
            case.dyn_pre_ppo = dyn.get("pre_ppo")
            case.dyn_final = dyn.get("final")
            case.ppo_rr_multiplier = case.ppo_rr_multiplier or dyn.get("ppo")
            case.required_rr_v1 = case.required_rr_v1 or dyn.get("final")

        # OB RR line (strategy-generated)
        if ob and (ts - ob.get("ts", ts)) <= window:
            case.entry = case.entry if case.entry is not None else ob.get("entry")
            case.stop = case.stop if case.stop is not None else ob.get("stop")
            case.target = case.target if case.target is not None else ob.get("target")
            case.actual_rr = case.actual_rr if case.actual_rr is not None else ob.get("rr")

        # Parse actual/required from reject line as a last resort
        case.actual_rr = case.actual_rr if case.actual_rr is not None else _safe_float(m.group("actual"))
        case.required_rr_v1 = case.required_rr_v1 if case.required_rr_v1 is not None else _safe_float(m.group("required"))

        # Compute actual RR from prices (most reliable)
        if case.entry and case.stop and case.target:
            case.actual_rr = _compute_rr(case.entry, case.stop, case.target)

        # Match confidence
        exact = True
        for bundle in (ob, ingress, enriched, block, dyn):
            if not bundle:
                exact = False
                continue
            ref_ts = bundle.get("ts") if isinstance(bundle, dict) else None
            if not isinstance(ref_ts, datetime) or abs((ts - ref_ts).total_seconds()) > 1.0:
                exact = False
        case.match_confidence = "exact" if exact else "time_window"

        cases.append(case)
        stats["cases_emitted"] += 1

    # attach strategy floor
    if "adaptive_ob" in strategy_min_rr:
        stats["strategy_min_rr"] = strategy_min_rr["adaptive_ob"]

    return cases, stats


def _case_to_json(case: ParsedCase) -> Dict[str, Any]:
    return {
        "ts": case.ts.isoformat(sep=" "),
        "symbol": case.symbol,
        "strategy": case.strategy,
        "side": case.side,
        "intent": case.intent,
        "entry": case.entry,
        "stop": case.stop,
        "target": case.target,
        "actual_rr": case.actual_rr,
        "required_rr_v1": case.required_rr_v1,
        "ppo_rr_multiplier": case.ppo_rr_multiplier,
        "ml_confidence": case.ml_confidence,
        "rl_is_agree": case.rl_is_agree,
        "rl_action_prob": case.rl_action_prob,
        "regime_name": case.regime_name,
        "regime_confidence": case.regime_confidence,
        "regime_weight": case.regime_weight,
        "volume_strength": case.volume_strength,
        "volume_bucket": case.volume_bucket,
        "momentum_strength": case.momentum_strength,
        "dyn_base_rr": case.dyn_base_rr,
        "dyn_relax": case.dyn_relax,
        "dyn_tight": case.dyn_tight,
        "dyn_regime_mult": case.dyn_regime_mult,
        "dyn_regime_weight": case.dyn_regime_weight,
        "dyn_regime_adjustment": case.dyn_regime_adjustment,
        "dyn_pre_ppo": case.dyn_pre_ppo,
        "dyn_final": case.dyn_final,
        "reject_reason": case.reject_reason,
        "match_confidence": case.match_confidence,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay RR rejections and compare v1 vs v2 required RR.")
    parser.add_argument("--log", required=True, help="Path to live trading log file")
    parser.add_argument("--strategy", default="adaptive_ob", help="Strategy name filter (default: adaptive_ob)")
    parser.add_argument("--only-rejected-by-rr", action="store_true", help="Only include RR-below-target rejects")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--md", required=True, help="Output markdown report path")
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Output JSONL path (default: rr_rejected_ob_cases_<YYYYMMDD>.jsonl derived from log filename)",
    )
    parser.add_argument(
        "--config",
        default="config/config.example.yaml",
        help="Config YAML to load dynamic RR params from (default: config/config.example.yaml)",
    )
    parser.add_argument("--window-sec", type=int, default=30, help="Correlation window in seconds (default: 30)")
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    out_csv = Path(args.out)
    out_md = Path(args.md)
    jsonl_path = Path(args.jsonl) if args.jsonl else _derive_default_jsonl_path(log_path)
    config_path = Path(args.config)

    cases, stats = extract_rr_rejected_cases(
        log_path=log_path,
        strategy=args.strategy,
        only_rejected_by_rr=bool(args.only_rejected_by_rr),
        window_sec=args.window_sec,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Load risk config + build v1/v2 rules
    # NOTE: keep imports local so this script can still run if core deps are missing.
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))
    from config.risk_config import RiskConfiguration  # noqa: E402
    from core.risk_rules import RiskRewardRatioRule  # noqa: E402

    risk_section = _load_config_risk_section(config_path)
    # Ensure starting capital exists for RiskConfiguration.
    if "equity_usd" not in risk_section or not _safe_float(risk_section.get("equity_usd")):
        risk_section = dict(risk_section)
        risk_section["equity_usd"] = 1000.0

    rr_dyn = risk_section.get("rr_dynamic") if isinstance(risk_section.get("rr_dynamic"), dict) else {}
    v1_risk = dict(risk_section)
    v2_risk = dict(risk_section)
    v1_risk["rr_dynamic"] = dict(rr_dyn or {})
    v2_risk["rr_dynamic"] = dict(rr_dyn or {})
    v1_risk["rr_dynamic"]["model_version"] = "v1"
    v2_risk["rr_dynamic"]["model_version"] = "v2"

    cfg_v1 = RiskConfiguration(v1_risk)
    cfg_v2 = RiskConfiguration(v2_risk)
    rule_v1 = RiskRewardRatioRule(config=cfg_v1)
    rule_v2 = RiskRewardRatioRule(config=cfg_v2)

    # Strategy floor from log (if present)
    strategy_floor = _safe_float(stats.get("strategy_min_rr"))

    rows: List[Dict[str, Any]] = []
    for case in cases:
        if case.entry is None or case.stop is None or case.target is None:
            continue

        signal: Dict[str, Any] = {
            "symbol": case.symbol,
            "strategy_name": case.strategy,
            "side": case.side or "long",
            "entry": float(case.entry),
            "stop": float(case.stop),
            "target": float(case.target),
            "ml_confidence": float(case.ml_confidence) if case.ml_confidence is not None else None,
            "rl_is_agree": bool(case.rl_is_agree) if case.rl_is_agree is not None else False,
            "rl_action_prob": float(case.rl_action_prob) if case.rl_action_prob is not None else None,
            "regime_name": case.regime_name or "neutral",
            "regime_confidence": float(case.regime_confidence) if case.regime_confidence is not None else None,
            "regime_weight": float(case.regime_weight) if case.regime_weight is not None else None,
            "volume_strength": float(case.volume_strength) if case.volume_strength is not None else None,
            "momentum_strength": float(case.momentum_strength) if case.momentum_strength is not None else None,
            "ppo_rr_multiplier": float(case.ppo_rr_multiplier) if case.ppo_rr_multiplier is not None else 1.0,
        }
        # Drop Nones so rule fallbacks behave like runtime.
        signal = {k: v for k, v in signal.items() if v is not None}
        if strategy_floor is not None:
            signal["strategy_min_rr"] = float(strategy_floor)

        required_v1_calc = rule_v1._calculate_dynamic_target(dict(signal))
        required_v2 = rule_v2._calculate_dynamic_target(dict(signal))

        actual_rr = _compute_rr(float(case.entry), float(case.stop), float(case.target))
        required_v1_log = case.required_rr_v1 if case.required_rr_v1 is not None else required_v1_calc

        accepted_v1 = actual_rr >= float(required_v1_log)
        accepted_v2 = actual_rr >= float(required_v2)

        rows.append(
            {
                "ts": case.ts.isoformat(sep=" "),
                "symbol": case.symbol,
                "strategy": case.strategy,
                "side": case.side,
                "entry": float(case.entry),
                "stop": float(case.stop),
                "target": float(case.target),
                "actual_rr": round(actual_rr, 4),
                "required_rr_v1": round(float(required_v1_log), 4),
                "accepted_v1": accepted_v1,
                "required_rr_v2": round(float(required_v2), 4),
                "accepted_v2": accepted_v2,
                "delta_required_v2_minus_v1": round(float(required_v2) - float(required_v1_log), 4),
                "ppo_rr_multiplier": case.ppo_rr_multiplier,
                "volume_strength": case.volume_strength,
                "volume_bucket": case.volume_bucket,
                "momentum_strength": case.momentum_strength,
                "ml_confidence": case.ml_confidence,
                "rl_is_agree": case.rl_is_agree,
                "rl_action_prob": case.rl_action_prob,
                "regime_confidence": case.regime_confidence,
                "match_confidence": case.match_confidence,
            }
        )

    # Write JSONL
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(_case_to_json(case), ensure_ascii=False) + "\n")

    # Write CSV
    csv_fields = [
        "ts",
        "symbol",
        "strategy",
        "side",
        "entry",
        "stop",
        "target",
        "actual_rr",
        "required_rr_v1",
        "accepted_v1",
        "required_rr_v2",
        "accepted_v2",
        "delta_required_v2_minus_v1",
        "ppo_rr_multiplier",
        "volume_strength",
        "volume_bucket",
        "momentum_strength",
        "ml_confidence",
        "rl_is_agree",
        "rl_action_prob",
        "regime_confidence",
        "match_confidence",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    # Summaries
    v1_reject = sum(1 for r in rows if not r["accepted_v1"])
    v2_reject = sum(1 for r in rows if not r["accepted_v2"])
    v1_reject_v2_accept = sum(1 for r in rows if (not r["accepted_v1"]) and r["accepted_v2"])
    v1_accept_v2_reject = sum(1 for r in rows if r["accepted_v1"] and (not r["accepted_v2"]))

    top_delta = sorted(rows, key=lambda r: r["delta_required_v2_minus_v1"])[:10]

    # Write markdown report
    lines: List[str] = []
    lines.append("# RR Replay Report — 20260120")
    lines.append("")
    lines.append(f"- Log: `{log_path}`")
    lines.append(f"- Strategy filter: `{args.strategy}`")
    lines.append(f"- JSONL: `{jsonl_path}`")
    lines.append(f"- CSV: `{out_csv}`")
    lines.append(f"- Config used for v2: `{config_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases found (RR rejects, filtered): **{len(cases)}**")
    lines.append(f"- Cases replayed (with prices): **{len(rows)}**")
    lines.append(f"- v1 rejects: **{v1_reject}**")
    lines.append(f"- v2 rejects: **{v2_reject}**")
    lines.append(f"- v1 reject → v2 accept: **{v1_reject_v2_accept}**")
    lines.append(f"- v1 accept → v2 reject: **{v1_accept_v2_reject}**")
    lines.append("")
    lines.append("## Top 10 Δ(required_rr_v2 - required_rr_v1) (most negative)")
    lines.append("")
    if not top_delta:
        lines.append("_No rows to report._")
    else:
        lines.append(
            "| ts | symbol | actual_rr | req_v1 | req_v2 | delta | ppo_rr | vol | mom | match |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in top_delta:
            lines.append(
                f"| {r['ts']} | {r['symbol']} | {r['actual_rr']:.4f} | {r['required_rr_v1']:.4f} | "
                f"{r['required_rr_v2']:.4f} | {r['delta_required_v2_minus_v1']:.4f} | {r.get('ppo_rr_multiplier')} | "
                f"{r.get('volume_strength')} | {r.get('momentum_strength')} | {r.get('match_confidence')} |"
            )

    lines.append("")
    lines.append("## Extraction Diagnostics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

