"""\
MR Low-Volume Gating Analyzer

Extracts Mean Reversion (MR) signal cycles and volume gating outcomes from live trading logs.

Primary focus:
- `waiting_room_drop` with reason_code == `volume.low_vol_tight_stop_far`
- MR signals and their computed/observed stop distance (bps)
- Join each case with nearest:
  - `mr_controller_decision` (vwap_std, bands, adx, atr, z)
  - `volume_decision_check` (bucket, volume_strength)

Outputs:
- CSV of per-case rows (cases + signals)
- Summary stats printed to stdout

Usage:
  C:/Users/sefaa/bearish-alpha-bot/.venv/Scripts/python.exe scripts/analyze_mr_low_vol_gating.py \
    --log-file logs/live_trading_20260118_202844_781510.log

Optional:
  --out-csv reports/mr_low_vol_gating_cases.csv
  --join-window-ms 5000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TS_LINE_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ")


def _parse_line_ts_ms(line: str) -> Optional[int]:
    m = TS_LINE_RE.match(line)
    if not m:
        return None
    # Log timestamps are emitted in UTC in this repo (matches embedded `ts_ms` fields).
    try:
        dt = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _extract_json_from_line(line: str) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    brace_idx = line.find("{")
    if brace_idx == -1:
        return None
    candidate = line[brace_idx:].strip()

    # JSON first
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Python dict string fallback (safe subset)
    try:
        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _parse_ts_utc_ms(value: Any) -> Optional[int]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def _bps(x: float) -> float:
    return float(x) * 10000.0


def _bps_delta(px: Optional[float], ref: Optional[float]) -> Optional[float]:
    if px is None or ref is None:
        return None
    if not math.isfinite(px) or not math.isfinite(ref) or px == 0:
        return None
    return (px - ref) / px * 10000.0


def _stop_price_from_bps(*, side: str, entry_price: float, stop_bps: float) -> Optional[float]:
    if not (math.isfinite(float(entry_price)) and float(entry_price) > 0):
        return None
    if not (math.isfinite(float(stop_bps)) and float(stop_bps) >= 0):
        return None
    frac = float(stop_bps) / 10000.0
    if side == "long":
        return float(entry_price) * (1.0 - frac)
    else:
        return float(entry_price) * (1.0 + frac)


def _rr_ratio(*, reward_bps: Optional[float], risk_bps: Optional[float]) -> Optional[float]:
    if reward_bps is None or risk_bps is None:
        return None
    if not (math.isfinite(float(reward_bps)) and math.isfinite(float(risk_bps))):
        return None
    if float(risk_bps) <= 0:
        return None
    return float(reward_bps) / float(risk_bps)


def _rr_ratio_net(*, reward_bps: Optional[float], risk_bps: Optional[float], cost_bps: float) -> Optional[float]:
    """Very simple net-RR approximation using a total round-trip cost in bps.

    - Net reward bps ~= reward_bps - cost_bps
    - Net risk bps   ~= risk_bps   + cost_bps
    """
    if reward_bps is None or risk_bps is None:
        return None
    if not (math.isfinite(float(reward_bps)) and math.isfinite(float(risk_bps)) and math.isfinite(float(cost_bps))):
        return None
    net_reward = float(reward_bps) - float(cost_bps)
    net_risk = float(risk_bps) + float(cost_bps)
    if net_risk <= 0:
        return None
    return net_reward / net_risk


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if any(not math.isfinite(float(x)) for x in xs) or any(not math.isfinite(float(y)) for y in ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = float(x) - mx
        dy = float(y) - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    if dx2 <= 0 or dy2 <= 0:
        return None
    return num / math.sqrt(dx2 * dy2)


@dataclass(frozen=True)
class MRControllerSnapshot:
    ts_ms: int  # emission time
    symbol: str
    px: Optional[float]
    vwap: Optional[float]
    vwap_std: Optional[float]
    adx: Optional[float]
    atr: Optional[float]
    atr_pct: Optional[float]
    z: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    band_multiplier: Optional[float]


@dataclass(frozen=True)
class VolumeSnapshot:
    ts_ms: int
    symbol: str
    timeframe: Optional[str]
    bucket: Optional[str]
    strength: Optional[float]


@dataclass(frozen=True)
class GateEvent:
    ts_ms: int
    event: str
    symbol: str
    strategy: Optional[str]
    side: Optional[str]
    reason_code: Optional[str]
    drop_reason: Optional[str]
    signal_id: Optional[str]
    dedupe_key: Optional[str]
    px_used: Optional[float]
    px_source: Optional[str]
    gate_threshold_bps: Optional[float]
    gate_margin_bps: Optional[float]
    stop_distance: Optional[float]
    stop_pct: Optional[float]


@dataclass(frozen=True)
class MRSignalLine:
    ts_ms: int
    symbol: str
    action: str  # SIGNAL/HOLD


@dataclass(frozen=True)
class MRIngressLine:
    ts_ms: int
    symbol: str
    side: Optional[str]
    reason: Optional[str]


def _nearest_by_ts(items: List[Any], ts_ms: int, *, window_ms: int) -> Optional[Any]:
    best = None
    best_dt = None
    for it in items:
        it_ts = getattr(it, "ts_ms", None)
        if it_ts is None:
            continue
        dt = abs(int(it_ts) - int(ts_ms))
        if dt <= int(window_ms) and (best_dt is None or dt < best_dt):
            best = it
            best_dt = dt
    return best


def _last_before_ts(items: List[Any], ts_ms: int, *, max_age_ms: int) -> Optional[Any]:
    """Return the latest item with `item.ts_ms <= ts_ms` within max_age_ms."""
    best = None
    best_ts = None
    for it in items:
        it_ts = getattr(it, "ts_ms", None)
        if it_ts is None:
            continue
        it_ts = int(it_ts)
        if it_ts > int(ts_ms):
            continue
        age = int(ts_ms) - it_ts
        if age < 0 or age > int(max_age_ms):
            continue
        if best_ts is None or it_ts > best_ts:
            best = it
            best_ts = it_ts
    return best


def _load_mr_config_from_log(log_file: Path) -> Dict[str, Any]:
    # Looks for: "- MR Config: {...}"
    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "- MR Config:" not in line:
                    continue
                idx = line.find("- MR Config:")
                if idx == -1:
                    continue
                payload = line[idx + len("- MR Config:"):].strip()
                try:
                    obj = ast.literal_eval(payload)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return {}
    except Exception:
        return {}
    return {}


def _compute_stop_from_snapshot(
    *,
    side: str,
    price: float,
    lower: Optional[float],
    upper: Optional[float],
    vwap_std: Optional[float],
    band_multiplier: Optional[float],
    stop_loss_std_delta: float,
    atr: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:
    """Replicates the stop logic from `src/strategies/mean_reversion.py` (high-level).

    Returns (stop_loss_price, stop_source_guess).
    """
    effective_vwap_std = vwap_std
    if effective_vwap_std is None or not math.isfinite(float(effective_vwap_std)) or float(effective_vwap_std) <= 0:
        # Derive from bands if possible
        try:
            if (
                lower is not None
                and upper is not None
                and band_multiplier is not None
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and math.isfinite(float(band_multiplier))
                and float(band_multiplier) > 0
                and float(upper) > float(lower)
            ):
                effective_vwap_std = (float(upper) - float(lower)) / (2.0 * float(band_multiplier))
        except Exception:
            effective_vwap_std = None

    stop_loss_price = None
    stop_source = None

    if (
        effective_vwap_std is not None
        and math.isfinite(float(effective_vwap_std))
        and float(effective_vwap_std) > 0
        and math.isfinite(float(stop_loss_std_delta))
        and float(stop_loss_std_delta) > 0
    ):
        delta = float(stop_loss_std_delta)
        if side == "long":
            # Long stop anchored to lower band in the strategy.
            if lower is not None and math.isfinite(float(lower)):
                stop_candidate = float(lower) - delta * float(effective_vwap_std)
            else:
                stop_candidate = float(price) - delta * float(effective_vwap_std)
            if stop_candidate >= float(price):
                stop_candidate = float(price) - delta * float(effective_vwap_std)
            stop_loss_price = stop_candidate
        else:
            if upper is not None and math.isfinite(float(upper)):
                stop_candidate = float(upper) + delta * float(effective_vwap_std)
            else:
                stop_candidate = float(price) + delta * float(effective_vwap_std)
            if stop_candidate <= float(price):
                stop_candidate = float(price) + delta * float(effective_vwap_std)
            stop_loss_price = stop_candidate

        if vwap_std is not None and math.isfinite(float(vwap_std)) and float(vwap_std) > 0:
            stop_source = "std_based"
        else:
            stop_source = "std_based_derived"

    # Fallback ATR-based stop
    if (stop_loss_price is None or not math.isfinite(float(stop_loss_price))) and atr is not None and math.isfinite(float(atr)) and float(atr) > 0:
        if side == "long":
            stop_loss_price = float(price) - float(atr) * 1.5
        else:
            stop_loss_price = float(price) + float(atr) * 1.5
        stop_source = "atr_fallback"

    return stop_loss_price, stop_source


def _band_width_bps(lower: Optional[float], upper: Optional[float], ref: Optional[float]) -> Optional[float]:
    if lower is None or upper is None or ref is None:
        return None
    if not (math.isfinite(float(lower)) and math.isfinite(float(upper)) and math.isfinite(float(ref)) and float(ref) != 0):
        return None
    return (float(upper) - float(lower)) / float(ref) * 10000.0


def _stop_price_from_gate(
    *,
    side: str,
    entry_price: float,
    stop_pct: Optional[float],
    stop_distance: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute stop_price using gate fields.

    Returns (stop_price, stop_distance_abs).
    """
    if not math.isfinite(float(entry_price)) or float(entry_price) <= 0:
        return None, None

    dist = None
    if stop_pct is not None and math.isfinite(float(stop_pct)) and float(stop_pct) > 0:
        dist = float(entry_price) * float(stop_pct)
    elif stop_distance is not None and math.isfinite(float(stop_distance)) and float(stop_distance) > 0:
        dist = float(stop_distance)

    if dist is None:
        return None, None

    if side == "long":
        return float(entry_price) - float(dist), float(dist)
    else:
        return float(entry_price) + float(dist), float(dist)


def _dist_outside_bps(side: str, price: Optional[float], lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    if side == "long":
        if lower is None:
            return None
        # how far below lower
        return (float(lower) - float(price)) / float(price) * 10000.0
    else:
        if upper is None:
            return None
        return (float(price) - float(upper)) / float(price) * 10000.0


def parse_log(
    log_file: Path,
) -> Tuple[List[MRControllerSnapshot], List[VolumeSnapshot], List[GateEvent], List[MRSignalLine], List[MRIngressLine]]:
    controllers: List[MRControllerSnapshot] = []
    volumes: List[VolumeSnapshot] = []
    gates: List[GateEvent] = []
    mr_signals: List[MRSignalLine] = []
    ingress: List[MRIngressLine] = []

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "{" not in line and "[MeanReversion] Cycle complete" not in line and "Signal ingress" not in line:
                continue

            line_ts_ms = _parse_line_ts_ms(line) or 0

            if "mr_controller_decision" in line and "{" in line:
                payload = _extract_json_from_line(line)
                if isinstance(payload, dict) and payload.get("event") == "mr_controller_decision":
                    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
                    derived = payload.get("derived") if isinstance(payload.get("derived"), dict) else {}
                    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

                    controllers.append(
                        MRControllerSnapshot(
                            ts_ms=int(line_ts_ms),
                            symbol=str(payload.get("symbol") or ""),
                            px=_as_float(inputs.get("px")),
                            vwap=_as_float(inputs.get("vwap")),
                            vwap_std=_as_float(inputs.get("vwap_std")),
                            adx=_as_float(inputs.get("adx")),
                            atr=_as_float(inputs.get("atr")),
                            atr_pct=_as_float(inputs.get("atr_pct")),
                            z=_as_float(derived.get("z")),
                            lower=_as_float(derived.get("lower")),
                            upper=_as_float(derived.get("upper")),
                            band_multiplier=_as_float(params.get("band_multiplier_new")),
                        )
                    )
                continue

            if "volume_decision_check" in line and "{" in line:
                payload = _extract_json_from_line(line)
                if isinstance(payload, dict) and payload.get("event") == "volume_decision_check":
                    ts_ms = _parse_ts_utc_ms(payload.get("timestamp")) or line_ts_ms
                    volumes.append(
                        VolumeSnapshot(
                            ts_ms=int(ts_ms),
                            symbol=str(payload.get("symbol") or ""),
                            timeframe=str(payload.get("timeframe") or "") or None,
                            bucket=str(payload.get("volume_bucket") or "") or None,
                            strength=_as_float(payload.get("volume_strength")),
                        )
                    )
                continue

            if ("waiting_room_drop" in line or "waiting_room_add" in line) and "{" in line:
                payload = _extract_json_from_line(line)
                if isinstance(payload, dict) and str(payload.get("event") or "").startswith("waiting_room_"):
                    if str(payload.get("strategy") or payload.get("strategy_name") or "") != "mean_reversion":
                        continue
                    ts_ms = _parse_ts_utc_ms(payload.get("ts_ms")) or line_ts_ms
                    gates.append(
                        GateEvent(
                            ts_ms=int(ts_ms),
                            event=str(payload.get("event") or ""),
                            symbol=str(payload.get("symbol") or ""),
                            strategy=str(payload.get("strategy") or payload.get("strategy_name") or "") or None,
                            side=str(payload.get("side") or "") or None,
                            reason_code=str(payload.get("reason_code") or "") or None,
                            drop_reason=str(payload.get("drop_reason") or "") or None,
                            signal_id=str(payload.get("signal_id") or payload.get("incoming_signal_id") or "") or None,
                            dedupe_key=str(payload.get("dedupe_key") or "") or None,
                            px_used=_as_float(payload.get("px_used")),
                            px_source=str(payload.get("px_source") or "") or None,
                            gate_threshold_bps=_as_float(payload.get("gate_threshold_bps")),
                            gate_margin_bps=_as_float(payload.get("gate_margin_bps")),
                            stop_distance=_as_float(payload.get("stop_distance")),
                            stop_pct=_as_float(payload.get("stop_pct")),
                        )
                    )
                continue

            if "[MeanReversion] Cycle complete" in line:
                # Example: "[MeanReversion] Cycle complete for BTC/USDT:USDT. Action: SIGNAL"
                try:
                    symbol = ""
                    if "for " in line:
                        symbol = line.split("for ", 1)[1].split(".", 1)[0].strip()
                    action = "UNKNOWN"
                    if "Action:" in line:
                        action = line.split("Action:", 1)[1].strip().split()[0].strip()
                    mr_signals.append(MRSignalLine(ts_ms=int(line_ts_ms), symbol=symbol, action=action))
                except Exception:
                    pass
                continue

            if "Signal ingress" in line and "[MEAN_REVERSION/" in line:
                # Example: "Signal ingress | side=long | intent_hint=entry | reason=..."
                try:
                    # Extract symbol within [MEAN_REVERSION/SYMBOL]
                    sym = ""
                    m1 = re.search(r"\[MEAN_REVERSION/(?P<sym>[^\]]+)\]", line)
                    if m1:
                        sym = m1.group("sym")
                    side = None
                    m2 = re.search(r"side=(long|short)", line)
                    if m2:
                        side = m2.group(1)
                    reason = None
                    if "reason=" in line:
                        reason = line.split("reason=", 1)[1].strip()
                    ingress.append(MRIngressLine(ts_ms=int(line_ts_ms), symbol=sym, side=side, reason=reason))
                except Exception:
                    pass
                continue

    return controllers, volumes, gates, mr_signals, ingress


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)

    def q(p: float) -> float:
        if not vals:
            return float("nan")
        k = (len(vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vals[int(k)])
        return float(vals[f] * (c - k) + vals[c] * (k - f))

    return {
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", required=True, type=Path)
    ap.add_argument("--out-csv", default=None, type=Path)
    ap.add_argument("--out-md", default=None, type=Path)
    ap.add_argument("--top-n", default=33, type=int)
    ap.add_argument("--join-window-ms", default=5000, type=int)
    ap.add_argument("--carry-forward-ms", default=15 * 60 * 1000, type=int)
    ap.add_argument("--gate-threshold-bps", default=15.0, type=float)
    ap.add_argument("--cost-bps", default=0.0, type=float, help="Approx total round-trip costs (fees+slippage) in bps")
    ap.add_argument("--reward-consistent-rr", default=2.0, type=float, help="Target RR for reward-consistent stop")
    ap.add_argument("--reward-consistent-min-stop-bps", default=0.0, type=float)
    ap.add_argument("--reward-consistent-max-stop-bps", default=200.0, type=float)
    ap.add_argument(
        "--entry-anchored-k",
        default=None,
        type=float,
        help="Entry-anchored stop multiplier k for stop_bps ~= k * std_bps (defaults to MR stop_loss_std_delta)",
    )
    args = ap.parse_args()

    log_file: Path = args.log_file
    if not log_file.exists():
        raise SystemExit(f"Log file not found: {log_file}")

    mr_cfg = _load_mr_config_from_log(log_file)
    stop_loss_std_delta = float(mr_cfg.get("stop_loss_std_delta", 1.0) or 1.0)
    entry_anchored_k = float(args.entry_anchored_k) if args.entry_anchored_k is not None else float(stop_loss_std_delta)

    controllers, volumes, gates, mr_signals, ingress = parse_log(log_file)

    # Index by symbol for faster joins
    controllers_by_sym: Dict[str, List[MRControllerSnapshot]] = {}
    for c in controllers:
        controllers_by_sym.setdefault(c.symbol, []).append(c)

    for sym in list(controllers_by_sym.keys()):
        controllers_by_sym[sym].sort(key=lambda x: x.ts_ms)

    volumes_by_sym: Dict[str, List[VolumeSnapshot]] = {}
    for v in volumes:
        volumes_by_sym.setdefault(v.symbol, []).append(v)

    for sym in list(volumes_by_sym.keys()):
        volumes_by_sym[sym].sort(key=lambda x: x.ts_ms)

    ingress_by_sym: Dict[str, List[MRIngressLine]] = {}
    for i in ingress:
        ingress_by_sym.setdefault(i.symbol, []).append(i)

    rows: List[Dict[str, Any]] = []

    def add_row(kind: str, ts_ms: int, symbol: str, side: Optional[str], action: Optional[str], gate: Optional[GateEvent]) -> None:
        ctrl = _nearest_by_ts(controllers_by_sym.get(symbol, []), ts_ms, window_ms=int(args.join_window_ms))
        if ctrl is None:
            ctrl = _last_before_ts(
                controllers_by_sym.get(symbol, []),
                ts_ms,
                max_age_ms=int(args.carry_forward_ms),
            )
        vol = _nearest_by_ts(volumes_by_sym.get(symbol, []), ts_ms, window_ms=int(args.join_window_ms))
        if vol is None:
            vol = _last_before_ts(
                volumes_by_sym.get(symbol, []),
                ts_ms,
                max_age_ms=int(args.carry_forward_ms),
            )

        price = None
        if gate and gate.px_used is not None:
            price = float(gate.px_used)
        elif ctrl and ctrl.px is not None:
            price = float(ctrl.px)

        # Side inference if missing
        eff_side = side
        if eff_side is None and ctrl and price is not None and ctrl.lower is not None and ctrl.upper is not None:
            try:
                if float(price) < float(ctrl.lower):
                    eff_side = "long"
                elif float(price) > float(ctrl.upper):
                    eff_side = "short"
            except Exception:
                eff_side = eff_side

        # Stop metrics: prefer observed stop_pct from gate events.
        stop_pct = gate.stop_pct if gate else None
        stop_pct_bps = _bps(float(stop_pct)) if stop_pct is not None else None

        stop_price_observed = None
        stop_distance_observed = None
        if price is not None and eff_side in ("long", "short") and gate is not None:
            stop_price_observed, stop_distance_observed = _stop_price_from_gate(
                side=eff_side,
                entry_price=float(price),
                stop_pct=gate.stop_pct,
                stop_distance=gate.stop_distance,
            )
        stop_distance_bps_observed = None
        if price is not None and stop_distance_observed is not None and float(price) > 0:
            stop_distance_bps_observed = float(stop_distance_observed) / float(price) * 10000.0

        stop_source_guess = None
        stop_price_expected = None
        expected_stop_pct_bps = None
        if price is not None and eff_side in ("long", "short") and ctrl is not None:
            stop_price_expected, stop_source_guess = _compute_stop_from_snapshot(
                side=eff_side,
                price=float(price),
                lower=ctrl.lower,
                upper=ctrl.upper,
                vwap_std=ctrl.vwap_std,
                band_multiplier=ctrl.band_multiplier,
                stop_loss_std_delta=stop_loss_std_delta,
                atr=ctrl.atr,
            )
            if stop_price_expected is not None and math.isfinite(float(stop_price_expected)):
                try:
                    expected_stop_dist = abs(float(price) - float(stop_price_expected))
                    if expected_stop_dist >= 0 and float(price) > 0:
                        expected_stop_pct_bps = expected_stop_dist / float(price) * 10000.0
                except Exception:
                    expected_stop_pct_bps = None

        # If observed missing, fall back to expected
        stop_pct_bps_final = stop_pct_bps if stop_pct_bps is not None else expected_stop_pct_bps

        gate_threshold_bps = (gate.gate_threshold_bps if (gate and gate.gate_threshold_bps is not None) else float(args.gate_threshold_bps))
        gate_margin_bps_expected = None
        if stop_pct_bps_final is not None:
            gate_margin_bps_expected = float(gate_threshold_bps) - float(stop_pct_bps_final)

        band_width_bps = None
        if ctrl is not None:
            band_width_bps = _band_width_bps(ctrl.lower, ctrl.upper, ctrl.vwap if ctrl.vwap is not None else price)

        dist_outside_bps = None
        if ctrl is not None and price is not None and eff_side in ("long", "short"):
            dist_outside_bps = _dist_outside_bps(eff_side, price, ctrl.lower, ctrl.upper)

        # Target guess: MR TP is typically VWAP.
        target_price = ctrl.vwap if (ctrl is not None and ctrl.vwap is not None) else None
        target_bps = None
        if price is not None and target_price is not None and float(price) > 0:
            try:
                if eff_side == "long":
                    target_bps = (float(target_price) - float(price)) / float(price) * 10000.0
                elif eff_side == "short":
                    target_bps = (float(price) - float(target_price)) / float(price) * 10000.0
            except Exception:
                target_bps = None

        rr_ratio_current = _rr_ratio(reward_bps=target_bps, risk_bps=stop_pct_bps_final)
        rr_ratio_net_current = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_pct_bps_final, cost_bps=float(args.cost_bps))

        # Decision-tree derived metrics
        std_bps = None
        atr_bps = None
        if ctrl is not None and price is not None and float(price) > 0:
            if ctrl.vwap_std is not None and math.isfinite(float(ctrl.vwap_std)):
                std_bps = float(ctrl.vwap_std) / float(price) * 10000.0
            if ctrl.atr is not None and math.isfinite(float(ctrl.atr)):
                atr_bps = float(ctrl.atr) / float(price) * 10000.0

        k_implied = None
        if std_bps is not None and stop_pct_bps_final is not None:
            try:
                if float(std_bps) > 0:
                    k_implied = float(stop_pct_bps_final) / float(std_bps)
            except Exception:
                k_implied = None

        reward_bps = target_bps
        cost_bps_assumed = float(args.cost_bps)
        net_rr_current = rr_ratio_net_current

        # Counterfactual 1: force stop=gate_threshold_bps (15bps by default)
        stop_bps_15 = float(gate_threshold_bps)
        stop_price_15 = None
        if price is not None and eff_side in ("long", "short"):
            stop_price_15 = _stop_price_from_bps(side=eff_side, entry_price=float(price), stop_bps=stop_bps_15)
        rr_ratio_stop15 = _rr_ratio(reward_bps=target_bps, risk_bps=stop_bps_15)
        rr_ratio_net_stop15 = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_bps_15, cost_bps=float(args.cost_bps))

        # Counterfactual 2: reward-consistent stop: stop_bps ~= target_bps / rr_target (clamped)
        rr_target = float(args.reward_consistent_rr)
        rc_min = float(args.reward_consistent_min_stop_bps)
        rc_max = float(args.reward_consistent_max_stop_bps)
        stop_bps_rc_raw = None
        if target_bps is not None and math.isfinite(float(target_bps)) and rr_target > 0:
            stop_bps_rc_raw = float(target_bps) / rr_target
            stop_bps_rc_raw = max(rc_min, min(rc_max, stop_bps_rc_raw))

        # Also show a "gate-pass" variant that ensures >= threshold
        stop_bps_rc_gate = None
        if stop_bps_rc_raw is not None:
            stop_bps_rc_gate = max(float(gate_threshold_bps), float(stop_bps_rc_raw))

        stop_price_rc_raw = None
        stop_price_rc_gate = None
        if price is not None and eff_side in ("long", "short"):
            if stop_bps_rc_raw is not None:
                stop_price_rc_raw = _stop_price_from_bps(side=eff_side, entry_price=float(price), stop_bps=float(stop_bps_rc_raw))
            if stop_bps_rc_gate is not None:
                stop_price_rc_gate = _stop_price_from_bps(side=eff_side, entry_price=float(price), stop_bps=float(stop_bps_rc_gate))

        rr_ratio_rc_raw = _rr_ratio(reward_bps=target_bps, risk_bps=stop_bps_rc_raw)
        rr_ratio_net_rc_raw = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_bps_rc_raw, cost_bps=float(args.cost_bps))
        rr_ratio_rc_gate = _rr_ratio(reward_bps=target_bps, risk_bps=stop_bps_rc_gate)
        rr_ratio_net_rc_gate = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_bps_rc_gate, cost_bps=float(args.cost_bps))

        # "Tradeable" here means: would pass the tight-stop gate (stop >= threshold) AND has positive reward-to-VWAP.
        def tradeable(stop_bps: Optional[float]) -> Optional[bool]:
            if stop_bps is None or target_bps is None:
                return None
            try:
                if not (math.isfinite(float(stop_bps)) and math.isfinite(float(target_bps))):
                    return None
                if float(target_bps) <= 0:
                    return False
                return float(stop_bps) >= float(gate_threshold_bps)
            except Exception:
                return None

        tradeable_current = tradeable(stop_pct_bps_final)
        tradeable_stop15 = tradeable(stop_bps_15)
        tradeable_rc_raw = tradeable(stop_bps_rc_raw)
        tradeable_rc_gate = tradeable(stop_bps_rc_gate)

        # Counterfactual 3: entry-anchored stop: stop_bps ~= k * std_bps
        stop_bps_entry = None
        if std_bps is not None and math.isfinite(float(std_bps)) and float(std_bps) >= 0:
            stop_bps_entry = float(entry_anchored_k) * float(std_bps)
        stop_price_entry = None
        if stop_bps_entry is not None and price is not None and eff_side in ("long", "short"):
            stop_price_entry = _stop_price_from_bps(side=eff_side, entry_price=float(price), stop_bps=float(stop_bps_entry))

        stop_bps_entry_gate = None
        if stop_bps_entry is not None:
            stop_bps_entry_gate = max(float(gate_threshold_bps), float(stop_bps_entry))
        stop_price_entry_gate = None
        if stop_bps_entry_gate is not None and price is not None and eff_side in ("long", "short"):
            stop_price_entry_gate = _stop_price_from_bps(side=eff_side, entry_price=float(price), stop_bps=float(stop_bps_entry_gate))

        rr_ratio_entry = _rr_ratio(reward_bps=target_bps, risk_bps=stop_bps_entry)
        rr_ratio_net_entry = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_bps_entry, cost_bps=float(args.cost_bps))
        rr_ratio_entry_gate = _rr_ratio(reward_bps=target_bps, risk_bps=stop_bps_entry_gate)
        rr_ratio_net_entry_gate = _rr_ratio_net(reward_bps=target_bps, risk_bps=stop_bps_entry_gate, cost_bps=float(args.cost_bps))

        tradeable_entry = tradeable(stop_bps_entry)
        tradeable_entry_gate = tradeable(stop_bps_entry_gate)

        row = {
            "kind": kind,
            "ts_ms": ts_ms,
            "symbol": symbol,
            "action": action,
            "side": eff_side,
            "signal_id": gate.signal_id if gate else None,
            "dedupe_key": gate.dedupe_key if gate else None,
            "reason_code": gate.reason_code if gate else None,
            "drop_reason": gate.drop_reason if gate else None,
            "entry_price": price,
            "px_source": gate.px_source if gate else None,
            "stop_price_observed": stop_price_observed,
            "stop_distance_observed": stop_distance_observed,
            "stop_distance_bps_observed": stop_distance_bps_observed,
            "stop_price_expected": stop_price_expected,
            "target_price": target_price,
            "target_bps": target_bps,
            "rr_ratio": rr_ratio_current,
            "rr_ratio_net": rr_ratio_net_current,
            "net_rr": net_rr_current,
            "reward_bps": reward_bps,
            "cost_bps_assumed": cost_bps_assumed,
            "std_bps": std_bps,
            "atr_bps": atr_bps,
            "k_implied": k_implied,
            "tradeable": tradeable_current,
            "stop_bps_cf_15": stop_bps_15,
            "stop_price_cf_15": stop_price_15,
            "rr_ratio_cf_15": rr_ratio_stop15,
            "rr_ratio_net_cf_15": rr_ratio_net_stop15,
            "tradeable_cf_15": tradeable_stop15,
            "stop_bps_rc_raw": stop_bps_rc_raw,
            "stop_price_rc_raw": stop_price_rc_raw,
            "rr_ratio_rc_raw": rr_ratio_rc_raw,
            "rr_ratio_net_rc_raw": rr_ratio_net_rc_raw,
            "tradeable_rc_raw": tradeable_rc_raw,
            "stop_bps_rc_gate": stop_bps_rc_gate,
            "stop_price_rc_gate": stop_price_rc_gate,
            "rr_ratio_rc_gate": rr_ratio_rc_gate,
            "rr_ratio_net_rc_gate": rr_ratio_net_rc_gate,
            "tradeable_rc_gate": tradeable_rc_gate,
            "entry_anchored_k": float(entry_anchored_k),
            "stop_bps_entry": stop_bps_entry,
            "stop_price_entry": stop_price_entry,
            "rr_ratio_entry": rr_ratio_entry,
            "rr_ratio_net_entry": rr_ratio_net_entry,
            "tradeable_entry": tradeable_entry,
            "stop_bps_entry_gate": stop_bps_entry_gate,
            "stop_price_entry_gate": stop_price_entry_gate,
            "rr_ratio_entry_gate": rr_ratio_entry_gate,
            "rr_ratio_net_entry_gate": rr_ratio_net_entry_gate,
            "tradeable_entry_gate": tradeable_entry_gate,
            "volume_bucket": vol.bucket if vol else None,
            "volume_strength": vol.strength if vol else None,
            "vwap": ctrl.vwap if ctrl else None,
            "vwap_std": ctrl.vwap_std if ctrl else None,
            "adx": ctrl.adx if ctrl else None,
            "atr": ctrl.atr if ctrl else None,
            "atr_pct": ctrl.atr_pct if ctrl else None,
            "z": ctrl.z if ctrl else None,
            "lower": ctrl.lower if ctrl else None,
            "upper": ctrl.upper if ctrl else None,
            "band_multiplier": ctrl.band_multiplier if ctrl else None,
            "band_width_bps": band_width_bps,
            "dist_outside_bps": dist_outside_bps,
            "stop_pct_bps_observed": stop_pct_bps,
            "stop_pct_bps_expected": expected_stop_pct_bps,
            "stop_pct_bps": stop_pct_bps_final,
            "gate_threshold_bps": gate_threshold_bps,
            "gate_margin_bps_observed": gate.gate_margin_bps if gate else None,
            "gate_margin_bps_expected": gate_margin_bps_expected,
            "stop_source_guess": stop_source_guess,
        }

        rows.append(row)

    # 1) Add all far-drop cases
    far_drops = [g for g in gates if (g.event == "waiting_room_drop" and g.reason_code == "volume.low_vol_tight_stop_far")]
    for g in far_drops:
        add_row("drop_far", g.ts_ms, g.symbol, g.side, None, g)

    # 2) Add all MR signals (Action: SIGNAL)
    for s in mr_signals:
        if s.action != "SIGNAL":
            continue
        # Attach ingress side if possible
        ing = _nearest_by_ts(ingress_by_sym.get(s.symbol, []), s.ts_ms, window_ms=int(args.join_window_ms))
        side = ing.side if ing else None

        # Try to attach nearest waiting_room event within join window, for outcome labeling.
        # This helps separate "normal" (no gating event) from "tight stop" gating.
        near_gate = _nearest_by_ts([g for g in gates if g.symbol == s.symbol], s.ts_ms, window_ms=int(args.join_window_ms))

        add_row("signal", s.ts_ms, s.symbol, side, s.action, near_gate)

    # Output CSV
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = Path("reports") / f"mr_low_vol_gating_{log_file.stem}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def _fmt(v: Any, *, digits: int = 6) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            if not math.isfinite(float(v)):
                return ""
            x = float(v)
            if abs(x) >= 1000:
                return f"{x:.2f}"
            if abs(x) >= 100:
                return f"{x:.3f}"
            return f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return str(v)

    def _md_table(rows_in: List[Dict[str, Any]], cols: List[str]) -> str:
        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        lines = []
        for r in rows_in:
            lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n")
        return header + sep + "".join(lines)

    # Summary
    tight_threshold = float(args.gate_threshold_bps)

    far_stop_bps = [r["stop_pct_bps"] for r in rows if r["kind"] == "drop_far" and isinstance(r.get("stop_pct_bps"), (int, float))]
    sig_stop_bps = [r["stop_pct_bps"] for r in rows if r["kind"] == "signal" and isinstance(r.get("stop_pct_bps"), (int, float))]

    sig_tight = [x for x in sig_stop_bps if x < tight_threshold]
    sig_normal = [x for x in sig_stop_bps if x >= tight_threshold]

    print(f"Parsed: controllers={len(controllers)} volumes={len(volumes)} gates={len(gates)} mr_signals={len(mr_signals)}")
    print(f"Far drops: {len(far_drops)}")
    print(f"CSV: {out_csv}")
    print("")

    if far_stop_bps:
        print("Far-drop stop_pct_bps quantiles:", _quantiles([float(x) for x in far_stop_bps]))
    if sig_stop_bps:
        print("All-signal stop_pct_bps quantiles:", _quantiles([float(x) for x in sig_stop_bps]))
        print(f"Signals tight(<{tight_threshold}bps): {len(sig_tight)}/{len(sig_stop_bps)}")
        print(f"Signals normal(>={tight_threshold}bps): {len(sig_normal)}/{len(sig_stop_bps)}")

    # Hypothesis helpers: compare vwap_std / band_width_bps on far-drop vs signal
    def collect(name: str, kind: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            if r.get("kind") != kind:
                continue
            v = r.get(name)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
        return out

    for metric in ("vwap_std", "band_width_bps", "atr_pct", "dist_outside_bps"):
        a = collect(metric, "drop_far")
        b = collect(metric, "signal")
        if a and b:
            print(f"\nMetric={metric}")
            print("  drop_far:", _quantiles(a))
            print("  signal  :", _quantiles(b))

    # Optional Markdown summary
    out_md = args.out_md
    if out_md is None:
        out_md = Path("reports") / f"mr_low_vol_gating_{log_file.stem}.md"

    try:
        far_rows = [r for r in rows if r.get("kind") == "drop_far"]
        far_rows_sorted = sorted(
            far_rows,
            key=lambda r: (
                0 if r.get("side") in ("long", "short") else 1,
                str(r.get("side") or ""),
                -float(r.get("rr_ratio")) if isinstance(r.get("rr_ratio"), (int, float)) else float("-inf"),
            ),
        )
    except Exception:
        far_rows_sorted = [r for r in rows if r.get("kind") == "drop_far"]

    top_n = max(1, int(args.top_n))
    far_top = far_rows_sorted[:top_n]

    md_lines: List[str] = []
    md_lines.append(f"# MR Low-Volume Gating Summary\n\n")
    md_lines.append(f"- Log: `{log_file.name}`\n")
    md_lines.append(f"- CSV: `{out_csv.as_posix()}`\n")
    md_lines.append(f"- Far drops: {len(far_drops)}\n")
    md_lines.append(f"- Gate threshold (bps): {_fmt(tight_threshold)}\n\n")
    md_lines.append(f"- Cost (bps, round-trip): {_fmt(float(args.cost_bps))}\n")
    md_lines.append(f"- Reward-consistent RR target: {_fmt(float(args.reward_consistent_rr))}\n")
    md_lines.append(f"- Reward-consistent clamp stop_bps: [{_fmt(float(args.reward_consistent_min_stop_bps))}, {_fmt(float(args.reward_consistent_max_stop_bps))}]\n\n")
    md_lines.append(f"- Entry-anchored k: {_fmt(float(entry_anchored_k))}\n\n")

    if far_stop_bps:
        md_lines.append("## Far-drop Stop Size (bps)\n\n")
        md_lines.append(f"Quantiles: `{_quantiles([float(x) for x in far_stop_bps])}`\n\n")

    # Tradeable summary (per scenario)
    def _count_true(key: str) -> int:
        c = 0
        for r in far_rows:
            if r.get(key) is True:
                c += 1
        return c

    md_lines.append("## Counterfactual Summary\n\n")
    md_lines.append(f"- tradeable (current): {_count_true('tradeable')}/{len(far_rows)}\n")
    md_lines.append(f"- tradeable (stop=threshold): {_count_true('tradeable_cf_15')}/{len(far_rows)}\n")
    md_lines.append(f"- tradeable (reward-consistent raw): {_count_true('tradeable_rc_raw')}/{len(far_rows)}\n")
    md_lines.append(f"- tradeable (reward-consistent + gate floor): {_count_true('tradeable_rc_gate')}/{len(far_rows)}\n\n")
    md_lines.append(f"- tradeable (entry-anchored): {_count_true('tradeable_entry')}/{len(far_rows)}\n")
    md_lines.append(f"- tradeable (entry-anchored + gate floor): {_count_true('tradeable_entry_gate')}/{len(far_rows)}\n\n")

    # RR summaries (gross + net)
    def _collect_metric(metric: str) -> List[float]:
        out: List[float] = []
        for r in far_rows:
            v = r.get(metric)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
        return out

    md_lines.append("## RR Distributions (far-drops)\n\n")
    for metric, label in (
        ("rr_ratio", "gross RR (current)"),
        ("rr_ratio_net", "net RR (current)"),
        ("rr_ratio_cf_15", "gross RR (stop=threshold)"),
        ("rr_ratio_net_cf_15", "net RR (stop=threshold)"),
        ("rr_ratio_rc_raw", "gross RR (reward-consistent raw)"),
        ("rr_ratio_net_rc_raw", "net RR (reward-consistent raw)"),
        ("rr_ratio_rc_gate", "gross RR (reward-consistent + gate floor)"),
        ("rr_ratio_net_rc_gate", "net RR (reward-consistent + gate floor)"),
        ("rr_ratio_entry", "gross RR (entry-anchored)"),
        ("rr_ratio_net_entry", "net RR (entry-anchored)"),
        ("rr_ratio_entry_gate", "gross RR (entry-anchored + gate floor)"),
        ("rr_ratio_net_entry_gate", "net RR (entry-anchored + gate floor)"),
    ):
        vals = _collect_metric(metric)
        if vals:
            md_lines.append(f"- {label}: `{_quantiles(vals)}`\n")
    md_lines.append("\n")

    md_lines.append("## Volatility Proxy Diagnostics (far-drops)\n\n")
    for metric, label in (
        ("std_bps", "std_bps = (vwap_std/entry)*1e4"),
        ("atr_bps", "atr_bps = (atr/entry)*1e4"),
        ("k_implied", "k_implied = stop_bps/std_bps"),
    ):
        vals = _collect_metric(metric)
        if vals:
            md_lines.append(f"- {label}: `{_quantiles(vals)}`\n")
    md_lines.append("\n")

    # Correlations (Dal 2): k_implied vs other metrics
    def _corr_pair(x_key: str, y_key: str) -> Optional[float]:
        xs: List[float] = []
        ys: List[float] = []
        for r in far_rows:
            x = r.get(x_key)
            y = r.get(y_key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(float(x)) and math.isfinite(float(y)):
                xs.append(float(x))
                ys.append(float(y))
        return _pearson_corr(xs, ys)

    md_lines.append("## Implied Multiplier Correlations (far-drops)\n\n")
    for y_key in ("volume_strength", "band_width_bps", "dist_outside_bps", "std_bps", "atr_bps"):
        corr = _corr_pair("k_implied", y_key)
        if corr is not None:
            md_lines.append(f"- corr(k_implied, {y_key}) = {_fmt(corr, digits=4)}\n")
    md_lines.append("\n")

    def _top_bottom(metric: str, n: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        valid = [r for r in far_rows if isinstance(r.get(metric), (int, float)) and math.isfinite(float(r.get(metric))) ]
        valid_sorted = sorted(valid, key=lambda r: float(r.get(metric)))
        bottom = valid_sorted[:n]
        top = list(reversed(valid_sorted[-n:]))
        return top, bottom

    md_lines.append("## Net-RR Extremes (far-drops)\n\n")
    for metric, label in (
        ("rr_ratio_net", "net RR (current)"),
        ("rr_ratio_net_cf_15", "net RR (stop=threshold)"),
        ("rr_ratio_net_rc_gate", "net RR (reward-consistent + gate floor)"),
        ("rr_ratio_net_entry_gate", "net RR (entry-anchored + gate floor)"),
    ):
        top, bottom = _top_bottom(metric, n=5)
        if not top and not bottom:
            continue
        md_lines.append(f"### {label}\n\n")
        cols_small = ["ts_ms", "side", "volume_strength", "z", "target_bps", "stop_pct_bps", metric, "signal_id"]
        if top:
            md_lines.append("Top 5:\n\n")
            md_lines.append(_md_table(top, cols_small))
            md_lines.append("\n")
        if bottom:
            md_lines.append("Bottom 5:\n\n")
            md_lines.append(_md_table(bottom, cols_small))
            md_lines.append("\n")

    # Characterize tradeable rows (z, target_bps, band_width_bps, vwap_std)
    def _collect_tradeable(metric: str, trade_key: str) -> List[float]:
        out: List[float] = []
        for r in far_rows:
            if r.get(trade_key) is not True:
                continue
            v = r.get(metric)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
        return out

    md_lines.append("## Tradeable Characteristics (by scenario)\n\n")
    for trade_key, label in (
        ("tradeable_cf_15", "stop=threshold"),
        ("tradeable_rc_gate", "reward-consistent + gate floor"),
        ("tradeable_rc_raw", "reward-consistent raw"),
        ("tradeable_entry_gate", "entry-anchored + gate floor"),
        ("tradeable_entry", "entry-anchored"),
    ):
        md_lines.append(f"### {label}\n\n")
        any_rows = False
        for metric in ("z", "target_bps", "band_width_bps", "vwap_std"):
            vals = _collect_tradeable(metric, trade_key)
            if vals:
                any_rows = True
                md_lines.append(f"- {metric} quantiles: `{_quantiles(vals)}`\n")
        if not any_rows:
            md_lines.append("- (no tradeable rows)\n")
        md_lines.append("\n")

    md_lines.append(f"## Far-Drops Table (first {len(far_top)}; set --top-n 33 for all)\n\n")
    cols = [
        "ts_ms",
        "side",
        "signal_id",
        "dedupe_key",
        "volume_bucket",
        "volume_strength",
        "entry_price",
        "stop_price_observed",
        "stop_pct_bps",
        "target_price",
        "target_bps",
        "rr_ratio",
        "rr_ratio_net",
        "tradeable",
        "std_bps",
        "atr_bps",
        "k_implied",
        "stop_bps_cf_15",
        "rr_ratio_cf_15",
        "rr_ratio_net_cf_15",
        "tradeable_cf_15",
        "stop_bps_rc_raw",
        "rr_ratio_rc_raw",
        "rr_ratio_net_rc_raw",
        "tradeable_rc_raw",
        "stop_bps_rc_gate",
        "rr_ratio_rc_gate",
        "rr_ratio_net_rc_gate",
        "tradeable_rc_gate",
        "stop_bps_entry",
        "rr_ratio_net_entry",
        "tradeable_entry",
        "stop_bps_entry_gate",
        "rr_ratio_net_entry_gate",
        "tradeable_entry_gate",
        "vwap_std",
        "band_width_bps",
        "adx",
        "z",
        "dist_outside_bps",
        "gate_margin_bps_observed",
        "stop_source_guess",
        "reason_code",
    ]
    md_lines.append(_md_table(far_top, cols))
    md_lines.append("\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(f"MD: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
