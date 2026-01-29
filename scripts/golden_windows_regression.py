#!/usr/bin/env python3
"""
Golden-window regression for Crash Guard (adaptive_ob).

Inputs:
  - windows.yaml: name,start,end,timezone=UTC,notes (+ optional incident block)
  - windows_expectations.yaml: per-window expected metric ranges (min/max)

Outputs:
  - results.json
  - results.csv

CI usage:
  python scripts/golden_windows_regression.py --validate \
    --windows-yaml scripts/windows.yaml \
    --expectations-yaml scripts/windows_expectations.yaml \
    --output-json artifacts/golden_windows/results.json \
    --output-csv artifacts/golden_windows/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml  # repo-local minimal yaml shim

from scripts.analyze_crash_guard_windows import analyze_window, _parse_iso_utc  # type: ignore
from scripts.replay_crash_guard_incident import compute_incident_metrics  # type: ignore


@dataclass(frozen=True)
class GoldenWindow:
    name: str
    start: datetime
    end: datetime
    timezone: str
    notes: str
    incident: Optional[Dict[str, Any]] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw) or {}


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _extract_mfe_mae_quantiles(rep: Dict[str, Any], *, horizon: str) -> Dict[str, Optional[float]]:
    mm = (rep.get("mfe_mae") or {}).get(str(horizon)) or {}
    return {
        "n": _as_int(mm.get("n")),
        "mfe_p50": _as_float(mm.get("mfe_median")),
        "mfe_p90": _as_float(mm.get("mfe_p90")),
        "mae_p50": _as_float(mm.get("mae_median")),
        "mae_p90": _as_float(mm.get("mae_p90")),
        "mfe_mean": _as_float(mm.get("mfe_mean")),
        "mae_mean": _as_float(mm.get("mae_mean")),
    }


def _flatten_bucket_counts(bucket_counts: Dict[str, Any]) -> str:
    try:
        return json.dumps(bucket_counts, sort_keys=True)
    except Exception:
        return str(bucket_counts)


def _validate_ranges(
    *,
    name: str,
    results: Dict[str, Any],
    expectations: Dict[str, Any],
) -> List[str]:
    exp = (expectations.get("expectations") or {}).get(name) or {}
    violations: List[str] = []

    def _check(metric: str, value: Optional[float], cfg: Dict[str, Any]) -> None:
        if value is None:
            return
        mn = _as_float(cfg.get("min"))
        mx = _as_float(cfg.get("max"))
        if mn is not None and value < mn:
            violations.append(f"{name}: {metric}={value:.6f} < min={mn:.6f}")
        if mx is not None and value > mx:
            violations.append(f"{name}: {metric}={value:.6f} > max={mx:.6f}")

    def _check_int(metric: str, value: Optional[int], cfg: Dict[str, Any]) -> None:
        if value is None:
            return
        mn = _as_int(cfg.get("min"))
        mx = _as_int(cfg.get("max"))
        if mn is not None and value < mn:
            violations.append(f"{name}: {metric}={value} < min={mn}")
        if mx is not None and value > mx:
            violations.append(f"{name}: {metric}={value} > max={mx}")

    for metric, cfg in exp.items():
        if metric == "incident":
            continue
        if not isinstance(cfg, dict):
            continue
        value = _as_float((results.get("metrics") or {}).get(metric))
        _check(metric, value, cfg)

    inc_exp = exp.get("incident") or {}
    inc = results.get("incident") or {}
    if isinstance(inc_exp, dict) and inc_exp:
        if not isinstance(inc, dict) or inc.get("error"):
            violations.append(f"{name}: incident metrics missing or errored")
            return violations
    if isinstance(inc_exp, dict) and isinstance(inc, dict):
        for metric, cfg in inc_exp.items():
            if not isinstance(cfg, dict):
                continue
            if metric == "panic_veto_no_reversal":
                value = _as_int(((inc.get("entry") or {}).get("by_reason") or {}).get("panic_veto_no_reversal"))
                _check_int("incident.panic_veto_no_reversal", value, cfg)
            elif metric == "churn_drop_count":
                value = _as_int((inc.get("churn") or {}).get("drop_count"))
                _check_int("incident.churn_drop_count", value, cfg)

    return violations


async def _run() -> int:
    ap = argparse.ArgumentParser(description="Golden window regression for Crash Guard (adaptive_ob).")
    ap.add_argument("--exchange", default="bingx")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--config", default="config/config.example.yaml")
    ap.add_argument("--windows-yaml", default="scripts/windows.yaml")
    ap.add_argument("--expectations-yaml", default="scripts/windows_expectations.yaml")
    ap.add_argument("--output-json", default="artifacts/golden_windows/results.json")
    ap.add_argument("--output-csv", default="artifacts/golden_windows/results.csv")
    ap.add_argument("--sample-step-min", type=int, default=1)
    ap.add_argument("--horizons-min", default="5,10")
    ap.add_argument("--lookback-days-volume", type=int, default=30)
    ap.add_argument("--lookback-hours-fast", type=int, default=6)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    horizons = [int(x.strip()) for x in str(args.horizons_min).split(",") if x.strip()]
    if not horizons:
        horizons = [5, 10]

    windows_doc = _load_yaml(Path(args.windows_yaml))
    windows_raw = windows_doc.get("windows") or []
    if not isinstance(windows_raw, list) or not windows_raw:
        raise SystemExit(f"No windows found in {args.windows_yaml}")

    windows: List[GoldenWindow] = []
    for w in windows_raw:
        if not isinstance(w, dict):
            continue
        name = str(w.get("name") or "").strip()
        if not name:
            continue
        start = _parse_iso_utc(str(w.get("start")))
        end = _parse_iso_utc(str(w.get("end")))
        tz = str(w.get("timezone") or "UTC").strip() or "UTC"
        notes = str(w.get("notes") or "").strip()
        windows.append(
            GoldenWindow(
                name=name,
                start=start,
                end=end,
                timezone=tz,
                notes=notes,
                incident=(w.get("incident") if isinstance(w.get("incident"), dict) else None),
            )
        )

    expectations = _load_yaml(Path(args.expectations_yaml)) if args.expectations_yaml else {}

    out: Dict[str, Any] = {
        "schema": "golden_windows_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": str(args.exchange),
        "symbol": str(args.symbol),
        "config": str(args.config),
        "windows": [],
    }

    rows: List[Dict[str, Any]] = []
    violations: List[str] = []

    for w in windows:
        rep = await analyze_window(
            exchange=str(args.exchange),
            symbol=str(args.symbol),
            cfg=cfg,
            window=type("W", (), {"name": w.name, "start": w.start, "end": w.end})(),  # Window-compatible
            sample_step_min=int(args.sample_step_min),
            horizons_min=horizons,
            lookback_days_volume=int(args.lookback_days_volume),
            lookback_hours_fast=int(args.lookback_hours_fast),
        )

        metrics = {
            "eligible_share": _as_float(((rep.get("rates") or {}).get("eligible_share"))),
            "accepted_rate": _as_float(((rep.get("rates") or {}).get("accepted_rate"))),
            "blocked_rate": _as_float(((rep.get("rates") or {}).get("blocked_rate"))),
            "strict_extreme_share": _as_float(((rep.get("rates") or {}).get("strict_extreme_share"))),
            "missing_reclaim": _as_int(((rep.get("counts") or {}).get("missing_reclaim"))),
        }

        mfe5 = _extract_mfe_mae_quantiles(rep, horizon="5")
        mfe10 = _extract_mfe_mae_quantiles(rep, horizon="10")
        bucket_counts = rep.get("volume_bucket_counts") or {}

        incident_summary = None
        if w.incident:
            try:
                pos_ids = w.incident.get("positions") or []
                if not isinstance(pos_ids, list):
                    pos_ids = []
                incident_summary = await compute_incident_metrics(
                    log_path=Path(str(w.incident.get("log"))),
                    exchange=str(args.exchange),
                    symbol=str(args.symbol),
                    window_start=w.start,
                    window_end=w.end,
                    config_path=Path(args.config),
                    position_ids=[str(p) for p in pos_ids if p],
                    lookback_hours=int(args.lookback_hours_fast),
                )
            except Exception as exc:
                incident_summary = {"error": f"{type(exc).__name__}: {exc}"}

        window_out = {
            "name": w.name,
            "start_utc": w.start.isoformat(),
            "end_utc": w.end.isoformat(),
            "timezone": w.timezone,
            "notes": w.notes,
            "metrics": metrics,
            "bucket_counts": bucket_counts,
            "mfe_mae": {"5m": mfe5, "10m": mfe10},
            "incident": incident_summary,
        }
        out["windows"].append(window_out)

        rows.append(
            {
                "name": w.name,
                "start_utc": w.start.isoformat(),
                "end_utc": w.end.isoformat(),
                "eligible_share": metrics["eligible_share"],
                "accepted_rate": metrics["accepted_rate"],
                "blocked_rate": metrics["blocked_rate"],
                "strict_extreme_share": metrics["strict_extreme_share"],
                "missing_reclaim": metrics["missing_reclaim"],
                "bucket_counts": _flatten_bucket_counts(bucket_counts),
                "mfe_5m_p50": mfe5.get("mfe_p50"),
                "mfe_5m_p90": mfe5.get("mfe_p90"),
                "mae_5m_p50": mfe5.get("mae_p50"),
                "mae_5m_p90": mfe5.get("mae_p90"),
                "mfe_10m_p50": mfe10.get("mfe_p50"),
                "mfe_10m_p90": mfe10.get("mfe_p90"),
                "mae_10m_p50": mfe10.get("mae_p50"),
                "mae_10m_p90": mfe10.get("mae_p90"),
                "incident_panic_veto_no_reversal": (
                    ((incident_summary or {}).get("entry") or {}).get("by_reason") or {}
                ).get("panic_veto_no_reversal")
                if isinstance(incident_summary, dict)
                else None,
                "incident_churn_drop_count": ((incident_summary or {}).get("churn") or {}).get("drop_count")
                if isinstance(incident_summary, dict)
                else None,
            }
        )

        violations.extend(_validate_ranges(name=w.name, results=window_out, expectations=expectations))

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    out_csv_path = Path(args.output_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if args.validate and violations:
        print("Golden-window regression violations:")
        for v in violations:
            print(f"- {v}")
        return 2

    if violations:
        print("Golden-window regression warnings:")
        for v in violations:
            print(f"- {v}")
    return 0


def main() -> int:
    import asyncio

    return int(asyncio.run(_run()))


if __name__ == "__main__":
    raise SystemExit(main())
