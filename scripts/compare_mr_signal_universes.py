"""Compare MR signal universe outputs across runs.

Inputs: one or more CSVs produced by scripts/analyze_mr_signal_universe.py
Outputs: a Markdown comparison report in reports/

This script is read-only: it only reads CSVs and writes a report.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"none"}:
            return None
        return int(float(s))
    except Exception:
        return None


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(values) - 1, f + 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


@dataclass
class Row:
    run_id: str
    ts_ms: Optional[int]
    volume_bucket: str
    outcome: str
    stop_bps: Optional[float]
    stop_obs_bps: Optional[float]
    stop_model_bps: Optional[float]
    target_bps: Optional[float]
    rr_net: Optional[float]


def infer_run_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("mr_signal_universe_") and base.endswith(".csv"):
        return base[len("mr_signal_universe_") : -len(".csv")]
    return os.path.splitext(base)[0]


def read_rows(csv_path: str) -> List[Row]:
    run_id = infer_run_id_from_path(csv_path)
    out: List[Row] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(
                Row(
                    run_id=r.get("run_id") or run_id,
                    ts_ms=_safe_int(r.get("ts_ms")),
                    volume_bucket=(r.get("volume_bucket") or "(missing)").strip() or "(missing)",
                    outcome=(r.get("final_outcome") or "unknown").strip() or "unknown",
                    stop_bps=_safe_float(r.get("stop_pct_bps")),
                    stop_obs_bps=_safe_float(r.get("stop_pct_bps_observed")),
                    stop_model_bps=_safe_float(r.get("stop_pct_bps_model")),
                    target_bps=_safe_float(r.get("target_bps")),
                    rr_net=_safe_float(r.get("rr_net")),
                )
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="Path to mr_signal_universe_<run>.csv (repeatable)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--micro-threshold-bps", type=float, default=15.0)
    args = ap.parse_args()

    all_rows: List[Row] = []
    for p in args.csv:
        all_rows.extend(read_rows(p))

    # Group by run
    by_run: Dict[str, List[Row]] = defaultdict(list)
    for r in all_rows:
        by_run[r.run_id].append(r)

    runs = sorted(by_run.keys())

    # Determine output
    repo_root = os.path.dirname(os.path.dirname(__file__))
    reports_dir = os.path.join(repo_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_md = args.out_md
    if out_md is None:
        if len(runs) == 1:
            out_md = os.path.join(reports_dir, f"mr_signal_universe_compare_{runs[0]}.md")
        else:
            out_md = os.path.join(reports_dir, f"mr_signal_universe_compare_{runs[0]}_vs_{runs[-1]}.md")

    micro_th = float(args.micro_threshold_bps)

    md: List[str] = []
    md.append("# MR Signal Universe Comparison")
    md.append("")
    md.append("## (1) Run overview")
    md.append("")

    overview_rows: List[List[str]] = []
    for run_id in runs:
        rows = by_run[run_id]
        outcomes = Counter([r.outcome for r in rows])
        buckets = Counter([r.volume_bucket for r in rows])

        stop_present = [r for r in rows if r.stop_bps is not None]
        micro = [r for r in stop_present if r.stop_bps < micro_th]
        big = [r for r in stop_present if r.stop_bps >= micro_th]

        obs = [r for r in stop_present if r.stop_obs_bps is not None]
        model = [r for r in stop_present if r.stop_obs_bps is None and r.stop_model_bps is not None]

        overview_rows.append(
            [
                run_id,
                str(len(rows)),
                ", ".join([f"{k}={v}" for k, v in outcomes.most_common(4)]),
                ", ".join([f"{k}={v}" for k, v in buckets.most_common(4)]),
                f"{len(stop_present)}",
                f"{len(micro)}/{len(stop_present)} ({(len(micro)/max(1,len(stop_present))*100):.1f}%)",
                f"{len(big)}",
                f"obs={len(obs)}, model={len(model)}",
            ]
        )

    md.append(
        md_table(
            [
                "run_id",
                "n",
                "outcomes(top)",
                "buckets(top)",
                "n_stop",
                f"micro(<{micro_th:.0f}bps)",
                f">={micro_th:.0f}bps",
                "stop_source",
            ],
            overview_rows,
        )
    )
    md.append("")

    md.append("## (2) Micro-stop by bucket (where stop exists)")
    md.append("")

    bucket_rows: List[List[str]] = []
    for run_id in runs:
        rows = by_run[run_id]
        buckets = sorted(set([r.volume_bucket for r in rows]))
        for b in buckets:
            b_rows = [r for r in rows if r.volume_bucket == b and r.stop_bps is not None]
            if not b_rows:
                continue
            micro = [r for r in b_rows if r.stop_bps < micro_th]
            rr_vals = [r.rr_net for r in b_rows if r.rr_net is not None]
            stop_vals = [r.stop_bps for r in b_rows if r.stop_bps is not None]
            target_vals = [r.target_bps for r in b_rows if r.target_bps is not None]

            bucket_rows.append(
                [
                    run_id,
                    b,
                    str(len(b_rows)),
                    f"{len(micro)}/{len(b_rows)} ({(len(micro)/len(b_rows)*100):.1f}%)",
                    f"{fmt(percentile(stop_vals,10))}/{fmt(percentile(stop_vals,50))}/{fmt(percentile(stop_vals,90))}",
                    f"{fmt(percentile(target_vals,10))}/{fmt(percentile(target_vals,50))}/{fmt(percentile(target_vals,90))}",
                    f"{fmt(percentile(rr_vals,10))}/{fmt(percentile(rr_vals,50))}/{fmt(percentile(rr_vals,90))}",
                ]
            )

    if bucket_rows:
        md.append(
            md_table(
                [
                    "run_id",
                    "bucket",
                    "n_stop",
                    f"micro(<{micro_th:.0f}bps)",
                    "stop_bps p10/p50/p90",
                    "target_bps p10/p50/p90",
                    "rr_net p10/p50/p90",
                ],
                bucket_rows,
            )
        )
    else:
        md.append("- No rows with stop_pct_bps present.")

    md.append("")
    md.append("## (3) Notes")
    md.append("")
    md.append("- Comparisons are only meaningful where `stop_pct_bps` exists.")
    md.append("- `stop_source` is inferred as observed when `stop_pct_bps_observed` is present; otherwise modeled when `stop_pct_bps_model` is present.")
    md.append("")

    with open(out_md, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(md))

    print(f"Wrote MD: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
