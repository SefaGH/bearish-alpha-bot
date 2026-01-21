#!/usr/bin/env python3
"""
Offline classification for RR-rejected Adaptive OB cases:

- Align each case timestamp to bar open-time per timeframe (floor).
  (Used as join key to resistance bands and as the reachability start.)
- Pick nearest-upper resistance band per timeframe (method preference: kmeans -> smc),
  then optionally choose a single "selected" band across timeframes.
- Classify TP location vs resistance band:
    TP < band_low      -> "TP band altı (konservatif)"
    band_low <= TP <= band_high -> "TP band içi (uyumlu)"
    TP > band_high     -> "TP band üstü (overshoot / muhtemel cap ihtiyacı)"
- Compute forward-horizon reachability metrics (MFE/MAE, touches, stopouts) using 5m OHLCV.

Production code is not modified; this is offline/diagnostic tooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_ts_utc(ts: str) -> datetime:
    text = str(ts).strip()
    if not text:
        raise ValueError("Empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Accept both "YYYY-mm-dd HH:MM:SS" and ISO8601.
    dt = datetime.fromisoformat(text.replace(" ", "T") if "T" not in text and " " in text else text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _iso_z(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _floor_ms(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 0:
        return int(ts_ms)
    return int(ts_ms // bucket_ms) * int(bucket_ms)


def _tp_band_label(tp: float, band_low: float, band_high: float) -> str:
    if tp < band_low:
        return "TP band altı (konservatif)"
    if tp <= band_high:
        return "TP band içi (uyumlu)"
    return "TP band üstü (overshoot / muhtemel cap ihtiyacı)"


def _tp_band_code(tp: float, band_low: float, band_high: float) -> str:
    if tp < band_low:
        return "TP_BELOW_BAND"
    if tp <= band_high:
        return "TP_IN_BAND"
    return "TP_ABOVE_BAND"


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@dataclass(frozen=True)
class BandPick:
    method: str
    level: float
    band_low: float
    band_high: float
    price_ref: float
    band_price: float


def _pick_band(
    *,
    bands_df: pd.DataFrame,
    ts_ms: int,
    timeframe: str,
    preferred_methods: Sequence[str],
) -> Optional[BandPick]:
    sub = bands_df[(bands_df["ts_ms"] == int(ts_ms)) & (bands_df["timeframe"] == timeframe)]
    if sub.empty:
        return None
    by_method = {str(r["method"]): r for _, r in sub.iterrows()}
    for m in preferred_methods:
        if m in by_method:
            r = by_method[m]
            return BandPick(
                method=str(r["method"]),
                level=float(r["nearest_res_level"]),
                band_low=float(r["band_low"]),
                band_high=float(r["band_high"]),
                price_ref=float(r.get("price", float("nan"))),
                band_price=float(r.get("price", float("nan"))),
            )
    # Fallback: pick any.
    r0 = sub.iloc[0]
    return BandPick(
        method=str(r0["method"]),
        level=float(r0["nearest_res_level"]),
        band_low=float(r0["band_low"]),
        band_high=float(r0["band_high"]),
        price_ref=float(r0.get("price", float("nan"))),
        band_price=float(r0.get("price", float("nan"))),
    )


def timeframe_to_ms(tf: str) -> int:
    t = str(tf).strip().lower()
    if t.endswith("m"):
        return int(t[:-1]) * 60_000
    if t.endswith("h"):
        return int(t[:-1]) * 60 * 60_000
    if t.endswith("d"):
        return int(t[:-1]) * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def _side_norm(side: Any) -> str:
    s = str(side or "").strip().lower()
    if s in {"long", "buy"}:
        return "long"
    if s in {"short", "sell"}:
        return "short"
    return s or "unknown"


def _first_idx(cond: pd.Series) -> Optional[int]:
    idx = cond[cond].index.tolist()
    return int(idx[0]) if idx else None


def _event_order(stop_i: Optional[int], other_i: Optional[int]) -> str:
    if stop_i is None and other_i is None:
        return "NONE"
    if stop_i is None:
        return "OTHER_ONLY"
    if other_i is None:
        return "STOP_ONLY"
    if stop_i < other_i:
        return "STOP_BEFORE_OTHER"
    if other_i < stop_i:
        return "OTHER_BEFORE_STOP"
    return "SAME_BAR"


def _reachability_window(
    *,
    ohlcv: pd.DataFrame,
    start_ts_ms: int,
    horizon_bars: int,
    side: str,
    entry: float,
    stop: Optional[float],
    tp: Optional[float],
    band_low: Optional[float],
    band_high: Optional[float],
) -> Dict[str, Any]:
    side = _side_norm(side)
    row: Dict[str, Any] = {}

    idxs = ohlcv.index[ohlcv["ts_ms"] == int(start_ts_ms)].tolist()
    if not idxs:
        row["reachability_missing_bar"] = True
        return row
    i0 = int(idxs[0])
    i1 = min(len(ohlcv), i0 + max(0, int(horizon_bars)))
    window = ohlcv.iloc[i0:i1]
    if window.empty:
        row["reachability_empty_window"] = True
        return row

    # Keep integer indices local to the window (0..bars-1) for event ordering.
    window = window.reset_index(drop=True)

    hi = float(window["high"].max())
    lo = float(window["low"].min())
    row["bars_available"] = int(len(window))
    row["window_high_max"] = hi
    row["window_low_min"] = lo

    if side == "short":
        mfe = entry - lo
        mae = hi - entry
        tp_hit = (tp is not None) and (window["low"] <= float(tp))
        stop_hit = (stop is not None) and (window["high"] >= float(stop))
        band_high_hit = (band_high is not None) and (window["high"] >= float(band_high))
        band_low_hit = (band_low is not None) and (window["high"] >= float(band_low))
    else:
        mfe = hi - entry
        mae = entry - lo
        tp_hit = (tp is not None) and (window["high"] >= float(tp))
        stop_hit = (stop is not None) and (window["low"] <= float(stop))
        band_high_hit = (band_high is not None) and (window["high"] >= float(band_high))
        band_low_hit = (band_low is not None) and (window["high"] >= float(band_low))

    row["mfe_abs"] = float(mfe)
    row["mae_abs"] = float(mae)
    row["touch_tp_within_h"] = bool(tp_hit.any()) if hasattr(tp_hit, "any") else bool(tp_hit)
    row["stopout_within_h"] = bool(stop_hit.any()) if hasattr(stop_hit, "any") else bool(stop_hit)
    row["touch_band_high_within_h"] = bool(band_high_hit.any()) if hasattr(band_high_hit, "any") else bool(band_high_hit)
    row["touch_band_low_within_h"] = bool(band_low_hit.any()) if hasattr(band_low_hit, "any") else bool(band_low_hit)

    stop_i = _first_idx(stop_hit) if hasattr(stop_hit, "index") else None
    tp_i = _first_idx(tp_hit) if hasattr(tp_hit, "index") else None
    bh_i = _first_idx(band_high_hit) if hasattr(band_high_hit, "index") else None

    row["stopout_idx"] = stop_i
    row["tp_touch_idx"] = tp_i
    row["band_high_touch_idx"] = bh_i
    row["stop_vs_tp_order"] = _event_order(stop_i, tp_i)
    row["stop_vs_band_high_order"] = _event_order(stop_i, bh_i)

    return row


def _case_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("ts", "")).strip(), str(item.get("symbol", "")).strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Classify RR-reject OB TPs vs resistance bands + reachability.")
    p.add_argument("--cases", type=Path, default=Path("rr_rejected_ob_cases_20260120.jsonl"))
    p.add_argument("--reco", type=Path, default=Path("ob_rr_reject_sl_tp_reco_20260120.jsonl"))
    p.add_argument("--bands", type=Path, default=Path("reports") / "resistance_bands_20260120_smc_lib.csv")
    p.add_argument("--band-timeframes", default="5m", help="Comma-separated band timeframes to consider (e.g. 1m,5m,30m)")
    p.add_argument(
        "--band-select-policy",
        default="prefer_tf_order",
        choices=["prefer_tf_order", "closest_level"],
        help="How to choose a single 'selected' band across multiple timeframes.",
    )
    p.add_argument("--method-preference", default="kmeans,smc", help="Comma-separated method preference order")
    p.add_argument("--reach-timeframe", default="5m", help="OHLCV timeframe used for reachability (default: 5m)")
    p.add_argument(
        "--reach-align-timeframe",
        help="Timeframe used to floor case timestamps for reachability start (default: reach-timeframe)",
    )
    p.add_argument("--ohlcv", type=Path, help="OHLCV CSV path for reach-timeframe (default: data_cache/ohlcv/bingx_BTC_USDT_USDT_{tf}.csv)")
    p.add_argument("--horizon-bars", type=int, default=12, help="Backward-compatible single horizon in reach-timeframe bars")
    p.add_argument("--horizons", default="12,24,36,48", help="Comma-separated horizons in reach-timeframe bars")
    p.add_argument("--out-csv", type=Path, default=Path("reports") / "rr_reject_ob_tp_vs_resistance_20260120.csv")
    p.add_argument(
        "--out-long-csv",
        type=Path,
        default=Path("reports") / "rr_reject_ob_tp_vs_resistance_20260120_long.csv",
        help="Long-form rows: case x horizon x scenario",
    )
    p.add_argument("--out-md", type=Path, default=Path("reports") / "rr_reject_ob_tp_vs_resistance_20260120.md")
    args = p.parse_args(list(argv) if argv is not None else None)

    cases = _read_jsonl(args.cases)
    recos = _read_jsonl(args.reco) if args.reco.exists() else []
    reco_by_key = {_case_key(r): r for r in recos}

    bands_df = pd.read_csv(args.bands)
    bands_df["ts_ms"] = bands_df["ts_ms"].astype("int64")
    bands_df["timeframe"] = bands_df["timeframe"].astype(str)
    bands_df["method"] = bands_df["method"].astype(str)

    reach_tf = str(args.reach_timeframe).strip()
    reach_align_tf = str(args.reach_align_timeframe or reach_tf).strip()
    ohlcv_path = args.ohlcv or (Path("data_cache") / "ohlcv" / f"bingx_BTC_USDT_USDT_{reach_tf}.csv")
    ohlcv = pd.read_csv(ohlcv_path)
    ohlcv["ts_ms"] = ohlcv["ts_ms"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        if col in ohlcv.columns:
            ohlcv[col] = ohlcv[col].astype(float)
    ohlcv = ohlcv.sort_values("ts_ms").drop_duplicates(subset=["ts_ms"]).reset_index(drop=True)

    reach_tf_ms = timeframe_to_ms(reach_tf)
    reach_align_tf_ms = timeframe_to_ms(reach_align_tf)

    preferred_methods = [m.strip() for m in str(args.method_preference).split(",") if m.strip()]
    if not preferred_methods:
        preferred_methods = ["kmeans", "smc"]

    band_tfs = [t.strip() for t in str(args.band_timeframes).split(",") if t.strip()]
    if not band_tfs:
        band_tfs = ["5m"]
    band_tf_ms = {tf: timeframe_to_ms(tf) for tf in band_tfs}

    horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]
    if not horizons:
        horizons = [int(args.horizon_bars)]

    out_rows: List[Dict[str, Any]] = []
    out_long_rows: List[Dict[str, Any]] = []
    for c in cases:
        ts_raw = str(c.get("ts", "")).strip()
        sym = str(c.get("symbol", "")).strip()
        side = _side_norm(c.get("side"))

        ts_ms = _ms(_parse_ts_utc(ts_raw))

        entry = float(c["entry"])
        stop_cur = _safe_float(c.get("stop"))
        tp_cur = _safe_float(c.get("target"))
        rr_cur = _safe_float(c.get("actual_rr"))
        req_rr = _safe_float(c.get("required_rr_v1") or c.get("dyn_final") or c.get("required_rr"))

        picks_by_tf: Dict[str, Tuple[int, Optional[BandPick]]] = {}
        for tf in band_tfs:
            open_ms = _floor_ms(ts_ms, band_tf_ms[tf])
            picks_by_tf[tf] = (
                int(open_ms),
                _pick_band(bands_df=bands_df, ts_ms=int(open_ms), timeframe=tf, preferred_methods=preferred_methods),
            )

        selected_tf: Optional[str] = None
        selected_open_ms: Optional[int] = None
        selected_band: Optional[BandPick] = None
        if args.band_select_policy == "closest_level":
            candidates: List[Tuple[float, str, int, BandPick]] = []
            for tf, (open_ms, pick) in picks_by_tf.items():
                if pick is None:
                    continue
                dist = float(pick.level) - float(entry)
                if dist <= 0:
                    continue
                candidates.append((dist, tf, int(open_ms), pick))
            if candidates:
                _, selected_tf, selected_open_ms, selected_band = min(candidates, key=lambda x: x[0])
        else:
            for tf in band_tfs:
                open_ms, pick = picks_by_tf[tf]
                if pick is not None:
                    selected_tf, selected_open_ms, selected_band = tf, int(open_ms), pick
                    break

        reco = reco_by_key.get(_case_key(c))
        hybrid_stop = _safe_float(((reco or {}).get("alternatives") or {}).get("hybrid", {}).get("stop"))
        hybrid_tp = _safe_float(((reco or {}).get("alternatives") or {}).get("hybrid", {}).get("target"))
        sl_only_stop = _safe_float(((reco or {}).get("alternatives") or {}).get("sl_only", {}).get("stop"))
        tp_only_tp = _safe_float(((reco or {}).get("alternatives") or {}).get("tp_only", {}).get("target"))

        reach_open_ms = _floor_ms(ts_ms, reach_align_tf_ms)

        row: Dict[str, Any] = {
            "case_ts": ts_raw,
            "case_ts_ms": ts_ms,
            "reach_timeframe": reach_tf,
            "reach_align_timeframe": reach_align_tf,
            "reach_bar_open_ts": _iso_z(reach_open_ms),
            "reach_bar_open_ts_ms": reach_open_ms,
            "symbol": sym,
            "strategy": c.get("strategy"),
            "side": side,
            "match_confidence": c.get("match_confidence"),
            "entry": entry,
            "stop_current": stop_cur,
            "tp_current": tp_cur,
            "actual_rr_current": rr_cur,
            "required_rr": req_rr,
            "hybrid_stop": hybrid_stop,
            "hybrid_tp": hybrid_tp,
            "sl_only_stop": sl_only_stop,
            "tp_only_tp": tp_only_tp,
            "horizons": ",".join(str(h) for h in horizons),
            "band_timeframes": ",".join(band_tfs),
            "band_select_policy": str(args.band_select_policy),
            "band_method_preference": ",".join(preferred_methods),
            "selected_band_tf": selected_tf,
            "selected_band_open_ts": _iso_z(int(selected_open_ms)) if selected_open_ms is not None else None,
        }

        # Per-timeframe band picks (wide columns).
        for tf, (open_ms, pick) in picks_by_tf.items():
            prefix = tf.replace("/", "_").replace(":", "_")
            row[f"{prefix}_band_open_ts"] = _iso_z(int(open_ms))
            row[f"{prefix}_band_open_ts_ms"] = int(open_ms)
            if pick is None:
                row[f"{prefix}_band_missing"] = True
                continue
            row[f"{prefix}_band_method"] = pick.method
            row[f"{prefix}_nearest_res_level"] = pick.level
            row[f"{prefix}_band_low"] = pick.band_low
            row[f"{prefix}_band_high"] = pick.band_high
            if tp_cur is not None:
                row[f"{prefix}_tp_current_vs_band_code"] = _tp_band_code(tp_cur, pick.band_low, pick.band_high)
            if hybrid_tp is not None:
                row[f"{prefix}_tp_hybrid_vs_band_code"] = _tp_band_code(hybrid_tp, pick.band_low, pick.band_high)

        # Selected band fields (used for reachability band-touch metrics).
        if selected_band is None:
            row["selected_band_missing"] = True
            sel_band_low = None
            sel_band_high = None
        else:
            row["selected_band_method"] = selected_band.method
            row["selected_nearest_res_level"] = selected_band.level
            row["selected_band_low"] = selected_band.band_low
            row["selected_band_high"] = selected_band.band_high
            sel_band_low = float(selected_band.band_low)
            sel_band_high = float(selected_band.band_high)
            if tp_cur is not None:
                row["selected_tp_current_vs_band_code"] = _tp_band_code(tp_cur, sel_band_low, sel_band_high)
            if hybrid_tp is not None:
                row["selected_tp_hybrid_vs_band_code"] = _tp_band_code(hybrid_tp, sel_band_low, sel_band_high)

        # Multi-horizon reachability (relative to reach_open_ms, using selected band).
        for h in horizons:
            reach_cur = _reachability_window(
                ohlcv=ohlcv,
                start_ts_ms=reach_open_ms,
                horizon_bars=int(h),
                side=side,
                entry=entry,
                stop=stop_cur,
                tp=tp_cur,
                band_low=sel_band_low,
                band_high=sel_band_high,
            )
            for k, v in reach_cur.items():
                row[f"h{h}_cur_{k}"] = v

            reach_hybrid = _reachability_window(
                ohlcv=ohlcv,
                start_ts_ms=reach_open_ms,
                horizon_bars=int(h),
                side=side,
                entry=entry,
                stop=hybrid_stop,
                tp=hybrid_tp,
                band_low=sel_band_low,
                band_high=sel_band_high,
            )
            for k, v in reach_hybrid.items():
                row[f"h{h}_hybrid_{k}"] = v

            reach_sl_only = _reachability_window(
                ohlcv=ohlcv,
                start_ts_ms=reach_open_ms,
                horizon_bars=int(h),
                side=side,
                entry=entry,
                stop=sl_only_stop,
                tp=tp_cur,
                band_low=sel_band_low,
                band_high=sel_band_high,
            )
            for k, v in reach_sl_only.items():
                row[f"h{h}_sl_only_{k}"] = v

            # Long-form rows (case x horizon x scenario) to answer "stopout then TP/band?" cleanly.
            for scenario, stop_v, tp_v, reach in [
                ("current", stop_cur, tp_cur, reach_cur),
                ("hybrid", hybrid_stop, hybrid_tp, reach_hybrid),
                ("sl_only", sl_only_stop, tp_cur, reach_sl_only),
            ]:
                out_long_rows.append(
                    {
                        "case_ts": ts_raw,
                        "bar_open_ts": _iso_z(reach_open_ms),
                        "symbol": sym,
                        "side": side,
                        "scenario": scenario,
                        "horizon_bars": int(h),
                        "entry": entry,
                        "stop": stop_v,
                        "tp": tp_v,
                        "selected_band_tf": selected_tf,
                        "selected_band_method": (selected_band.method if selected_band else None),
                        "selected_band_low": sel_band_low,
                        "selected_band_high": sel_band_high,
                        **{k: v for k, v in reach.items()},
                    }
                )

        out_rows.append(row)

    _write_csv(args.out_csv, out_rows)
    _write_csv(args.out_long_csv, out_long_rows)

    # Minimal markdown summary.
    out_md_lines: List[str] = []
    out_md_lines.append("# RR-Reject OB: TP vs Multi-TF Resistance Bands")
    out_md_lines.append("")
    out_md_lines.append(f"- Cases: **{len(out_rows)}**")
    out_md_lines.append(f"- Reachability OHLCV: `{reach_tf}` (`{ohlcv_path}`)")
    out_md_lines.append(f"- Reach alignment: `{reach_align_tf}`")
    out_md_lines.append(f"- Horizons (bars): `{','.join(str(h) for h in horizons)}`")
    out_md_lines.append(f"- Band timeframes: `{','.join(band_tfs)}` | Select policy: `{args.band_select_policy}`")
    out_md_lines.append(f"- Band method preference: `{','.join(preferred_methods)}`")
    out_md_lines.append("")

    df_out = pd.DataFrame(out_rows)
    df_long = pd.DataFrame(out_long_rows)
    if not df_out.empty:
        out_md_lines.append("## TP vs Band (Per TF)")
        for tf in band_tfs:
            prefix = tf.replace("/", "_").replace(":", "_")
            col = f"{prefix}_tp_current_vs_band_code"
            if col not in df_out.columns:
                continue
            vc = df_out[col].fillna("missing").value_counts().to_dict()
            out_md_lines.append(f"### {tf} (current TP)")
            for k, v in vc.items():
                out_md_lines.append(f"- **{k}**: {int(v)}")
        out_md_lines.append("")

        out_md_lines.append("## Reachability Summary (Selected Band)")
        if df_long.empty:
            out_md_lines.append("- (no reachability rows)")
        else:
            for h in horizons:
                out_md_lines.append(f"### Horizon h={h} ({reach_tf})")
                for scenario in ["current", "hybrid", "sl_only"]:
                    sub = df_long[(df_long["horizon_bars"] == int(h)) & (df_long["scenario"] == scenario)]
                    if sub.empty:
                        continue
                    n = int(len(sub))
                    n_stop = int(sub["stopout_within_h"].astype(bool).sum())
                    n_tp = int(sub["touch_tp_within_h"].astype(bool).sum())
                    n_bh = int(sub["touch_band_high_within_h"].astype(bool).sum())

                    both_tp = sub[sub["stopout_within_h"].astype(bool) & sub["touch_tp_within_h"].astype(bool)]
                    n_both_tp = int(len(both_tp))
                    n_tp_after = int((both_tp["stop_vs_tp_order"].astype(str) == "STOP_BEFORE_OTHER").sum())

                    both_bh = sub[sub["stopout_within_h"].astype(bool) & sub["touch_band_high_within_h"].astype(bool)]
                    n_both_bh = int(len(both_bh))
                    n_bh_after = int((both_bh["stop_vs_band_high_order"].astype(str) == "STOP_BEFORE_OTHER").sum())

                    out_md_lines.append(
                        f"- `{scenario}`: stopout={n_stop}/{n} | tp_touch={n_tp}/{n} | band_high_touch={n_bh}/{n}"
                    )
                    out_md_lines.append(f"  - stopout->TP (both): {n_tp_after}/{n_both_tp}")
                    out_md_lines.append(f"  - stopout->band_high (both): {n_bh_after}/{n_both_bh}")
                out_md_lines.append("")

    # Small table preview.
    preview_h = horizons[0] if horizons else int(args.horizon_bars)
    cols = [
        "case_ts",
        "reach_bar_open_ts",
        "entry",
        "tp_current",
        "selected_band_tf",
        "selected_band_low",
        "selected_band_high",
        "selected_tp_current_vs_band_code",
        "hybrid_tp",
        "selected_tp_hybrid_vs_band_code",
        f"h{preview_h}_cur_stopout_within_h",
        f"h{preview_h}_cur_touch_tp_within_h",
        f"h{preview_h}_hybrid_stopout_within_h",
        f"h{preview_h}_hybrid_touch_tp_within_h",
    ]
    have_cols = [c for c in cols if c in df_out.columns]
    if have_cols:
        out_md_lines.append("## Cases (Preview)")
        out_md_lines.append("| " + " | ".join(have_cols) + " |")
        out_md_lines.append("|" + "|".join(["---"] * len(have_cols)) + "|")
        preview = df_out.sort_values(["reach_bar_open_ts", "case_ts"]).head(10)
        for _, r in preview.iterrows():
            out_md_lines.append("| " + " | ".join(str(r.get(c, "")) for c in have_cols) + " |")
        out_md_lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(out_md_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
