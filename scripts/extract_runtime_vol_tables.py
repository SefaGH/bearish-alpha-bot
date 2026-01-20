"""Extract runtime volatility telemetry tables from a live trading log.

Goal: reproduce Faz-0-style C/D/E summary tables from runtime events.

Inputs:
- SIGNAL_BREAKDOWN lines (StrategyCoordinator or ProductionCoordinator)
- TRADE_CLOSED lines (PositionManager)

Outputs (under reports/ by default):
- runtime_vol_estimator_trades_<run_id>.csv
- runtime_vol_estimator_bucket_summary_<run_id>.csv
- runtime_vol_estimator_report_<run_id>.md

This is intentionally best-effort and fail-closed: missing fields simply yield NaNs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


RUN_ID_RE = re.compile(r"live_trading_(\d{8}_\d{6}_\d+)\.log$")


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _extract_json_after_marker(line: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = line.find(marker)
    if idx < 0:
        return None
    payload = line[idx + len(marker) :].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _infer_run_id(log_path: Path) -> str:
    m = RUN_ID_RE.search(log_path.name)
    return m.group(1) if m else log_path.stem


def _compute_stop_bps(entry_price: Optional[float], stop_price: Optional[float]) -> Optional[float]:
    if not entry_price or not stop_price:
        return None
    if entry_price <= 0 or stop_price <= 0:
        return None
    return abs(stop_price - entry_price) / entry_price * 10_000.0


@dataclass
class TradeRow:
    run_id: str
    trade_id: Optional[str]
    symbol: Optional[str]
    strategy: Optional[str]
    side: Optional[str]
    timeframe: Optional[str]
    volume_bucket: Optional[str]
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    stop_bps_effective: Optional[float]
    vol_std_bps: Optional[float]
    vol_atr_bps: Optional[float]
    vol_rs_bps: Optional[float]
    vol_gk_bps: Optional[float]
    vol_yz_bps: Optional[float]


def _iter_trade_rows(log_path: Path) -> Iterable[TradeRow]:
    run_id_from_name = _infer_run_id(log_path)

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            payload = _extract_json_after_marker(line, "TRADE_CLOSED ")
            if not payload:
                continue

            run_id = str(payload.get("run_id") or run_id_from_name)
            entry_md = payload.get("entry_metadata")
            entry_md = entry_md if isinstance(entry_md, dict) else {}
            entry_inds = entry_md.get("entry_indicators")
            entry_inds = entry_inds if isinstance(entry_inds, dict) else {}
            entry_lvls = entry_md.get("entry_levels")
            entry_lvls = entry_lvls if isinstance(entry_lvls, dict) else {}

            entry_price = _safe_float(payload.get("entry_price"))
            stop_price = _safe_float(entry_lvls.get("stop_price"))
            target_price = _safe_float(entry_lvls.get("target_price"))

            yield TradeRow(
                run_id=run_id,
                trade_id=payload.get("trade_id"),
                symbol=payload.get("symbol"),
                strategy=payload.get("strategy") or payload.get("strategy_name"),
                side=str(payload.get("side")).lower() if payload.get("side") is not None else None,
                timeframe=payload.get("timeframe"),
                volume_bucket=payload.get("volume_bucket_at_entry"),
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                stop_bps_effective=_compute_stop_bps(entry_price, stop_price),
                vol_std_bps=_safe_float(entry_inds.get("vol_std_bps")),
                vol_atr_bps=_safe_float(entry_inds.get("vol_atr_bps")),
                vol_rs_bps=_safe_float(entry_inds.get("vol_rs_bps")),
                vol_gk_bps=_safe_float(entry_inds.get("vol_gk_bps")),
                vol_yz_bps=_safe_float(entry_inds.get("vol_yz_bps")),
            )


def _bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["volume_bucket"] = df["volume_bucket"].fillna("")

    estimators = ["vol_std_bps", "vol_atr_bps", "vol_rs_bps", "vol_gk_bps", "vol_yz_bps"]

    def q(series: pd.Series, p: float) -> float:
        return float(series.quantile(p))

    rows: List[Dict[str, Any]] = []
    for (run_id, bucket), g in df.groupby(["run_id", "volume_bucket"], dropna=False):
        row: Dict[str, Any] = {"run_id": run_id, "volume_bucket": bucket if bucket != "" else None}

        for col in estimators:
            s = g[col].dropna()
            row[f"{col}_n_x"] = int(s.shape[0])
            row[f"{col}_p10"] = q(s, 0.10) if not s.empty else None
            row[f"{col}_p50"] = q(s, 0.50) if not s.empty else None
            row[f"{col}_p90"] = q(s, 0.90) if not s.empty else None

        stop = g["stop_bps_effective"].dropna()
        row["stop_bps_n"] = int(stop.shape[0])
        row["micro_stop_lt15_rate"] = float((stop < 15).mean()) if not stop.empty else None

        for col in estimators:
            s = g[col].dropna()
            row[f"{col}_n_y"] = int(s.shape[0])
            row[f"{col}_lt15_rate"] = float((s < 15).mean()) if not s.empty else None

            med_vol = row.get(f"{col}_p50")
            if med_vol and med_vol > 0:
                row[f"k_needed_to15_{col}_p50"] = 15.0 / float(med_vol)
            else:
                row[f"k_needed_to15_{col}_p50"] = None

            # k implied = median(stop / vol)
            if not stop.empty and not s.empty:
                aligned = g[["stop_bps_effective", col]].dropna()
                if not aligned.empty:
                    implied = aligned["stop_bps_effective"] / aligned[col]
                    row[f"k_implied_{col}_p50"] = float(implied.median())
                else:
                    row[f"k_implied_{col}_p50"] = None
            else:
                row[f"k_implied_{col}_p50"] = None

        # correlations
        for col in estimators:
            aligned = g[["stop_bps_effective", col]].dropna()
            if aligned.shape[0] >= 3:
                row[f"corr_stop_vs_{col}"] = float(aligned["stop_bps_effective"].corr(aligned[col]))
            else:
                row[f"corr_stop_vs_{col}"] = None

        rows.append(row)

    out = pd.DataFrame(rows)

    # Column ordering to match earlier Faz-0 summaries (loosely)
    desired = [
        "run_id",
        "volume_bucket",
        "vol_std_bps_n_x",
        "vol_std_bps_p10",
        "vol_std_bps_p50",
        "vol_std_bps_p90",
        "vol_atr_bps_n_x",
        "vol_atr_bps_p10",
        "vol_atr_bps_p50",
        "vol_atr_bps_p90",
        "vol_rs_bps_n_x",
        "vol_rs_bps_p10",
        "vol_rs_bps_p50",
        "vol_rs_bps_p90",
        "vol_gk_bps_n_x",
        "vol_gk_bps_p10",
        "vol_gk_bps_p50",
        "vol_gk_bps_p90",
        "vol_yz_bps_n_x",
        "vol_yz_bps_p10",
        "vol_yz_bps_p50",
        "vol_yz_bps_p90",
        "stop_bps_n",
        "micro_stop_lt15_rate",
        "vol_std_bps_n_y",
        "vol_std_bps_lt15_rate",
        "k_implied_vol_std_bps_p50",
        "k_needed_to15_vol_std_bps_p50",
        "vol_atr_bps_n_y",
        "vol_atr_bps_lt15_rate",
        "k_implied_vol_atr_bps_p50",
        "k_needed_to15_vol_atr_bps_p50",
        "vol_rs_bps_n_y",
        "vol_rs_bps_lt15_rate",
        "k_implied_vol_rs_bps_p50",
        "k_needed_to15_vol_rs_bps_p50",
        "vol_gk_bps_n_y",
        "vol_gk_bps_lt15_rate",
        "k_implied_vol_gk_bps_p50",
        "k_needed_to15_vol_gk_bps_p50",
        "vol_yz_bps_n_y",
        "vol_yz_bps_lt15_rate",
        "k_implied_vol_yz_bps_p50",
        "k_needed_to15_vol_yz_bps_p50",
        "corr_stop_vs_vol_std_bps",
        "corr_stop_vs_vol_atr_bps",
        "corr_stop_vs_vol_rs_bps",
        "corr_stop_vs_vol_gk_bps",
        "corr_stop_vs_vol_yz_bps",
    ]

    # Map internal names to desired names
    out = out.rename(
        columns={
            "vol_std_bps_lt15_rate": "vol_std_bps_lt15_rate",
            "vol_atr_bps_lt15_rate": "vol_atr_bps_lt15_rate",
            "vol_rs_bps_lt15_rate": "vol_rs_bps_lt15_rate",
            "vol_gk_bps_lt15_rate": "vol_gk_bps_lt15_rate",
            "vol_yz_bps_lt15_rate": "vol_yz_bps_lt15_rate",
            "corr_stop_vs_vol_std_bps": "corr_stop_vs_vol_std_bps",
            "corr_stop_vs_vol_atr_bps": "corr_stop_vs_vol_atr_bps",
            "corr_stop_vs_vol_rs_bps": "corr_stop_vs_vol_rs_bps",
            "corr_stop_vs_vol_gk_bps": "corr_stop_vs_vol_gk_bps",
            "corr_stop_vs_vol_yz_bps": "corr_stop_vs_vol_yz_bps",
        }
    )

    # Ensure all desired columns exist
    for c in desired:
        if c not in out.columns:
            out[c] = None

    return out[desired]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract runtime vol estimator tables from a live trading log")
    parser.add_argument("log", type=str, help="Path to live_trading_*.log")
    parser.add_argument("--out", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = _infer_run_id(log_path)

    trades = list(_iter_trade_rows(log_path))
    df_trades = pd.DataFrame([t.__dict__ for t in trades])

    trades_csv = out_dir / f"runtime_vol_estimator_trades_{run_id}.csv"
    df_trades.to_csv(trades_csv, index=False)

    df_summary = _bucket_summary(df_trades)
    summary_csv = out_dir / f"runtime_vol_estimator_bucket_summary_{run_id}.csv"
    df_summary.to_csv(summary_csv, index=False)

    md_path = out_dir / f"runtime_vol_estimator_report_{run_id}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Runtime Volatility Telemetry Summary\n\n")
        f.write(f"Log: `{log_path.name}`\n\n")
        f.write(f"Run: `{run_id}`\n\n")
        f.write("## Coverage\n\n")
        f.write(f"- Closed trades parsed: {len(df_trades)}\n")
        if not df_trades.empty:
            cov = {
                k: int(df_trades[k].notna().sum())
                for k in ["vol_std_bps", "vol_atr_bps", "vol_rs_bps", "vol_gk_bps", "vol_yz_bps", "stop_bps_effective"]
                if k in df_trades.columns
            }
            f.write("- Non-null counts: " + ", ".join([f"{k}={v}" for k, v in cov.items()]) + "\n")
        f.write("\n")

        f.write("## C/D/E Bucket Summary (best-effort)\n\n")
        if df_summary.empty:
            f.write("No closed trades (or no parsable telemetry) in this log.\n")
        else:
            f.write(df_summary.to_markdown(index=False))
            f.write("\n")

    print(f"Wrote: {trades_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
