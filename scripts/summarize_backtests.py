#!/usr/bin/env python3
# Summarize backtest CSVs into a Markdown report and (optionally) send a Telegram brief.
from __future__ import annotations
import os, re, glob, json, math
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import requests

BT_DIR = os.getenv("BT_DIR", "data/backtests")
OUT_MD = os.getenv("OUT_MD", "data/backtests/REPORT.md")

def infer_symbol_from_filename(fname: str) -> str:
    base = os.path.basename(fname).replace(".csv","")
    # forms: "str_BTC_USDT_YYYY..." or "BTC_USDT_YYYY..."
    parts = base.split("_")
    if parts and parts[0].lower() == "str":
        parts = parts[1:]  # drop 'str'
    # consume until last 1-2 chunks are timestamp-like
    # we'll join until we hit a numeric-like chunk length>=8
    acc = []
    for p in parts:
        if p.isdigit() and len(p) >= 8:
            break
        acc.append(p)
    symbol = "/".join(["_".join(acc).split("_")[0], "_".join(acc).split("_")[1]]) if len(acc)>=2 else "_".join(acc)
    symbol = symbol.replace("__","_").replace("_","/")
    return symbol if symbol else "UNKNOWN"

def load_all() -> pd.DataFrame:
    files = glob.glob(os.path.join(BT_DIR, "*.csv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            if "strategy" not in df.columns:
                # fallback by filename
                df["strategy"] = ["short_the_rip" if os.path.basename(f).startswith("str_") else "oversold_bounce"] * len(df)
            df["symbol"] = infer_symbol_from_filename(f)
            rows.append(df)
        except Exception as e:
            print("skip file:", f, e)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def top_k(df: pd.DataFrame, k=3) -> pd.DataFrame:
    return df.sort_values(["avg_pnl","win_rate","trades","rr"], ascending=[False, False, False, False]).head(k)

def make_report(df: pd.DataFrame) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Backtest Nightly Report\n", f"_Generated: {ts}_\n"]
    for strat in sorted(df["strategy"].unique()):
        sdf = df[df["strategy"]==strat]
        lines.append(f"\n## Strategy: **{strat}**\n")
        for sym in sorted(sdf["symbol"].unique()):
            ss = sdf[sdf["symbol"]==sym]
            if ss.empty: 
                continue
            lines.append(f"\n### {sym}\n")
            top = top_k(ss, 5).copy()
            # reorder useful cols
            cols = [c for c in ["rsi_max","rsi_min","tp_pct","sl_atr_mult","require_band_touch","require_ema_align","trades","win_rate","avg_pnl","rr","net_pnl","source_file"] if c in top.columns]
            lines.append(top[cols].to_markdown(index=False))
            lines.append("")
    return "\n".join(lines)

def save_report(md: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path

def maybe_telegram(md: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    chat  = os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat:
        return
    # brief: extract first top lines per strategy
    brief = []
    for line in md.splitlines():
        if line.startswith("## Strategy:") or line.startswith("### "):
            brief.append(line.replace("**",""))
        if len(brief) > 12:
            break
    text = "\n".join(brief[:12])
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text or "Nightly backtest report ready.", "parse_mode":"HTML"},
                      timeout=10)
    except Exception as e:
        print("telegram send failed:", e)

def main():
    df = load_all()
    if df.empty:
        print("No backtest CSVs found in", BT_DIR)
        return
    md = make_report(df)
    out = save_report(md, OUT_MD)
    print("Wrote", out)
    maybe_telegram(md)

if __name__ == "__main__":
    main()