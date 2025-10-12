#!/usr/bin/env python3
# Param Sweep (MVP) for OversoldBounce (30m)
# - Fetches 30m OHLCV via ccxt using CcxtClient
# - Enriches with indicators
# - Runs a grid over (rsi_max, tp_pct, sl_atr_mult) and simulates next-candle TP/SL
# - Writes results to data/backtests/<symbol>_<ts>.csv

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os, itertools
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import yaml

from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators as ind_enrich

DATA_DIR = "data"
BT_DIR = os.path.join(DATA_DIR, "backtests")

def _df_from_ohlcv(rows):
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")

def simulate_long_nextbar(df: pd.DataFrame, tp_pct: float, sl_atr_mult: float | None, fallback_sl_pct: float | None) -> Dict[str, float]:
    pnls = []
    df = df.dropna().copy()
    for i in range(len(df)-1):
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i])
        hi = float(df["high"].iloc[i+1])
        lo = float(df["low"].iloc[i+1])

        tp = entry * (1 + tp_pct)
        if sl_atr_mult is not None:
            sl = entry - sl_atr_mult * atr
        else:
            sl_pct = float(fallback_sl_pct) if fallback_sl_pct is not None else 0.05
            sl = entry * (1 - sl_pct)

        if lo <= sl:
            pnls.append(-(entry - sl)/entry)
            continue
        if hi >= tp:
            pnls.append((tp - entry)/entry)
            continue
    if not pnls:
        return {"trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "rr": 0.0, "net_pnl": 0.0}
    wins = [x for x in pnls if x>0]
    losses = [-x for x in pnls if x<0]
    win_rate = len(wins)/len(pnls)
    avg_gain = sum(wins)/max(1,len(wins))
    avg_loss = sum(losses)/max(1,len(losses))
    rr = (avg_gain/avg_loss) if (avg_gain>0 and avg_loss>0) else 0.0
    return {"trades": len(pnls), "win_rate": win_rate, "avg_pnl": sum(pnls)/len(pnls), "rr": rr, "net_pnl": sum(pnls)}

def sweep_ob(df30: pd.DataFrame, grid: Dict[str, List[Any]], fallbacks: Dict[str, Any]) -> pd.DataFrame:
    res = []
    for rsi_max, tp_pct, sl_atr_mult in itertools.product(grid["rsi_max"], grid["tp_pct"], grid["sl_atr_mult"]):
        mask = df30["rsi"] <= rsi_max
        sub = df30.loc[mask].copy()
        if len(sub) < 5:
            continue
        sim = simulate_long_nextbar(sub, tp_pct=float(tp_pct), sl_atr_mult=(None if sl_atr_mult is None else float(sl_atr_mult)), fallback_sl_pct=fallbacks.get("sl_pct"))
        res.append({
            "strategy": "oversold_bounce",
            "rsi_max": float(rsi_max),
            "tp_pct": float(tp_pct),
            "sl_atr_mult": (None if sl_atr_mult is None else float(sl_atr_mult)),
            **sim
        })
    if not res:
        return pd.DataFrame()
    return pd.DataFrame(res).sort_values(["avg_pnl","win_rate","trades"], ascending=[False, False, False])

def main():
    symbol = os.getenv("BT_SYMBOL", "BTC/USDT")
    exchange = os.getenv("BT_EXCHANGE", os.getenv("EXECUTION_EXCHANGE", "kucoinfutures"))
    limit = int(os.getenv("BT_LIMIT", "1000"))
    grid = {
        "rsi_max": [20, 23, 25, 27, 30],
        "tp_pct": [0.008, 0.010, 0.012, 0.015],
        "sl_atr_mult": [0.8, 1.0, 1.2, 1.5]
    }
    cfg_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    ind_cfg = cfg.get("indicators", {}) or {}
    ob_cfg = (cfg.get("signals", {}) or {}).get("oversold_bounce", {}) or {}
    fallback_sl_pct = ob_cfg.get("sl_pct")

    clients = build_clients_from_env()
    if exchange not in clients:
        if clients:
            exchange, client = next(iter(clients.items()))
        else:
            raise SystemExit("No exchange available. Set EXCHANGES=... in ENV")
    else:
        client = clients[exchange]

    # Validate and get the correct symbol format for this exchange
    validated_symbol = client.validate_and_get_symbol(symbol)

    rows = client.ohlcv(validated_symbol, timeframe="30m", limit=limit)
    df30 = _df_from_ohlcv(rows)
    df30i = ind_enrich(df30, ind_cfg).dropna()

    dfres = sweep_ob(df30i, grid, {"sl_pct": fallback_sl_pct})
    if dfres.empty:
        print("No results produced. Check data length or grid ranges.")
        return

    os.makedirs(BT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(BT_DIR, f"{symbol.replace('/','_')}_{ts}.csv")
    dfres.to_csv(out_path, index=False)
    print(f"✅ Wrote: {out_path}")
    print(dfres.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
