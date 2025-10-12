#!/usr/bin/env python3
# Param Sweep (MVP) for ShortTheRip (30m entries with 1h context)
# - Fetches 30m and 1h OHLCV via CcxtClient
# - Enriches both with indicators
# - Entry when 30m RSI >= rsi_min and optional EMA alignment & "band touch" proxy
# - Simulates next-candle short TP/SL
# - Writes results to data/backtests/str_<symbol>_<ts>.csv

from __future__ import annotations
import os, itertools
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import yaml

from core.multi_exchange import build_clients_from_env
from core.indicators import enrich as ind_enrich

DATA_DIR = "data"
BT_DIR = os.path.join(DATA_DIR, "backtests")

def _df_from_ohlcv(rows):
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")

def fetch(client, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    rows = client.ohlcv(symbol, timeframe=tf, limit=limit)
    return _df_from_ohlcv(rows)

def align_1h_to_30m(df30: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    # join-asof: for each 30m ts take the last 1h bar <= ts
    joined = pd.merge_asof(
        df30.sort_index(),
        df1h.sort_index(),
        left_index=True, right_index=True,
        direction="backward", suffixes=("", "_1h")
    )
    return joined

def simulate_short_nextbar(df: pd.DataFrame, tp_pct: float, sl_atr_mult: float | None, fallback_sl_pct: float | None) -> Dict[str, float]:
    entries, pnls = 0, []
    df = df.dropna().copy()
    for i in range(len(df)-1):
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i])
        hi = float(df["high"].iloc[i+1])
        lo = float(df["low"].iloc[i+1])

        tp = entry * (1 - tp_pct)
        if sl_atr_mult is not None:
            sl = entry + sl_atr_mult * atr
        else:
            sl_pct = float(fallback_sl_pct) if fallback_sl_pct is not None else 0.05
            sl = entry * (1 + sl_pct)

        if hi >= sl:
            pnls.append(-(sl - entry)/entry)
            entries += 1
            continue
        if lo <= tp:
            pnls.append((entry - tp)/entry)
            entries += 1
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

def sweep_str(df30i: pd.DataFrame, df1hi: pd.DataFrame, grid, fallbacks) -> pd.DataFrame:
    # align 1h columns into 30m frame
    dfj = align_1h_to_30m(df30i, df1hi)

    res = []
    for rsi_min, tp_pct, sl_atr_mult, require_band_touch, require_ema_align in itertools.product(
        grid["rsi_min"], grid["tp_pct"], grid["sl_atr_mult"], grid["require_band_touch"], grid["require_ema_align"]
    ):
        # base condition: RSI on 30m
        mask = dfj["rsi"] >= rsi_min

        # EMA alignment on 30m (ema21 < ema50 <= ema200)
        if require_ema_align:
            mask = mask & (dfj["ema21"] < dfj["ema50"]) & (dfj["ema50"] <= dfj["ema200"])

        # "band touch" proxy: price touches or exceeds ema50 on 1h (or 30m as fallback)
        if require_band_touch:
            if "ema50_1h" in dfj.columns:
                mask = mask & (dfj["close_1h"] >= dfj["ema50_1h"])
            else:
                mask = mask & (dfj["close"] >= dfj["ema50"])

        sub = dfj.loc[mask].copy()
        if len(sub) < 5:
            continue
        sim = simulate_short_nextbar(sub, tp_pct=float(tp_pct), sl_atr_mult=(None if sl_atr_mult is None else float(sl_atr_mult)), fallback_sl_pct=fallbacks.get("sl_pct"))
        res.append({
            "strategy": "short_the_rip",
            "rsi_min": float(rsi_min),
            "tp_pct": float(tp_pct),
            "sl_atr_mult": (None if sl_atr_mult is None else float(sl_atr_mult)),
            "require_band_touch": bool(require_band_touch),
            "require_ema_align": bool(require_ema_align),
            **sim
        })
    if not res:
        return pd.DataFrame()
    return pd.DataFrame(res).sort_values(["avg_pnl", "win_rate", "trades"], ascending=[False, False, False])

def main():
    symbol = os.getenv("BT_SYMBOL", "BTC/USDT")
    exchange = os.getenv("BT_EXCHANGE", os.getenv("EXECUTION_EXCHANGE", "bingx"))
    limit30 = int(os.getenv("BT_LIMIT_30M", "1000"))
    limit1h = int(os.getenv("BT_LIMIT_1H", "1000"))

    grid = {
        "rsi_min": [58, 60, 62, 64],
        "tp_pct": [0.008, 0.010, 0.012, 0.015],
        "sl_atr_mult": [1.0, 1.2, 1.5],
        "require_band_touch": [True, False],
        "require_ema_align": [True, False],
    }

    cfg_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    ind_cfg = cfg.get("indicators", {}) or {}
    # No direct STR config needed; we just sweep

    clients = build_clients_from_env()
    if exchange not in clients:
        if clients:
            exchange, client = next(iter(clients.items()))
        else:
            raise SystemExit("No exchange available. Set EXCHANGES=... in ENV")
    else:
        client = clients[exchange]

    # fetch data
    df30 = fetch(client, symbol, "30m", limit30)
    df1h = fetch(client, symbol, "1h", limit1h)

    # enrich
    df30i = ind_enrich(df30, ind_cfg).dropna()
    df1hi = ind_enrich(df1h, ind_cfg).dropna()

    # sweep
    dfres = sweep_str(df30i, df1hi, grid, {"sl_pct": None})
    if dfres.empty:
        print("No results produced. Check data length or grid ranges.")
        return

    # write CSV
    os.makedirs(BT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(BT_DIR, f"str_{symbol.replace('/','_')}_{ts}.csv")
    dfres.to_csv(out_path, index=False)
    print(f"✅ Wrote: {out_path}")
    print(dfres.head(10).to_string(index=False))

if __name__ == "__main__":
    main()