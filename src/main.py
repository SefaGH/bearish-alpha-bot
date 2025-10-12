#!/usr/bin/env python3
# Bearish Alpha Bot — Orchestrated MVP
# Flow: load env+config -> build clients+universe -> fetch OHLCV (30m/1h/4h) -> indicators -> regime -> strategies -> telegram+artifacts

from __future__ import annotations
import os, json, time, traceback
from datetime import datetime
from typing import Dict, List
import pandas as pd
import yaml

from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators as ind_enrich
from core.regime import is_bearish_regime
from core.notify import Telegram
from strategies.oversold_bounce import OversoldBounce
from strategies.short_the_rip import ShortTheRip
from core.state import load_state, save_state, load_day_stats, save_day_stats

DATA_DIR = "data"

# ---------- helpers ----------

def load_config():
    cfg_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_tg():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    return Telegram(token, chat_id) if token and chat_id else None

def _df_from_ohlcv(rows):
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
    return df

def fetch_ohlcv(client, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    rows = client.ohlcv(symbol, timeframe=timeframe, limit=limit)
    return _df_from_ohlcv(rows)

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def save_signals_csv(signals: List[dict]):
    ensure_data_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_DIR, f"signals_{ts}.csv")
    pd.DataFrame(signals).to_csv(path, index=False)
    return path

# ---------- main orchestration ----------

def run_once():
    cfg = load_config()
    tg = build_tg()

    # Build clients from ENV (EXCHANGES=BingX,...)
    clients = build_clients_from_env()
    if tg:
        tg.send("🔎 <b>Bearish Alpha Bot</b> tarama başlıyor (paper)")

    # Universe
    from universe import build_universe
    universe = build_universe(clients, cfg)
    # Optionally clamp per exchange top N for fast MVP
    max_per_ex = int(cfg.get("universe", {}).get("top_n_per_exchange", 20) or 20)

    signals_out = []
    for ex_name, client in clients.items():
        symbols = universe.get(ex_name, [])[:max_per_ex]
        if not symbols:
            continue

        # Multi-timeframe fetch
        for sym in symbols:
            try:
                df_30 = fetch_ohlcv(client, sym, "30m", limit=250)
                df_1h = fetch_ohlcv(client, sym, "1h", limit=250)
                df_4h = fetch_ohlcv(client, sym, "4h", limit=250)
                if df_4h.empty or df_1h.empty or df_30.empty:
                    continue

                # indicators
                ind_cfg = cfg.get("indicators", {}) or {}
                df_30i = ind_enrich(df_30, ind_cfg)
                df_1hi = ind_enrich(df_1h, ind_cfg)
                df_4hi = ind_enrich(df_4h, ind_cfg)

                # regime
                ignore_regime = bool(cfg.get("signals", {}).get("oversold_bounce", {}).get("ignore_regime", False))
                bearish_ok = is_bearish_regime(df_4hi)
                if not ignore_regime and not bearish_ok:
                    continue

                # strategies
                s_cfg = cfg.get("signals", {}) or {}
                out_sig = None

                if s_cfg.get("oversold_bounce", {}).get("enable", True):
                    ob = OversoldBounce(s_cfg.get("oversold_bounce"))
                    out_sig = out_sig or ob.signal(df_30i)

                if s_cfg.get("short_the_rip", {}).get("enable", True):
                    strp = ShortTheRip(s_cfg.get("short_the_rip"))
                    out_sig = out_sig or strp.signal(df_30i, df_1hi)

                if out_sig:
                    msg = f"⚡ <b>{ex_name}</b> | <code>{sym}</code> | {out_sig['side'].upper()} — {out_sig['reason']}"
                    if tg: tg.send(msg)
                    signals_out.append({
                        "ts": datetime.utcnow().isoformat(),
                        "exchange": ex_name,
                        "symbol": sym,
                        "side": out_sig.get("side"),
                        "reason": out_sig.get("reason"),
                    })

            except Exception as e:
                # Best-effort logging; do not crash the whole scan
                if tg:
                    tg.send(f"⚠️ Hata: <code>{ex_name}:{sym}</code> — {type(e).__name__}: {str(e)[:140]}")
                else:
                    print("error", ex_name, sym, e)
                continue

    # artifacts
    if signals_out:
        csv_path = save_signals_csv(signals_out)
        if tg:
            tg.send(f"📦 Sinyaller CSV yazıldı: <code>{csv_path}</code>")
    else:
        if tg:
            tg.send("ℹ️ Bu turda sinyal yok.")

    return signals_out

if __name__ == "__main__":
    try:
        out = run_once()
        print(f"✅ Done. signals={len(out)}")
    except Exception as e:
        print("FATAL:", e)
        traceback.print_exc()
        # Telegram varsa kritik hatayı bildir
        tg = build_tg()
        if tg:
            tg.send(f"🛑 FATAL: {type(e).__name__}: {str(e)[:200]}")
