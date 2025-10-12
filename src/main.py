#!/usr/bin/env python3
# Bearish Alpha Bot — Orchestrated MVP (run summary + artifact guarantee)

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
from core.exec_engine import ExecEngine
from core.sizing import position_size_usdt
from core.trailing import initial_stops
from core.limits import clamp_amount, meets_or_scale_notional
from core.normalize import amount_to_precision

DATA_DIR = "data"

def load_config():
    cfg_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    # Apply risk overrides from environment
    risk = cfg.setdefault("risk", {})
    if os.getenv("RISK_EQUITY_USD"):
        risk["equity_usd"] = float(os.getenv("RISK_EQUITY_USD"))
    if os.getenv("RISK_PER_TRADE_RISK_PCT"):
        risk["per_trade_risk_pct"] = float(os.getenv("RISK_PER_TRADE_RISK_PCT"))
    if os.getenv("RISK_RISK_USD_CAP"):
        risk["risk_usd_cap"] = float(os.getenv("RISK_RISK_USD_CAP"))
    if os.getenv("RISK_MAX_NOTIONAL_PER_TRADE"):
        risk["max_notional_per_trade"] = float(os.getenv("RISK_MAX_NOTIONAL_PER_TRADE"))
    if os.getenv("RISK_MIN_STOP_PCT"):
        risk["min_stop_pct"] = float(os.getenv("RISK_MIN_STOP_PCT"))
    if os.getenv("RISK_DAILY_MAX_TRADES"):
        risk["daily_max_trades"] = int(os.getenv("RISK_DAILY_MAX_TRADES"))
    if os.getenv("RISK_MIN_AMOUNT_BEHAVIOR"):
        risk["min_amount_behavior"] = os.getenv("RISK_MIN_AMOUNT_BEHAVIOR")
    if os.getenv("RISK_MIN_NOTIONAL_BEHAVIOR"):
        risk["min_notional_behavior"] = os.getenv("RISK_MIN_NOTIONAL_BEHAVIOR")
    
    return cfg

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
    today = datetime.utcnow().strftime("%Y%m%d")
    path = os.path.join(DATA_DIR, f"signals_{today}.csv")
    pd.DataFrame(signals).to_csv(path, index=False)
    return path

def has_min_bars(*dfs, min_bars: int = 120) -> bool:
    return all(df is not None and len(df) >= min_bars for df in dfs)

def execute_live_order(client, symbol: str, signal: dict, cfg: dict, tg: Telegram | None, last_row: pd.Series):
    """
    Execute a live order based on signal and config.
    Returns: dict with execution result or None if skipped
    """
    try:
        # Get risk parameters
        risk = cfg.get("risk", {})
        equity_usd = float(risk.get("equity_usd", 50))
        per_trade_risk_pct = float(risk.get("per_trade_risk_pct", 0.005))
        risk_usd_cap = float(risk.get("risk_usd_cap", 1.0))
        max_notional = float(risk.get("max_notional_per_trade", 20))
        min_stop_pct = float(risk.get("min_stop_pct", 0.003))
        min_amount_behavior = risk.get("min_amount_behavior", "skip")
        min_notional_behavior = risk.get("min_notional_behavior", "skip")
        
        # Calculate risk USD
        risk_usd = equity_usd * per_trade_risk_pct
        risk_usd = min(risk_usd, risk_usd_cap)
        
        # Get current price and ATR
        entry = float(last_row["close"])
        atr = float(last_row.get("atr", entry * 0.01))
        
        # Calculate stops using core.trailing
        side = signal["side"]
        tp_pct = signal.get("tp_pct", 0.01)
        sl_atr_mult = signal.get("sl_atr_mult", 1.0)
        sl_pct = signal.get("sl_pct")
        
        tp, sl = initial_stops(side, entry, atr, sl_atr_mult, tp_pct)
        
        # Override SL with sl_pct if provided
        if sl_pct is not None:
            if side == "buy":
                sl = entry * (1 - sl_pct)
            else:
                sl = entry * (1 + sl_pct)
        
        # Enforce min stop distance
        stop_dist = abs(entry - sl)
        min_stop_dist = entry * min_stop_pct
        if stop_dist < min_stop_dist:
            sl = entry - min_stop_dist if side == "buy" else entry + min_stop_dist
        
        # Calculate position size
        qty = position_size_usdt(entry, sl, risk_usd, side)
        
        # Apply amount limits
        qty = clamp_amount(client, symbol, qty, min_amount_behavior)
        if qty == 0.0:
            msg = f"⚠️ Live order skipped for {symbol}: amount below minimum"
            if tg:
                tg.send(msg)
            return {"status": "skipped", "reason": "min_amount"}
        
        # Apply notional limits
        qty = meets_or_scale_notional(client, symbol, entry, qty, min_notional_behavior)
        if qty == 0.0:
            msg = f"⚠️ Live order skipped for {symbol}: notional below minimum"
            if tg:
                tg.send(msg)
            return {"status": "skipped", "reason": "min_notional"}
        
        # Check max notional
        notional = entry * qty
        if notional > max_notional:
            qty = max_notional / entry
            qty = clamp_amount(client, symbol, qty, min_amount_behavior)
            if qty == 0.0:
                msg = f"⚠️ Live order skipped for {symbol}: can't satisfy constraints"
                if tg:
                    tg.send(msg)
                return {"status": "skipped", "reason": "constraints"}
        
        # Normalize to exchange precision
        qty_str = amount_to_precision(client, symbol, qty)
        qty_final = float(qty_str)
        
        # Execute via ExecEngine
        exec_cfg = cfg.get("execution", {})
        fee_pct = float(exec_cfg.get("fee_pct", 0.0006))
        slip_pct = float(exec_cfg.get("max_slippage_pct", 0.001))
        
        engine = ExecEngine(mode="live", client=client, fee_pct=fee_pct, slip_pct=slip_pct, tg=tg)
        order = engine.market_order(symbol, side, qty_final)
        
        msg = f"✅ Live order executed: {symbol} {side} {qty_final} @ ~{entry:.2f} (TP: {tp:.2f}, SL: {sl:.2f})"
        if tg:
            tg.send(msg)
        
        return {
            "status": "executed",
            "symbol": symbol,
            "side": side,
            "qty": qty_final,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "order": order
        }
        
    except Exception as e:
        msg = f"❌ Live order failed for {symbol}: {type(e).__name__}: {str(e)[:200]}"
        if tg:
            tg.send(msg)
        return {"status": "error", "error": str(e)}

def run_once():
    # Check mode and configure
    mode = os.getenv("MODE", "paper").lower()
    
    cfg = load_config()
    
    # Force live mode if MODE=live
    if mode == "live":
        cfg.setdefault("execution", {})["enable_live"] = True
    
    # Determine execution exchange for live orders
    from universe import pick_execution_exchange
    exec_exchange = pick_execution_exchange()
    
    enable_live = cfg.get("execution", {}).get("enable_live", False)
    
    tg = build_tg()

    # --- RUN SUMMARY (always create artifact) ---
    ensure_data_dir()
    summary_path = os.path.join(DATA_DIR, "RUN_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Run start (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MODE: {mode}\n")
        f.write(f"EXCHANGES: {os.getenv('EXCHANGES','')}\n")
        f.write(f"EXECUTION_EXCHANGE: {exec_exchange}\n")

    clients = build_clients_from_env()
    
    # Load state for live mode
    state = load_state()
    day_stats = load_day_stats()
    
    mode_label = "live" if enable_live else "paper"
    if tg:
        tg.send(f"🔎 <b>Bearish Alpha Bot</b> tarama başlıyor ({mode_label})")

    from universe import build_universe
    universe = build_universe(clients, cfg)
    max_per_ex = int(cfg.get("universe", {}).get("top_n_per_exchange", 20) or 20)

    signals_out = []
    csv_path = None
    
    # Daily trade limit check
    daily_max = int(cfg.get("risk", {}).get("daily_max_trades", 999))
    trades_today = day_stats.get("signals", 0)

    for ex_name, client in clients.items():
        symbols = universe.get(ex_name, [])[:max_per_ex]
        if not symbols:
            continue

        for sym in symbols:
            try:
                # --- Fetch ---
                df_30 = fetch_ohlcv(client, sym, "30m", limit=250)
                df_1h = fetch_ohlcv(client, sym, "1h",  limit=250)
                df_4h = fetch_ohlcv(client, sym, "4h",  limit=250)

                # --- Data sufficiency: skip thin/empty markets ---
                if not has_min_bars(df_30, df_1h, df_4h, min_bars=120):
                    continue

                # --- Indicators + dropna ---
                ind_cfg = cfg.get("indicators", {}) or {}
                df_30i = ind_enrich(df_30, ind_cfg).dropna()
                df_1hi = ind_enrich(df_1h,  ind_cfg).dropna()
                df_4hi = ind_enrich(df_4h,  ind_cfg).dropna()

                if df_30i.empty or df_1hi.empty or df_4hi.empty:
                    continue

                # --- Regime filter (4h) ---
                ignore_regime = bool(cfg.get("signals", {}).get("oversold_bounce", {}).get("ignore_regime", False))
                bearish_ok = is_bearish_regime(df_4hi)
                if not ignore_regime and not bearish_ok:
                    continue

                # --- Strategies ---
                s_cfg = cfg.get("signals", {}) or {}
                out_sig = None

                if s_cfg.get("oversold_bounce", {}).get("enable", True):
                    ob = OversoldBounce(s_cfg.get("oversold_bounce"))
                    if len(df_30i) >= 50:
                        out_sig = out_sig or ob.signal(df_30i)

                if s_cfg.get("short_the_rip", {}).get("enable", True):
                    strp = ShortTheRip(s_cfg.get("short_the_rip"))
                    if len(df_30i) >= 50 and len(df_1hi) >= 50:
                        out_sig = out_sig or strp.signal(df_30i, df_1hi)

                if out_sig:
                    # Check daily limit
                    if trades_today >= daily_max:
                        if tg:
                            tg.send(f"⚠️ Daily trade limit reached ({daily_max}), skipping signals.")
                        break
                    
                    msg = f"⚡ <b>{ex_name}</b> | <code>{sym}</code> | {out_sig['side'].upper()} — {out_sig['reason']}"
                    if tg: 
                        tg.send(msg)
                    
                    sig_record = {
                        "ts": datetime.utcnow().isoformat(),
                        "exchange": ex_name,
                        "symbol": sym,
                        "side": out_sig.get("side"),
                        "reason": out_sig.get("reason"),
                    }
                    
                    # Execute live order only on execution exchange
                    if enable_live and ex_name == exec_exchange:
                        last_row = df_30i.iloc[-1]
                        exec_result = execute_live_order(client, sym, out_sig, cfg, tg, last_row)
                        sig_record["execution"] = exec_result
                        
                        # Update state and day stats
                        if exec_result and exec_result.get("status") == "executed":
                            state["open"][sym] = {
                                "entry_time": sig_record["ts"],
                                "side": out_sig["side"],
                                "entry": exec_result.get("entry"),
                                "tp": exec_result.get("tp"),
                                "sl": exec_result.get("sl"),
                                "qty": exec_result.get("qty"),
                            }
                            save_state(state)
                            
                            day_stats["signals"] = day_stats.get("signals", 0) + 1
                            save_day_stats(day_stats)
                            trades_today += 1
                    
                    signals_out.append(sig_record)

            except Exception as e:
                if tg:
                    tg.send(f"⚠️ {ex_name}:{sym} skip — {type(e).__name__}: {str(e)[:140]}")
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

    # --- Append run summary
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"Total signals: {len(signals_out)}\n")
            if csv_path:
                f.write(f"CSV: {csv_path}\n")
    except Exception:
        pass

    return signals_out

if __name__ == "__main__":
    try:
        out = run_once()
        print(f"✅ Done. signals={len(out)}")
    except Exception as e:
        print("FATAL:", e)
        traceback.print_exc()
        tg = build_tg()
        if tg:
            tg.send(f"🛑 FATAL: {type(e).__name__}: {str(e)[:200]}")