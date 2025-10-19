#!/usr/bin/env python3
# Bearish Alpha Bot — Orchestrated MVP with Adaptive Strategies

from __future__ import annotations
import os, json, time, traceback, logging
from datetime import datetime, timezone
from typing import Dict, List
import pandas as pd
import yaml

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators as ind_enrich
from core.regime import is_bearish_regime
from core.notify import Telegram

# Base strategies (mevcut)
from strategies.oversold_bounce import OversoldBounce
from strategies.short_the_rip import ShortTheRip

# ADAPTIVE STRATEGIES (YENİ)
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from core.adaptive_monitor import adaptive_monitor

from core.state import load_state, save_state, load_day_stats, save_day_stats
from core.trailing import initial_stops
from core.sizing import position_size_usdt
from core.limits import clamp_amount, meets_or_scale_notional
from core.normalize import amount_to_precision
from core.exec_engine import ExecEngine

DATA_DIR = "data"
logger = logging.getLogger(__name__)

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
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_DIR, f"signals_{ts}.csv")
    pd.DataFrame(signals).to_csv(path, index=False)
    return path

def has_min_bars(*dfs, min_bars: int = 120) -> bool:
    return all(df is not None and len(df) >= min_bars for df in dfs)

def get_risk_params():
    """Extract risk parameters from config file first, then env vars."""
    cfg = load_config()
    risk_cfg = cfg.get('risk', {})
    
    return {
        'equity_usd': float(
            risk_cfg.get('equity_usd') or 
            os.getenv('RISK_EQUITY_USD', '100')
        ),
        'per_trade_risk_pct': float(
            risk_cfg.get('per_trade_risk_pct') or 
            os.getenv('RISK_PER_TRADE_RISK_PCT', '0.01')
        ),
        'risk_usd_cap': float(
            risk_cfg.get('risk_usd_cap') or 
            os.getenv('RISK_RISK_USD_CAP', '5')
        ),
        'max_notional_per_trade': float(
            risk_cfg.get('max_notional_per_trade') or 
            os.getenv('RISK_MAX_NOTIONAL_PER_TRADE', '20')
        ),
        'min_stop_pct': float(
            risk_cfg.get('min_stop_pct') or 
            os.getenv('RISK_MIN_STOP_PCT', '0.003')
        ),
        'daily_max_trades': int(
            risk_cfg.get('daily_max_trades') or 
            os.getenv('RISK_DAILY_MAX_TRADES', '5')
        ),
        'min_amount_behavior': (
            risk_cfg.get('min_amount_behavior') or 
            os.getenv('RISK_MIN_AMOUNT_BEHAVIOR', 'skip')
        ),
        'min_notional_behavior': (
            risk_cfg.get('min_notional_behavior') or 
            os.getenv('RISK_MIN_NOTIONAL_BEHAVIOR', 'skip')
        ),
    }

def execute_signal(engine: ExecEngine, client, symbol: str, signal: dict, risk_params: dict, tg: Telegram | None):
    """Execute a signal with proper sizing and risk management."""
    try:
        # Get latest price
        ticker = client.ticker(symbol)
        price = float(ticker.get('last') or ticker.get('close', 0))
        if price <= 0:
            if tg: tg.send(f"⚠️ {symbol}: invalid price {price}, skipping")
            return None
        
        # Get ATR from signal metadata (if available)
        atr = signal.get('atr', price * 0.02)
        
        # Calculate stops
        side = signal.get('side', 'buy')
        tp_pct = signal.get('tp_pct', 0.015)
        sl_atr_mult = signal.get('sl_atr_mult', 1.2)
        tp, sl = initial_stops(side, price, atr, sl_atr_mult, tp_pct)
        
        # Ensure minimum stop distance
        stop_dist_pct = abs(sl - price) / price
        if stop_dist_pct < risk_params['min_stop_pct']:
            if tg: tg.send(f"⚠️ {symbol}: stop too tight {stop_dist_pct:.4f} < {risk_params['min_stop_pct']}, skipping")
            return None
        
        # Calculate position size (adaptive multiplier varsa kullan)
        position_multiplier = signal.get('position_multiplier', 1.0)
        base_risk_usd = min(
            risk_params['equity_usd'] * risk_params['per_trade_risk_pct'],
            risk_params['risk_usd_cap']
        )
        risk_usd = base_risk_usd * position_multiplier
        
        qty = position_size_usdt(price, sl, risk_usd, side)
        
        # Check max notional
        notional = price * qty
        if notional > risk_params['max_notional_per_trade']:
            qty = risk_params['max_notional_per_trade'] / price
            if tg: tg.send(f"ℹ️ {symbol}: capped to max notional, qty={qty:.6f}")
        
        # Apply lot size limits
        qty = clamp_amount(client, symbol, qty, behavior=risk_params['min_amount_behavior'])
        if qty <= 0:
            if tg: tg.send(f"⚠️ {symbol}: qty below min lot size, skipping")
            return None
        
        # Apply notional limits
        qty = meets_or_scale_notional(client, symbol, price, qty, behavior=risk_params['min_notional_behavior'])
        if qty <= 0:
            if tg: tg.send(f"⚠️ {symbol}: notional below minimum, skipping")
            return None
        
        # Precision normalize
        qty_str = amount_to_precision(client, symbol, qty)
        qty_final = float(qty_str)
        
        # Execute order
        order_side = 'sell' if side == 'sell' else 'buy'
        order = engine.market_order(symbol, order_side, qty_final)
        
        # Adaptive veya base strategy?
        strategy_type = signal.get('strategy_type', 'base')
        msg = f"✅ <b>EXECUTED ({strategy_type})</b> {symbol} {order_side.upper()} qty={qty_final:.6f} @ ~{price:.4f} | TP={tp:.4f} SL={sl:.4f}"
        if tg:
            tg.send(msg)
        
        return {
            'symbol': symbol,
            'side': side,
            'qty': qty_final,
            'price': price,
            'tp': tp,
            'sl': sl,
            'strategy_type': strategy_type,
            'order_id': order.get('id', 'N/A'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        if tg:
            tg.send(f"❌ <b>EXEC FAILED</b> {symbol}: {type(e).__name__}: {str(e)[:150]}")
        traceback.print_exc()
        return None

def run_once():
    cfg = load_config()
    tg = build_tg()
    
    # Adaptive strategies enable flag
    use_adaptive = cfg.get('adaptive_strategies', {}).get('enable', True)
    
    # Mode
    mode = os.getenv('MODE', 'paper').lower()
    is_live = (mode == 'live')
    
    # Execution exchange
    from universe import pick_execution_exchange
    exec_exchange = pick_execution_exchange()

    # RUN SUMMARY
    ensure_data_dir()
    summary_path = os.path.join(DATA_DIR, "RUN_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Run start (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MODE: {mode}\n")
        f.write(f"ADAPTIVE: {use_adaptive}\n")
        f.write(f"EXECUTION_EXCHANGE: {exec_exchange}\n")
        f.write(f"EXCHANGES: {os.getenv('EXCHANGES','')}\n")

    clients = build_clients_from_env()
    
    # Initialize execution engine if live mode
    exec_engine = None
    risk_params = None
    if is_live:
        if exec_exchange not in clients:
            raise SystemExit(f"Live mode requires EXECUTION_EXCHANGE={exec_exchange} in clients")
        exec_client = clients[exec_exchange]
        fee_pct = float(os.getenv('EXEC_FEE_PCT', '0.0006'))
        slip_pct = float(os.getenv('EXEC_SLIP_PCT', '0.0005'))
        exec_engine = ExecEngine('live', exec_client, fee_pct, slip_pct, tg)
        risk_params = get_risk_params()
        if tg:
            strategy_mode = "ADAPTIVE" if use_adaptive else "BASE"
            tg.send(f"🚀 <b>LIVE MODE ({strategy_mode})</b> activated on {exec_exchange} | Risk: {risk_params['risk_usd_cap']}$ cap")
    else:
        if tg:
            strategy_mode = "adaptive" if use_adaptive else "base"
            tg.send(f"🔎 <b>Bearish Alpha Bot ({strategy_mode})</b> tarama başlıyor (paper)")

    from universe import build_universe
    universe = build_universe(clients, cfg)
    max_per_ex = int(cfg.get("universe", {}).get("top_n_per_exchange", 20) or 20)

    signals_out = []
    executions = []
    csv_path = None
    
    # Stats
    day_stats = load_day_stats() if is_live else None
    trades_today = day_stats.get('signals', 0) if day_stats else 0
    
    # Adaptive stats
    adaptive_signals = 0
    base_signals = 0

    for ex_name, client in clients.items():
        symbols = universe.get(ex_name, [])[:max_per_ex]
        if not symbols:
            continue

        for sym in symbols:
            try:
                # Check daily trade limit
                if is_live and trades_today >= risk_params['daily_max_trades']:
                    if tg:
                        tg.send(f"⚠️ Daily trade limit ({risk_params['daily_max_trades']}) reached")
                    break
                
                # Fetch data
                df_30 = fetch_ohlcv(client, sym, "30m", limit=250)
                df_1h = fetch_ohlcv(client, sym, "1h",  limit=250)
                df_4h = fetch_ohlcv(client, sym, "4h",  limit=250)

                # Data sufficiency
                if not has_min_bars(df_30, df_1h, df_4h, min_bars=120):
                    continue

                # Indicators
                ind_cfg = cfg.get("indicators", {}) or {}
                df_30i = ind_enrich(df_30, ind_cfg).dropna()
                df_1hi = ind_enrich(df_1h,  ind_cfg).dropna()
                df_4hi = ind_enrich(df_4h,  ind_cfg).dropna()

                if df_30i.empty or df_1hi.empty or df_4hi.empty:
                    continue

                # Regime filter
                ignore_regime = bool(cfg.get("signals", {}).get("oversold_bounce", {}).get("ignore_regime", False))
                bearish_ok = is_bearish_regime(df_4hi)
                if not ignore_regime and not bearish_ok:
                    continue

                # === STRATEGY SELECTION ===
                s_cfg = cfg.get("signals", {}) or {}
                out_sig = None
                
                # ADAPTIVE veya BASE stratejileri kullan
                if use_adaptive:
                    # === ADAPTIVE OVERSOLD BOUNCE ===
                    if s_cfg.get("oversold_bounce", {}).get("enable", True):
                        adaptive_ob = AdaptiveOversoldBounce(s_cfg.get("oversold_bounce"))
                        if len(df_30i) >= 50:
                            out_sig = out_sig or adaptive_ob.signal(df_30i, df_1hi)
                            
                            # Monitor'e kaydet
                            if out_sig and out_sig.get('is_adaptive'):
                                adaptive_monitor.record_adaptive_signal(sym, out_sig)
                                adaptive_signals += 1
                    
                    # === ADAPTIVE SHORT THE RIP ===
                    if not out_sig and s_cfg.get("short_the_rip", {}).get("enable", True):
                        adaptive_str = AdaptiveShortTheRip(s_cfg.get("short_the_rip"))
                        if len(df_30i) >= 50 and len(df_1hi) >= 50:
                            out_sig = out_sig or adaptive_str.signal(df_30i, df_1hi)
                            
                            # Monitor'e kaydet
                            if out_sig and out_sig.get('is_adaptive'):
                                adaptive_monitor.record_adaptive_signal(sym, out_sig)
                                adaptive_signals += 1
                else:
                    # === BASE STRATEGIES (eski) ===
                    if s_cfg.get("oversold_bounce", {}).get("enable", True):
                        ob = OversoldBounce(s_cfg.get("oversold_bounce"))
                        if len(df_30i) >= 50:
                            out_sig = out_sig or ob.signal(df_30i)
                            if out_sig:
                                base_signals += 1

                    if not out_sig and s_cfg.get("short_the_rip", {}).get("enable", True):
                        strp = ShortTheRip(s_cfg.get("short_the_rip"))
                        if len(df_30i) >= 50 and len(df_1hi) >= 50:
                            out_sig = out_sig or strp.signal(df_30i, df_1hi)
                            if out_sig:
                                base_signals += 1

                # Process signal
                if out_sig:
                    # Enrich with ATR
                    if 'atr' in df_30i.columns and len(df_30i) > 0:
                        out_sig['atr'] = float(df_30i['atr'].iloc[-1])
                    
                    # Notify
                    strategy_type = out_sig.get('strategy_type', 'base')
                    emoji = "🤖" if strategy_type == 'adaptive' else "⚡"
                    msg = f"{emoji} <b>{ex_name} ({strategy_type})</b> | <code>{sym}</code> | {out_sig['side'].upper()} — {out_sig['reason']}"
                    if tg: tg.send(msg)
                    
                    signals_out.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "exchange": ex_name,
                        "symbol": sym,
                        "side": out_sig.get("side"),
                        "reason": out_sig.get("reason"),
                        "strategy_type": strategy_type
                    })
                    
                    # Execute if live
                    if is_live and ex_name == exec_exchange:
                        exec_result = execute_signal(exec_engine, client, sym, out_sig, risk_params, tg)
                        if exec_result:
                            executions.append(exec_result)
                            trades_today += 1
                            if day_stats:
                                day_stats['signals'] = trades_today
                                save_day_stats(day_stats)

            except Exception as e:
                if tg:
                    tg.send(f"⚠️ {ex_name}:{sym} skip — {type(e).__name__}: {str(e)[:140]}")
                else:
                    logger.error(f"error {ex_name} {sym}: {e}")
                continue
        
        # Break if limit reached
        if is_live and trades_today >= risk_params['daily_max_trades']:
            break

    # Save artifacts
    if signals_out:
        csv_path = save_signals_csv(signals_out)
        if tg:
            tg.send(f"📦 Sinyaller CSV: <code>{csv_path}</code>")
    else:
        if tg:
            tg.send("ℹ️ Bu turda sinyal yok.")

    # Adaptive monitoring summary
    if use_adaptive and adaptive_signals > 0:
        summary = adaptive_monitor.get_summary()
        logger.info(f"📊 Adaptive Summary: {summary}")
        if tg:
            tg.send(
                f"📊 <b>Adaptive Stats</b>\n"
                f"Adaptive signals: {adaptive_signals}\n"
                f"Total processed: {summary['symbols_processed']}\n"
                f"Active symbols: {summary['active_symbols']}"
            )

    # Append summary
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"Total signals: {len(signals_out)}\n")
            f.write(f"Adaptive signals: {adaptive_signals}\n")
            f.write(f"Base signals: {base_signals}\n")
            f.write(f"Total executions: {len(executions)}\n")
            if csv_path:
                f.write(f"CSV: {csv_path}\n")
    except Exception:
        pass

    return signals_out


async def main_live_trading():
    """
    Main entry point for live trading mode using ProductionCoordinator.
    Supports both paper trading and live trading based on TRADING_MODE env var.
    """
    from core.production_coordinator import ProductionCoordinator
    
    logger.info("="*70)
    logger.info("LIVE TRADING MODE - Starting Production Coordinator")
    logger.info("="*70)
    
    try:
        # Initialize coordinator
        coordinator = ProductionCoordinator()
        
        # Load exchange clients from environment
        logger.info("Loading exchange clients from environment...")
        exchange_clients = build_clients_from_env()
        
        if not exchange_clients:
            raise SystemExit("ERROR: No exchange clients configured. Set EXCHANGES environment variable.")
        
        logger.info(f"Loaded exchanges: {list(exchange_clients.keys())}")
        
        # Portfolio configuration from environment
        portfolio_config = {
            'equity_usd': float(os.getenv('EQUITY_USD', '100'))
        }
        logger.info(f"Portfolio config: {portfolio_config}")
        
        # Initialize production system
        logger.info("Initializing production system...")
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        # Get trading mode and duration from environment
        mode = os.getenv('TRADING_MODE', 'paper').lower()
        duration = int(os.getenv('TRADING_DURATION', '0')) or None
        
        if mode == 'live':
            logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        else:
            logger.info(f"📝 Paper trading mode - No real executions")
        
        if duration:
            logger.info(f"Trading duration: {duration} seconds")
        else:
            logger.info("Trading duration: Unlimited (until manual stop)")
        
        # Run production loop
        logger.info("Starting production trading loop...")
        await coordinator.run_production_loop(mode=mode, duration=duration)
        
        logger.info("Production trading loop completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt - shutting down gracefully")
    except Exception as e:
        logger.error(f"FATAL ERROR in live trading: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        
        # Try to send notification
        tg = build_tg()
        if tg:
            tg.send(f"🛑 FATAL ERROR in live trading: {type(e).__name__}: {str(e)[:200]}")
        
        raise


async def run_with_pipeline():
    """
    Stub for pipeline mode - to be implemented.
    This is a placeholder to prevent errors when --pipeline is used.
    """
    logger.error("Pipeline mode is not yet implemented")
    raise NotImplementedError("Pipeline mode is not yet implemented. Use --live mode instead.")


# ... rest of the file (async functions etc.) remains same ...

if __name__ == "__main__":
    import sys
    import asyncio
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bearish Alpha Bot - Trading System')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode with ProductionCoordinator')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode (used with --live)')
    parser.add_argument('--pipeline', action='store_true', help='Run with Market Data Pipeline mode')
    
    args = parser.parse_args()
    
    # Check for pipeline mode
    if args.pipeline or '--pipeline' in sys.argv:
        logger.info("Starting with Market Data Pipeline mode")
        asyncio.run(run_with_pipeline())
    elif args.live or '--live' in sys.argv:
        # Set TRADING_MODE based on --paper flag
        if args.paper or '--paper' in sys.argv:
            os.environ.setdefault('TRADING_MODE', 'paper')
            logger.info("Using paper trading mode")
        asyncio.run(main_live_trading())
    else:
        # Run traditional one-shot mode
        try:
            out = run_once()
            print(f"✅ Done. signals={len(out)}")
        except Exception as e:
            print("FATAL:", e)
            traceback.print_exc()
            tg = build_tg()
            if tg:
                tg.send(f"🛑 FATAL: {type(e).__name__}: {str(e)[:200]}")
