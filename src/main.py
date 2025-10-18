#!/usr/bin/env python3
# Bearish Alpha Bot — Orchestrated MVP (run summary + artifact guarantee)

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
from strategies.oversold_bounce import OversoldBounce
from strategies.short_the_rip import ShortTheRip
from core.state import load_state, save_state, load_day_stats, save_day_stats
from core.trailing import initial_stops
from core.sizing import position_size_usdt
from core.limits import clamp_amount, meets_or_scale_notional
from core.normalize import amount_to_precision
from core.exec_engine import ExecEngine

DATA_DIR = "data"

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
    # Config'i yükle
    cfg = load_config()
    risk_cfg = cfg.get('risk', {})
    
    # Config'den al, yoksa ENV'den, o da yoksa default
    return {
        'equity_usd': float(
            risk_cfg.get('equity_usd') or 
            os.getenv('RISK_EQUITY_USD', '100')  # Default 100
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
        atr = signal.get('atr', price * 0.02)  # fallback 2% ATR
        
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
        
        # Calculate position size
        risk_usd = min(
            risk_params['equity_usd'] * risk_params['per_trade_risk_pct'],
            risk_params['risk_usd_cap']
        )
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
        
        if tg:
            tg.send(f"✅ <b>EXECUTED</b> {symbol} {order_side.upper()} qty={qty_final:.6f} @ ~{price:.4f} | TP={tp:.4f} SL={sl:.4f}")
        
        return {
            'symbol': symbol,
            'side': side,
            'qty': qty_final,
            'price': price,
            'tp': tp,
            'sl': sl,
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
    
    # Determine mode
    mode = os.getenv('MODE', 'paper').lower()
    is_live = (mode == 'live')
    
    # Get execution exchange
    from universe import pick_execution_exchange
    exec_exchange = pick_execution_exchange()

    # --- RUN SUMMARY (always create artifact) ---
    ensure_data_dir()
    summary_path = os.path.join(DATA_DIR, "RUN_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Run start (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MODE: {mode}\n")
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
            tg.send(f"🚀 <b>LIVE MODE</b> activated on {exec_exchange} | Risk: {risk_params['risk_usd_cap']}$ cap, {risk_params['daily_max_trades']} max trades/day")
    else:
        if tg:
            tg.send("🔎 <b>Bearish Alpha Bot</b> tarama başlıyor (paper)")

    from universe import build_universe
    universe = build_universe(clients, cfg)
    max_per_ex = int(cfg.get("universe", {}).get("top_n_per_exchange", 20) or 20)

    signals_out = []
    executions = []
    csv_path = None
    
    # Load daily stats for live mode trade counting
    day_stats = load_day_stats() if is_live else None
    trades_today = day_stats.get('signals', 0) if day_stats else 0

    for ex_name, client in clients.items():
        symbols = universe.get(ex_name, [])[:max_per_ex]
        if not symbols:
            continue

        for sym in symbols:
            try:
                # Check daily trade limit in live mode
                if is_live and trades_today >= risk_params['daily_max_trades']:
                    if tg:
                        tg.send(f"⚠️ Daily trade limit ({risk_params['daily_max_trades']}) reached, stopping scan")
                    break
                
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
                    # Enrich signal with ATR for execution
                    if 'atr' in df_30i.columns and len(df_30i) > 0:
                        out_sig['atr'] = float(df_30i['atr'].iloc[-1])
                    
                    # Notify signal
                    msg = f"⚡ <b>{ex_name}</b> | <code>{sym}</code> | {out_sig['side'].upper()} — {out_sig['reason']}"
                    if tg: tg.send(msg)
                    
                    signals_out.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "exchange": ex_name,
                        "symbol": sym,
                        "side": out_sig.get("side"),
                        "reason": out_sig.get("reason"),
                    })
                    
                    # Execute if live mode and on execution exchange
                    if is_live and ex_name == exec_exchange:
                        exec_result = execute_signal(exec_engine, client, sym, out_sig, risk_params, tg)
                        if exec_result:
                            executions.append(exec_result)
                            trades_today += 1
                            # Update day stats
                            if day_stats:
                                day_stats['signals'] = trades_today
                                save_day_stats(day_stats)

            except Exception as e:
                if tg:
                    tg.send(f"⚠️ {ex_name}:{sym} skip — {type(e).__name__}: {str(e)[:140]}")
                else:
                    print("error", ex_name, sym, e)
                continue
        
        # Break outer loop if daily limit reached
        if is_live and trades_today >= risk_params['daily_max_trades']:
            break

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
            f.write(f"Total executions: {len(executions)}\n")
            if csv_path:
                f.write(f"CSV: {csv_path}\n")
    except Exception:
        pass

    return signals_out

async def run_with_pipeline():
    """Market Data Pipeline ile optimize edilmiş çalışma modu"""
    from core.market_data_pipeline import MarketDataPipeline
    import asyncio
    import signal
    
    logger = logging.getLogger(__name__)
    
    cfg = load_config()
    tg = build_tg()
    
    # Pipeline duration from environment
    duration_minutes = int(os.getenv('PIPELINE_DURATION_MINUTES', '0'))
    max_iterations = int(os.getenv('PIPELINE_MAX_ITERATIONS', '0'))
    
    if duration_minutes > 0:
        max_iterations = duration_minutes * 2  # Her 30 saniye bir iteration
        logger.info(f"Pipeline will run for {duration_minutes} minutes ({max_iterations} iterations)")
    elif max_iterations > 0:
        logger.info(f"Pipeline will run for {max_iterations} iterations")
    else:
        logger.info("Pipeline running in CONTINUOUS mode (until interrupted)")
        max_iterations = float('inf')  # Sonsuz
    
    # Build clients
    required_symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
        'LTC/USDT:USDT', 'AVAX/USDT:USDT'
    ]
    
    logger.info("Building exchange clients...")
    clients = build_clients_from_env(required_symbols=required_symbols)
    
    if not clients:
        logger.error("No exchange clients available")
        if tg:
            tg.send("🛑 No exchange clients available for pipeline mode")
        return
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline with {len(clients)} exchanges")
    pipeline = MarketDataPipeline(clients)
    
    # Setup signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Timeframes to track
    timeframes = ['30m', '1h', '4h']
    
    # Start data feeds
    logger.info(f"Starting data feeds for {len(required_symbols)} symbols")
    await pipeline.start_feeds_async(required_symbols, timeframes)
    
    if tg:
        tg.send(
            f"🚀 <b>Pipeline Mode Started</b>\n"
            f"Symbols: {len(required_symbols)}\n"
            f"Timeframes: {len(timeframes)}\n"
            f"Duration: {'Continuous' if max_iterations == float('inf') else f'{max_iterations} iterations'}"
        )
    
    # Ana döngü
    iteration = 0
    start_time = datetime.now(timezone.utc)
    signals_total = 0
    errors_total = 0
    
    try:
        while iteration < max_iterations and not shutdown_event.is_set():
            try:
                iteration += 1
                logger.info(f"=" * 60)
                logger.info(f"Pipeline iteration {iteration}")
                logger.info(f"=" * 60)
                
                # Pipeline health check
                health = pipeline.get_health_status()
                if health['overall_status'] != 'healthy':
                    logger.warning(f"Pipeline health issue: {health}")
                    errors_total += 1
                    if tg and errors_total % 10 == 0:  # Her 10 hatada bir bildir
                        tg.send(
                            f"⚠️ Pipeline health: {health['overall_status']}\n"
                            f"Error rate: {health['error_rate']}%\n"
                            f"Errors total: {errors_total}"
                        )
                
                # Process each symbol
                signals_count = 0
                for symbol in required_symbols:
                    try:
                        # Get cached data from pipeline (fast!)
                        df_30m = pipeline.get_latest_ohlcv(symbol, '30m')
                        df_1h = pipeline.get_latest_ohlcv(symbol, '1h')
                        df_4h = pipeline.get_latest_ohlcv(symbol, '4h')
                        
                        if df_30m is None or len(df_30m) < 50:
                            logger.debug(f"Insufficient 30m data for {symbol}")
                            continue
                        
                        # Indicators are already included in pipeline data
                        ind_cfg = cfg.get("indicators", {}) or {}
                        
                        # Ensure indicators are present
                        if 'rsi' not in df_30m.columns:
                            df_30m = ind_enrich(df_30m, ind_cfg).dropna()
                        if df_1h is not None and 'rsi' not in df_1h.columns:
                            df_1h = ind_enrich(df_1h, ind_cfg).dropna()
                        if df_4h is not None and 'rsi' not in df_4h.columns:
                            df_4h = ind_enrich(df_4h, ind_cfg).dropna()
                        
                        # Regime check
                        ignore_regime = cfg.get("signals", {}).get("oversold_bounce", {}).get("ignore_regime", False)
                        if df_4h is not None and len(df_4h) >= 50 and not ignore_regime:
                            bearish = is_bearish_regime(df_4h)
                            if not bearish:
                                logger.debug(f"{symbol} not in bearish regime, skipping")
                                continue
                        
                        # Strategy signals
                        signal = None
                        
                        # OversoldBounce
                        if cfg.get("signals", {}).get("oversold_bounce", {}).get("enable", True):
                            ob = OversoldBounce(cfg.get("signals", {}).get("oversold_bounce"))
                            signal = ob.signal(df_30m)
                        
                        # ShortTheRip
                        if not signal and cfg.get("signals", {}).get("short_the_rip", {}).get("enable", True):
                            str_cfg = cfg.get("signals", {}).get("short_the_rip")
                            strp = ShortTheRip(str_cfg)
                            if df_1h is not None and len(df_1h) >= 50:
                                signal = strp.signal(df_30m, df_1h)
                        
                        # Process signal
                        if signal:
                            signals_count += 1
                            signals_total += 1
                            msg = f"⚡ Pipeline | {symbol} | {signal['side'].upper()} — {signal['reason']}"
                            if tg: 
                                tg.send(msg)
                            logger.info(f"Signal generated: {msg}")
                            
                            # TODO: Execute trade if live mode
                            mode = os.getenv('MODE', 'paper').lower()
                            if mode == 'live':
                                logger.info(f"TODO: Execute live trade for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        errors_total += 1
                        continue
                
                # Log iteration summary
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(f"Iteration {iteration} complete:")
                logger.info(f"  Signals this iteration: {signals_count}")
                logger.info(f"  Total signals: {signals_total}")
                logger.info(f"  Total errors: {errors_total}")
                logger.info(f"  Elapsed time: {elapsed/60:.1f} minutes")
                logger.info(f"  Health: {health['overall_status']}")
                
                # Periodic summary to Telegram
                if iteration % 20 == 0 and tg:  # Every 10 minutes
                    tg.send(
                        f"📊 <b>Pipeline Status</b>\n"
                        f"Iteration: {iteration}\n"
                        f"Signals: {signals_total}\n"
                        f"Errors: {errors_total}\n"
                        f"Uptime: {elapsed/60:.1f}m\n"
                        f"Health: {health['overall_status']}"
                    )
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 saniye bekle
                
            except asyncio.CancelledError:
                logger.info("Pipeline cancelled")
                break
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}", exc_info=True)
                errors_total += 1
                if tg and errors_total == 1:  # İlk hatada bildir
                    tg.send(f"❌ Pipeline error: {type(e).__name__}: {str(e)[:150]}")
                await asyncio.sleep(60)  # Hata durumunda 1 dk bekle
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    finally:
        # Shutdown
        logger.info("Shutting down pipeline...")
        pipeline.shutdown()
        
        # Final summary
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        summary_msg = (
            f"Pipeline stopped after {iteration} iterations\n"
            f"Total runtime: {elapsed/60:.1f} minutes\n"
            f"Total signals: {signals_total}\n"
            f"Total errors: {errors_total}"
        )
        logger.info(summary_msg)
        
        if tg:
            tg.send(f"✅ <b>Pipeline Stopped</b>\n{summary_msg}")

async def main_live_trading():
    """Main entry point for live trading mode using Phase 3.4 infrastructure."""
    import asyncio
    from core.production_coordinator import ProductionCoordinator
    
    logger = logging.getLogger(__name__)
    logger.info("Starting live trading mode with Phase 3.4 infrastructure")
    
    try:
        # Load configuration
        cfg = load_config()
        risk_params = get_risk_params()
        
        # Build exchange clients (Phase 1)
        clients = build_clients_from_env()
        
        if not clients:
            raise RuntimeError("No exchange clients available")
        
        # Portfolio configuration
        portfolio_config = {
            'equity_usd': risk_params['equity_usd']
        }
        
        # Initialize production coordinator
        coordinator = ProductionCoordinator()
        
        # Initialize production system
        init_result = await coordinator.initialize_production_system(
            exchange_clients=clients,
            portfolio_config=portfolio_config
        )
        
        if not init_result['success']:
            raise RuntimeError(f"Failed to initialize production system: {init_result.get('reason')}")
        
        logger.info("Production system initialized successfully")
        
        # Get trading mode from environment
        trading_mode = os.getenv('TRADING_MODE', 'paper')  # 'paper', 'live', 'simulation'
        duration = float(os.getenv('TRADING_DURATION', '0'))  # 0 = indefinite
        
        # Start production loop
        if duration > 0:
            logger.info(f"Starting {trading_mode} trading for {duration} seconds")
            await coordinator.run_production_loop(mode=trading_mode, duration=duration)
        else:
            logger.info(f"Starting {trading_mode} trading (indefinite)")
            await coordinator.run_production_loop(mode=trading_mode)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in live trading mode: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    import asyncio
    
    # Check for pipeline mode
    if '--pipeline' in sys.argv:
        logger = logging.getLogger(__name__)
        logger.info("Starting with Market Data Pipeline mode")
        asyncio.run(run_with_pipeline())
    elif len(sys.argv) > 1 and sys.argv[1] == '--live':
        # Run live trading mode
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
