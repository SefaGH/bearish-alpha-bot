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

import os, itertools, logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import yaml

from core.multi_exchange import build_clients_from_env
from core.indicators import add_indicators as ind_enrich

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    print(f"Input data length: {len(df30)}")
    print(f"RSI range in data: {df30['rsi'].min():.1f} - {df30['rsi'].max():.1f}")
    
    res = []
    total_combinations = len(list(itertools.product(grid["rsi_max"], grid["tp_pct"], grid["sl_atr_mult"])))
    print(f"Testing {total_combinations} parameter combinations...")
    
    for rsi_max, tp_pct, sl_atr_mult in itertools.product(grid["rsi_max"], grid["tp_pct"], grid["sl_atr_mult"]):
        mask = df30["rsi"] <= rsi_max
        sub = df30.loc[mask].copy()
        
        if len(sub) < 5:
            continue
        
        print(f"RSI≤{rsi_max}: {len(sub)} qualifying candles")
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
        "rsi_max": [25, 30, 35, 40, 45],
        "tp_pct": [0.008, 0.012, 0.016, 0.020],
        "sl_atr_mult": [0.8, 1.0, 1.2, 1.5, 2.0]
    }
    
    logger.info("=" * 60)
    logger.info("BACKTEST PARAM SWEEP - OversoldBounce Strategy")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Limit: {limit} candles")
    
    cfg_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
    try:
        logger.debug(f"Loading config from: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info(f"Config loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load config from {cfg_path}: {e}. Using defaults.")
        cfg = {}
    
    ind_cfg = cfg.get("indicators", {}) or {}
    ob_cfg = (cfg.get("signals", {}) or {}).get("oversold_bounce", {}) or {}
    fallback_sl_pct = ob_cfg.get("sl_pct")

    try:
        logger.info("Building exchange clients from environment...")
        clients = build_clients_from_env()
        logger.info(f"Available exchanges: {list(clients.keys())}")
    except Exception as e:
        logger.error(f"❌ Failed to build exchange clients: {type(e).__name__}: {e}")
        logger.error("Please check that EXCHANGES environment variable is set and credentials are valid")
        raise SystemExit(1) from e
    
    if exchange not in clients:
        if clients:
            old_exchange = exchange
            exchange, client = next(iter(clients.items()))
            logger.warning(f"Exchange '{old_exchange}' not available. Using '{exchange}' instead.")
        else:
            logger.error("❌ No exchange available. Set EXCHANGES=... in environment")
            raise SystemExit(1)
    else:
        client = clients[exchange]
        logger.info(f"Using exchange: {exchange}")

    # Validate and get the correct symbol format for this exchange
    try:
        logger.info(f"Validating symbol '{symbol}' on {exchange}...")
        validated_symbol = client.validate_and_get_symbol(symbol)
        logger.info(f"✓ Using symbol: {validated_symbol}")
    except Exception as e:
        logger.error(f"❌ Symbol validation failed: {type(e).__name__}: {e}")
        logger.error(f"   Exchange: {exchange}")
        logger.error(f"   Requested symbol: {symbol}")
        raise SystemExit(1) from e

    try:
        logger.info(f"Fetching OHLCV data: {validated_symbol} 30m limit={limit}...")
        rows = client.ohlcv(validated_symbol, timeframe="30m", limit=limit)
        logger.info(f"✓ Fetched {len(rows)} candles")
    except Exception as e:
        logger.error(f"❌ Failed to fetch OHLCV data: {type(e).__name__}: {e}")
        logger.error(f"   Exchange: {exchange}")
        logger.error(f"   Symbol: {validated_symbol}")
        logger.error(f"   Timeframe: 30m")
        logger.error(f"   Limit: {limit}")
        raise SystemExit(1) from e
    
    try:
        df30 = _df_from_ohlcv(rows)
        logger.info(f"Processing indicators...")
        df30i = ind_enrich(df30, ind_cfg).dropna()
        logger.info(f"✓ Data ready: {len(df30i)} candles after indicator enrichment")
    except Exception as e:
        logger.error(f"❌ Data processing failed: {type(e).__name__}: {e}")
        raise SystemExit(1) from e

    try:
        logger.info("Running parameter sweep...")
        dfres = sweep_ob(df30i, grid, {"sl_pct": fallback_sl_pct})
    except Exception as e:
        logger.error(f"❌ Parameter sweep failed: {type(e).__name__}: {e}")
        raise SystemExit(1) from e
    
    if dfres.empty:
        logger.warning("⚠️ No results produced. Check data length or grid ranges.")
        logger.warning(f"   Data length: {len(df30i)} candles")
        logger.warning(f"   RSI range: {df30i['rsi'].min():.1f} - {df30i['rsi'].max():.1f}")
        return

    try:
        os.makedirs(BT_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(BT_DIR, f"{symbol.replace('/','_')}_{ts}.csv")
        dfres.to_csv(out_path, index=False)
        logger.info(f"✅ Results written to: {out_path}")
        print(f"✅ Wrote: {out_path}")
        print(dfres.head(10).to_string(index=False))
    except Exception as e:
        logger.error(f"❌ Failed to write results: {type(e).__name__}: {e}")
        raise SystemExit(1) from e

if __name__ == "__main__":
    main()
