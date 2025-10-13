#!/usr/bin/env python3
# Param Sweep (MVP) for ShortTheRip (30m entries with 1h context)

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

def fetch(client, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    # Use bulk fetch for limits > 500 (up to 2000 candles)
    if limit > 500:
        rows = client.fetch_ohlcv_bulk(symbol, timeframe=tf, target_limit=limit)
    else:
        rows = client.ohlcv(symbol, timeframe=tf, limit=limit)
    return _df_from_ohlcv(rows)

def align_1h_to_30m(df30: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge_asof(
        df30.sort_index(),
        df1h.sort_index(),
        left_index=True, right_index=True,
        direction="backward", suffixes=("", "_1h")
    )
    return joined

def simulate_short_nextbar(df: pd.DataFrame, tp_pct: float, sl_atr_mult: float | None, fallback_sl_pct: float | None) -> Dict[str, float]:
    pnls = []
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
            continue
        if lo <= tp:
            pnls.append((entry - tp)/entry)
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
    dfj = align_1h_to_30m(df30i, df1hi)
    print(f"Aligned data length: {len(dfj)}")
    print(f"RSI range: {dfj['rsi'].min():.1f} - {dfj['rsi'].max():.1f}")
    
    res = []
    total_combinations = len(list(itertools.product(
        grid["rsi_min"], grid["tp_pct"], grid["sl_atr_mult"], 
        grid["require_band_touch"], grid["require_ema_align"]
    )))
    print(f"Testing {total_combinations} parameter combinations...")
    
    for rsi_min, tp_pct, sl_atr_mult, require_band_touch, require_ema_align in itertools.product(
        grid["rsi_min"], grid["tp_pct"], grid["sl_atr_mult"], grid["require_band_touch"], grid["require_ema_align"]
    ):
        mask = dfj["rsi"] >= rsi_min
        if require_ema_align:
            mask = mask & (dfj["ema21"] < dfj["ema50"]) & (dfj["ema50"] <= dfj["ema200"])
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
    return pd.DataFrame(res).sort_values(["avg_pnl","win_rate","trades"], ascending=[False, False, False])

def main():
    symbol = os.getenv("BT_SYMBOL", "BTC/USDT")
    exchange = os.getenv("BT_EXCHANGE", os.getenv("EXECUTION_EXCHANGE", "kucoinfutures"))
    limit30 = int(os.getenv("BT_LIMIT_30M", "1000"))
    limit1h = int(os.getenv("BT_LIMIT_1H", "1000"))

    logger.info("=" * 60)
    logger.info("BACKTEST PARAM SWEEP - ShortTheRip Strategy")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"30m limit: {limit30} candles")
    logger.info(f"1h limit: {limit1h} candles")

    grid = {
        "rsi_min": [55, 60, 65, 70, 75],
        "tp_pct": [0.008, 0.012, 0.016, 0.020],
        "sl_atr_mult": [1.0, 1.2, 1.5, 2.0],
        "require_band_touch": [True, False],
        "require_ema_align": [True, False],
    }

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
        logger.info(f"Fetching 30m OHLCV data: {validated_symbol} limit={limit30}...")
        df30 = fetch(client, validated_symbol, "30m", limit30)
        logger.info(f"✓ Fetched {len(df30)} 30m candles")
        
        logger.info(f"Fetching 1h OHLCV data: {validated_symbol} limit={limit1h}...")
        df1h = fetch(client, validated_symbol, "1h", limit1h)
        logger.info(f"✓ Fetched {len(df1h)} 1h candles")
    except Exception as e:
        logger.error(f"❌ Failed to fetch OHLCV data: {type(e).__name__}: {e}")
        logger.error(f"   Exchange: {exchange}")
        logger.error(f"   Symbol: {validated_symbol}")
        raise SystemExit(1) from e

    try:
        logger.info(f"Processing indicators...")
        df30i = ind_enrich(df30, ind_cfg).dropna()
        df1hi = ind_enrich(df1h, ind_cfg).dropna()
        logger.info(f"✓ Data ready: {len(df30i)} 30m candles, {len(df1hi)} 1h candles after indicator enrichment")
    except Exception as e:
        logger.error(f"❌ Data processing failed: {type(e).__name__}: {e}")
        raise SystemExit(1) from e

    try:
        logger.info("Running parameter sweep...")
        dfres = sweep_str(df30i, df1hi, grid, {"sl_pct": None})
    except Exception as e:
        logger.error(f"❌ Parameter sweep failed: {type(e).__name__}: {e}")
        raise SystemExit(1) from e
    
    if dfres.empty:
        logger.warning("⚠️ No results produced. Check data length or grid ranges.")
        logger.warning(f"   30m data length: {len(df30i)} candles")
        logger.warning(f"   1h data length: {len(df1hi)} candles")
        logger.warning(f"   30m RSI range: {df30i['rsi'].min():.1f} - {df30i['rsi'].max():.1f}")
        return

    try:
        os.makedirs(BT_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(BT_DIR, f"str_{symbol.replace('/','_')}_{ts}.csv")
        dfres.to_csv(out_path, index=False)
        logger.info(f"✅ Results written to: {out_path}")
        print(f"✅ Wrote: {out_path}")
        print(dfres.head(10).to_string(index=False))
    except Exception as e:
        logger.error(f"❌ Failed to write results: {type(e).__name__}: {e}")
        raise SystemExit(1) from e

if __name__ == "__main__":
    main()
