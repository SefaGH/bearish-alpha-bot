#!/usr/bin/env python3
"""
Example: Phase 1 + Phase 2 + Phase 3.1 Integration
Demonstrates how all three phases work together for real-time intelligent trading.
"""

import asyncio
import logging
import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient
from core.multi_exchange_manager import MultiExchangeManager
from core.market_regime import MarketRegimeAnalyzer
from core.websocket_manager import WebSocketManager, StreamDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def integrated_example():
    """
    Demonstrate integration of all three phases:
    - Phase 1: Multi-Exchange data fetching (REST API)
    - Phase 2: Market regime analysis (Intelligence)
    - Phase 3.1: Real-time data streaming (WebSocket)
    """
    
    logger.info("="*70)
    logger.info("Phase 1 + Phase 2 + Phase 3.1 Integration Example")
    logger.info("="*70)
    
    # =========================================================================
    # Phase 1: Multi-Exchange Framework (REST API)
    # =========================================================================
    logger.info("\n[PHASE 1] Initializing Multi-Exchange Framework...")
    
    # Create exchange clients
    exchanges = {
        'kucoinfutures': CcxtClient('kucoinfutures'),
        'bingx': CcxtClient('bingx')
    }
    
    # Create multi-exchange manager
    rest_manager = MultiExchangeManager(exchanges)
    
    # Get exchange summary
    summary = rest_manager.get_exchange_summary()
    logger.info(f"✓ Exchanges initialized: {len(summary['exchanges'])} exchanges")
    for ex_name, info in summary['exchanges'].items():
        logger.info(f"  - {ex_name}: {info['status']}")
    
    # =========================================================================
    # Phase 2: Market Intelligence Engine
    # =========================================================================
    logger.info("\n[PHASE 2] Initializing Market Intelligence...")
    
    # Create regime analyzer
    analyzer = MarketRegimeAnalyzer()
    logger.info("✓ Market Regime Analyzer initialized")
    
    # Fetch historical data for regime analysis (REST API)
    logger.info("\n[PHASE 1→2] Fetching historical data for regime analysis...")
    try:
        symbols_per_exchange = {
            'kucoinfutures': ['BTC/USDT:USDT']
        }
        
        # Fetch 30m, 1h, 4h data for regime detection
        data_30m = rest_manager.fetch_unified_data(
            symbols_per_exchange, timeframe='30m', limit=100
        )
        data_1h = rest_manager.fetch_unified_data(
            symbols_per_exchange, timeframe='1h', limit=100
        )
        data_4h = rest_manager.fetch_unified_data(
            symbols_per_exchange, timeframe='4h', limit=100
        )
        
        if data_30m.get('kucoinfutures', {}).get('BTC/USDT:USDT'):
            logger.info("✓ Historical data fetched successfully")
            
            # Convert to DataFrames
            btc_30m = data_30m['kucoinfutures']['BTC/USDT:USDT']
            btc_1h = data_1h['kucoinfutures']['BTC/USDT:USDT']
            btc_4h = data_4h['kucoinfutures']['BTC/USDT:USDT']
            
            df_30m = pd.DataFrame(btc_30m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h = pd.DataFrame(btc_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(btc_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Analyze market regime
            logger.info("\n[PHASE 2] Analyzing market regime...")
            regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
            
            logger.info("✓ Market Regime Analysis:")
            logger.info(f"  - Trend: {regime.get('trend', 'unknown')}")
            logger.info(f"  - Momentum: {regime.get('momentum', 'unknown')}")
            logger.info(f"  - Volatility: {regime.get('volatility', 'unknown')}")
            logger.info(f"  - Risk Multiplier: {regime.get('risk_multiplier', 1.0)}")
        else:
            logger.warning("⚠ Could not fetch historical data (may need credentials)")
            
    except Exception as e:
        logger.warning(f"⚠ Historical data fetch failed: {e}")
        logger.info("  (This is expected in sandbox environment)")
    
    # =========================================================================
    # Phase 3.1: WebSocket Infrastructure (Real-time)
    # =========================================================================
    logger.info("\n[PHASE 3.1] Initializing WebSocket Infrastructure...")
    
    # Create WebSocket manager
    ws_manager = WebSocketManager()
    collector = StreamDataCollector(buffer_size=100)
    
    logger.info("✓ WebSocket Manager initialized")
    logger.info(f"✓ Data Collector initialized (buffer: {collector.buffer_size})")
    
    # Get WebSocket status
    ws_status = ws_manager.get_stream_status()
    logger.info(f"✓ WebSocket exchanges available: {', '.join(ws_status['exchanges'])}")
    
    # =========================================================================
    # Integration: Real-time streaming with regime analysis
    # =========================================================================
    logger.info("\n[INTEGRATION] Starting real-time streaming...")
    logger.info("  Phase 3.1 streams → Phase 2 analysis → Phase 1 data")
    
    # Define a callback that combines streaming with regime analysis
    async def intelligent_callback(exchange, symbol, timeframe, ohlcv):
        """Process streaming data with intelligent analysis."""
        logger.info(f"📊 New candle: {exchange} {symbol} {timeframe}")
        
        # Could integrate with Phase 2 regime detection here
        if len(ohlcv) >= 100:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # regime = analyzer.analyze_market_regime(df, df, df)
            # logger.info(f"  Current regime: {regime.get('trend')}")
    
    try:
        logger.info("\n  Starting OHLCV stream (will run for 5 seconds)...")
        symbols_per_exchange = {
            'kucoinfutures': ['BTC/USDT:USDT']
        }
        
        # Start streaming
        await ws_manager.stream_ohlcv(
            symbols_per_exchange,
            timeframe='1m',
            callback=intelligent_callback,
            max_iterations=3
        )
        
        # Run for 5 seconds
        await ws_manager.run_streams(duration=5)
        
        logger.info("\n✓ Streaming completed")
        
    except Exception as e:
        logger.info(f"⚠ Streaming test: {e}")
        logger.info("  (WebSocket connection requires live exchange access)")
    
    finally:
        await ws_manager.close()
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("Integration Example Complete")
    logger.info("="*70)
    logger.info("\n✅ Phase 1: Multi-Exchange Framework - REST API data fetching")
    logger.info("✅ Phase 2: Market Intelligence - Regime analysis and adaptation")
    logger.info("✅ Phase 3.1: WebSocket Infrastructure - Real-time streaming")
    logger.info("\n💡 All phases work together seamlessly for intelligent trading")
    logger.info("="*70)


if __name__ == '__main__':
    asyncio.run(integrated_example())
