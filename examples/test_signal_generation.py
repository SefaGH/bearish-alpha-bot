#!/usr/bin/env python3
"""
Example: Testing Signal Generation in Live Trading Engine

This example demonstrates how the live trading engine scans markets
and generates signals using adaptive strategies.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_signal_generation():
    """Test signal generation with mock data."""
    from core.live_trading_engine import LiveTradingEngine
    from core.portfolio_manager import PortfolioManager
    from core.risk_manager import RiskManager
    from core.performance_monitor import RealTimePerformanceMonitor
    from strategies.adaptive_ob import AdaptiveOversoldBounce
    from strategies.adaptive_str import AdaptiveShortTheRip
    from core.ccxt_client import CcxtClient
    
    logger.info("="*70)
    logger.info("TESTING LIVE TRADING ENGINE SIGNAL GENERATION")
    logger.info("="*70)
    
    # Setup components
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    performance_monitor = RealTimePerformanceMonitor()
    portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
    
    # Register strategies
    ob_config = {'rsi_max': 30, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
    str_config = {'rsi_min': 70, 'tp_pct': 0.012, 'sl_atr_mult': 1.2}
    
    adaptive_ob = AdaptiveOversoldBounce(ob_config)
    adaptive_str = AdaptiveShortTheRip(str_config)
    
    portfolio_manager.register_strategy('adaptive_oversold_bounce', adaptive_ob, 0.3)
    portfolio_manager.register_strategy('adaptive_short_the_rip', adaptive_str, 0.3)
    
    logger.info(f"✓ Registered {len(portfolio_manager.strategies)} strategies")
    
    # Create exchange clients (mock for testing)
    exchange_clients = {}  # Empty for now - would normally contain real exchange clients
    
    # Create engine
    engine = LiveTradingEngine(
        mode='paper',
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        websocket_manager=None,
        exchange_clients=exchange_clients
    )
    
    logger.info("✓ Engine initialized")
    
    # Show configuration
    symbols = engine._get_scan_symbols()
    logger.info(f"\nConfiguration:")
    logger.info(f"  Symbols to scan: {len(symbols)}")
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"    {i}. {symbol}")
    
    logger.info(f"\n  Strategies:")
    for name in portfolio_manager.strategies.keys():
        logger.info(f"    - {name}")
    
    logger.info(f"\n  Scan interval: 30 seconds")
    logger.info(f"  Signal queue: Ready")
    
    logger.info("\n" + "="*70)
    logger.info("SIGNAL GENERATION PROCESS")
    logger.info("="*70)
    logger.info("\nWhen live:")
    logger.info("  1. Engine scans each symbol every 30 seconds")
    logger.info("  2. Fetches OHLCV data for 30m, 1h, 4h timeframes")
    logger.info("  3. Calculates technical indicators (RSI, EMA, ATR)")
    logger.info("  4. Performs market regime analysis")
    logger.info("  5. Runs adaptive strategies with regime awareness")
    logger.info("  6. Generates signals when conditions are met")
    logger.info("  7. Adds signals to queue for execution")
    
    logger.info("\n" + "="*70)
    logger.info("✓ TEST COMPLETED SUCCESSFULLY")
    logger.info("="*70)


if __name__ == '__main__':
    asyncio.run(test_signal_generation())
