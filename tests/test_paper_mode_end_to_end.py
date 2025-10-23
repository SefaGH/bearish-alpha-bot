#!/usr/bin/env python3
"""
End-to-End Paper Mode Test

Tests the complete flow from initialization to trading loop execution.
This test simulates a real paper mode session to identify any blocking issues.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient
from unittest.mock import MagicMock, patch


async def test_end_to_end_paper_mode():
    """
    Test complete end-to-end flow in paper mode.
    
    This simulates what happens when the bot is started in production:
    1. Coordinator initialization
    2. Exchange client setup
    3. Trading engine startup
    4. Production loop execution for a short duration
    """
    # Configure logging at INFO level (non-debug mode)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("END-TO-END PAPER MODE TEST")
    logger.info("="*70)
    
    start_time = datetime.now(timezone.utc)
    
    # Step 1: Create coordinator
    logger.info("\n[STEP 1] Creating ProductionCoordinator...")
    coordinator = ProductionCoordinator()
    logger.info("✅ Coordinator created")
    
    # Step 2: Setup mock exchange client
    logger.info("\n[STEP 2] Setting up mock exchange clients...")
    mock_client = MagicMock(spec=CcxtClient)
    
    # Mock OHLCV data that will generate signals
    # Constants for timestamp generation
    BASE_TIMESTAMP = 1000000
    MINUTE_IN_MS = 60000
    
    def generate_ohlcv(symbol, timeframe, limit=100):
        """
        Generate mock OHLCV with RSI patterns.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe string (e.g., '30m', '1h')
            limit: Number of candles to generate
            
        Returns:
            List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        # Simulate oversold condition (RSI < 30) for last few candles
        data = []
        base_price = 30000 if 'BTC' in symbol else 2000
        
        for i in range(limit):
            timestamp = BASE_TIMESTAMP + (i * MINUTE_IN_MS)
            if i < limit - 5:
                # Normal prices
                open_price = base_price + (i % 10) * 10
                high = open_price + 20
                low = open_price - 20
                close = open_price + 5
            else:
                # Create oversold pattern in last 5 candles
                open_price = base_price - (limit - i) * 50
                high = open_price + 10
                low = open_price - 30
                close = low + 5  # Recovering
            
            volume = 1000.0
            data.append([timestamp, open_price, high, low, close, volume])
        
        return data
    
    mock_client.ohlcv.side_effect = generate_ohlcv
    mock_client.fetch_ticker = MagicMock(return_value={'last': 30000, 'close': 30000})
    
    exchange_clients = {'bingx': mock_client}
    logger.info(f"✅ Mock exchange clients setup: {list(exchange_clients.keys())}")
    
    # Step 3: Initialize production system
    logger.info("\n[STEP 3] Initializing production system...")
    init_result = await coordinator.initialize_production_system(
        exchange_clients=exchange_clients,
        portfolio_config={'equity_usd': 1000.0},
        mode='paper',
        trading_symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT']
    )
    
    if not init_result['success']:
        logger.error(f"❌ Initialization failed: {init_result.get('reason')}")
        return False
    
    logger.info("✅ Production system initialized")
    logger.info(f"   Components: {init_result.get('components', [])}")
    logger.info(f"   Active symbols: {init_result.get('active_symbols_count', 0)}")
    
    # Step 4: Start trading engine
    logger.info("\n[STEP 4] Starting trading engine...")
    engine_result = await coordinator.trading_engine.start_live_trading(mode='paper')
    
    if not engine_result['success']:
        logger.error(f"❌ Engine start failed: {engine_result.get('reason')}")
        return False
    
    logger.info("✅ Trading engine started")
    logger.info(f"   State: {engine_result.get('state')}")
    logger.info(f"   Mode: {engine_result.get('mode')}")
    logger.info(f"   Active tasks: {engine_result.get('active_tasks', 0)}")
    
    # Step 5: Run production loop for short duration
    logger.info("\n[STEP 5] Running production loop...")
    logger.info("   Duration: 5 seconds")
    logger.info("   Symbols: 2 (BTC/USDT:USDT, ETH/USDT:USDT)")
    
    # Track if loop actually runs
    initial_processed_count = coordinator.processed_symbols_count
    
    # Run the production loop
    try:
        await asyncio.wait_for(
            coordinator.run_production_loop(mode='paper', duration=5.0, continuous=False),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        logger.error("❌ Production loop timed out!")
        return False
    
    # Step 6: Verify loop executed
    logger.info("\n[STEP 6] Verifying loop execution...")
    final_processed_count = coordinator.processed_symbols_count
    symbols_processed = final_processed_count - initial_processed_count
    
    logger.info(f"   Symbols processed: {symbols_processed}")
    logger.info(f"   Engine state: {coordinator.trading_engine.state.value}")
    
    if symbols_processed == 0:
        logger.error("❌ No symbols were processed - loop may not have executed!")
        return False
    
    logger.info("✅ Loop executed successfully")
    
    # Step 7: Cleanup
    logger.info("\n[STEP 7] Cleaning up...")
    await coordinator.stop_system()
    logger.info("✅ System stopped cleanly")
    
    # Calculate total time
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*70)
    logger.info("✅ END-TO-END TEST PASSED")
    logger.info("="*70)
    logger.info(f"Total time: {duration:.2f}s")
    logger.info(f"Symbols processed: {symbols_processed}")
    logger.info("="*70)
    
    return True


async def main():
    """Run the end-to-end test."""
    try:
        success = await test_end_to_end_paper_mode()
        if not success:
            print("\n❌ TEST FAILED")
            sys.exit(1)
        else:
            print("\n✅ ALL TESTS PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
