#!/usr/bin/env python3
"""
Test Trading Loop Startup Issue

This test verifies that the trading loop starts correctly regardless of debug mode.
Tests the specific issue where debug mode OFF prevents loop from starting.
"""

import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.live_trading_engine import LiveTradingEngine, EngineState
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager


async def test_trading_loop_starts_without_debug_mode():
    """
    Test that trading loop starts when debug mode is OFF (INFO level).
    
    This is a critical test for the issue where the bot works with
    debug mode ON but not with debug mode OFF.
    """
    # Configure logging at INFO level (debug mode OFF)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("TEST: Trading Loop Startup (Debug Mode OFF)")
    logger.info("="*70)
    
    # Create coordinator
    coordinator = ProductionCoordinator()
    
    # Initialize with minimal mocked components
    mock_clients = {
        'bingx': MagicMock()
    }
    
    # Mock the exchange client methods
    mock_clients['bingx'].ohlcv = MagicMock(return_value=[
        [1000, 100, 105, 95, 102, 1000] for _ in range(200)
    ])
    
    # Initialize coordinator
    init_result = await coordinator.initialize_production_system(
        exchange_clients=mock_clients,
        portfolio_config={'equity_usd': 1000.0},
        mode='paper',
        trading_symbols=['BTC/USDT:USDT']
    )
    
    logger.info(f"Initialization result: {init_result}")
    assert init_result['success'], f"Initialization failed: {init_result.get('reason')}"
    assert coordinator.is_initialized, "Coordinator not initialized"
    assert coordinator.active_symbols, "No active symbols configured"
    
    # Verify trading engine state
    assert coordinator.trading_engine is not None, "Trading engine is None"
    
    # Start the trading engine explicitly
    logger.info("Starting trading engine...")
    engine_start_result = await coordinator.trading_engine.start_live_trading(mode='paper')
    logger.info(f"Engine start result: {engine_start_result}")
    
    assert engine_start_result['success'], f"Engine start failed: {engine_start_result.get('reason')}"
    assert coordinator.trading_engine.state == EngineState.RUNNING, \
        f"Engine not running: {coordinator.trading_engine.state.value}"
    
    # Now test if run_production_loop() starts the main loop
    logger.info("Testing run_production_loop() entry...")
    
    # Set a very short duration to test loop entry
    loop_task = asyncio.create_task(
        coordinator.run_production_loop(mode='paper', duration=2.0, continuous=False)
    )
    
    # Wait a bit for loop to start
    await asyncio.sleep(1.5)
    
    # Check that is_running was set to True
    assert coordinator.is_running, "coordinator.is_running is False - loop didn't start!"
    
    # Wait for duration to complete
    await asyncio.wait_for(loop_task, timeout=5.0)
    
    logger.info("✅ TEST PASSED: Trading loop started successfully with debug mode OFF")
    
    # Cleanup
    await coordinator.stop_system()


async def test_trading_loop_starts_with_debug_mode():
    """
    Test that trading loop starts when debug mode is ON (DEBUG level).
    
    This is the control test to verify the loop works with debug mode ON.
    """
    # Configure logging at DEBUG level (debug mode ON)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("TEST: Trading Loop Startup (Debug Mode ON)")
    logger.info("="*70)
    
    # Create coordinator
    coordinator = ProductionCoordinator()
    
    # Initialize with minimal mocked components
    mock_clients = {
        'bingx': MagicMock()
    }
    
    # Mock the exchange client methods
    mock_clients['bingx'].ohlcv = MagicMock(return_value=[
        [1000, 100, 105, 95, 102, 1000] for _ in range(200)
    ])
    
    # Initialize coordinator
    init_result = await coordinator.initialize_production_system(
        exchange_clients=mock_clients,
        portfolio_config={'equity_usd': 1000.0},
        mode='paper',
        trading_symbols=['BTC/USDT:USDT']
    )
    
    logger.info(f"Initialization result: {init_result}")
    assert init_result['success'], f"Initialization failed: {init_result.get('reason')}"
    
    # Start the trading engine
    engine_start_result = await coordinator.trading_engine.start_live_trading(mode='paper')
    assert engine_start_result['success'], f"Engine start failed: {engine_start_result.get('reason')}"
    
    # Test run_production_loop() entry
    loop_task = asyncio.create_task(
        coordinator.run_production_loop(mode='paper', duration=2.0, continuous=False)
    )
    
    # Wait a bit for loop to start
    await asyncio.sleep(1.5)
    
    # Check that is_running was set to True
    assert coordinator.is_running, "coordinator.is_running is False - loop didn't start!"
    
    # Wait for duration to complete
    await asyncio.wait_for(loop_task, timeout=5.0)
    
    logger.info("✅ TEST PASSED: Trading loop started successfully with debug mode ON")
    
    # Cleanup
    await coordinator.stop_system()


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRADING LOOP STARTUP TESTS")
    print("="*70)
    print()
    
    try:
        # Test 1: Debug mode OFF (INFO level)
        print("Running Test 1: Debug Mode OFF...")
        await test_trading_loop_starts_without_debug_mode()
        print("✅ Test 1 PASSED\n")
        
        # Test 2: Debug mode ON (DEBUG level)
        print("Running Test 2: Debug Mode ON...")
        await test_trading_loop_starts_with_debug_mode()
        print("✅ Test 2 PASSED\n")
        
        print("="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
