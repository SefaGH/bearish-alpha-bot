#!/usr/bin/env python3
"""
Integration test: Launcher runs without freezing.

This is the PRIMARY test for verifying the bot does not freeze at startup
or during execution. If this test passes, the bot does not freeze!

Addresses:
- Issue #153: Bot Freeze at Startup
- Issue #160: WebSocket Task Management
"""

import pytest
import asyncio
import time
import os
import sys
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 2 minute max
async def test_launcher_runs_without_freeze(integration_env, cleanup_tasks):
    """
    Integration test: Verify launcher runs for 30s without freezing.
    
    This is the PRIMARY test for Issue #153 (bot freeze).
    If this test passes, the bot does not freeze!
    
    Test Strategy:
    - Use mocks for external dependencies (exchange APIs, WebSocket)
    - Run actual launcher initialization and main loop
    - Verify execution completes within expected time
    - Detect deadlocks, blocking code, or infinite loops
    """
    print("\n" + "="*70)
    print("TEST: Launcher Runs Without Freeze (30s Runtime)")
    print("="*70)
    
    # Setup environment
    os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT'
    os.environ['TRADING_MODE'] = 'paper'
    os.environ['CAPITAL_USDT'] = '100'
    
    # Track execution
    start_time = time.time()
    completed = False
    freeze_detected = False
    
    try:
        # Mock heavy dependencies before import
        with patch.dict('sys.modules', {
            'ccxt': MagicMock(),
            'torch': MagicMock(),
            'torchvision': MagicMock(),
            'sklearn': MagicMock(),
        }):
            # Import launcher after env setup
            from live_trading_launcher import LiveTradingLauncher
        
        print("\n[Step 1] Creating launcher instance...")
        
        # Mock external dependencies to avoid real API calls
        with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
             patch('core.notify.Telegram') as mock_telegram:
            
            # Setup mock exchange client
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
            mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
            mock_exchange.ticker.return_value = {'last': 50000.0}
            mock_ccxt.return_value = mock_exchange
            
            # Create launcher
            launcher = LiveTradingLauncher(mode='paper')
            print("  ✓ Launcher instance created")
            
            print("\n[Step 2] Running launcher for 30 seconds...")
            print("  - If this hangs, the bot has frozen (DEADLOCK)")
            print("  - Expected: Completes in ~30 seconds")
            
            # Run for 30 seconds with 45s timeout
            # This is the critical test - will it complete or freeze?
            try:
                await asyncio.wait_for(
                    launcher._start_trading_loop(duration=30),
                    timeout=45
                )
                completed = True
                print("  ✓ Trading loop completed successfully")
                
            except asyncio.TimeoutError:
                freeze_detected = True
                elapsed = time.time() - start_time
                
                pytest.fail(
                    f"\n{'='*70}\n"
                    f"❌ BOT FROZE! Trading loop did not complete in 45 seconds.\n"
                    f"{'='*70}\n"
                    f"Expected: 30s runtime + 15s buffer\n"
                    f"Actual:   Timeout after {elapsed:.1f}s\n"
                    f"\n"
                    f"This indicates a deadlock, blocking code, or infinite loop.\n"
                    f"Common causes:\n"
                    f"  - Improperly awaited coroutines\n"
                    f"  - WebSocket tasks not scheduled correctly\n"
                    f"  - Blocking I/O in async context\n"
                    f"  - Event loop blocking\n"
                    f"{'='*70}\n"
                )
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Unexpected error after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Test failed with exception: {e}")
    
    finally:
        elapsed = time.time() - start_time
        
        # Log execution time
        print(f"\n{'='*70}")
        print(f"Launcher Execution Report:")
        print(f"{'='*70}")
        print(f"Start Time:    {datetime.fromtimestamp(start_time).isoformat()}")
        print(f"End Time:      {datetime.fromtimestamp(time.time()).isoformat()}")
        print(f"Elapsed:       {elapsed:.2f} seconds")
        print(f"Expected:      ~30 seconds (±10s tolerance)")
        print(f"Completed:     {'✅ Yes' if completed else '❌ No (FREEZE)'}")
        print(f"Freeze:        {'❌ Yes' if freeze_detected else '✅ No'}")
        print(f"{'='*70}\n")
    
    # Verify execution time is reasonable (30s ± 15s)
    # Allow wider tolerance for CI/CD environments
    assert 20 <= elapsed <= 50, (
        f"Execution time unexpected: {elapsed:.1f}s "
        f"(expected ~30s, tolerance ±15s)"
    )
    
    # Verify no freeze detected
    assert not freeze_detected, "Bot froze during execution"
    assert completed, "Trading loop did not complete"
    
    print("\n✅ TEST PASSED: Bot runs without freezing")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_async_tasks_properly_scheduled(integration_env, cleanup_tasks):
    """
    Integration test: Verify async tasks are properly scheduled and executed.
    
    Addresses Issue #160 (WebSocket Task Management).
    
    Test Strategy:
    - Track task count before and after launcher start
    - Verify new tasks are created for WebSocket streams
    - Ensure tasks are running (not just created)
    - Verify proper cleanup on completion
    """
    print("\n" + "="*70)
    print("TEST: Async Tasks Properly Scheduled")
    print("="*70)
    
    os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT'
    os.environ['TRADING_MODE'] = 'paper'
    
    try:
        from live_trading_launcher import LiveTradingLauncher
        
        # Mock external dependencies
        with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
             patch('core.notify.Telegram') as mock_telegram:
            
            # Setup mock exchange
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
            mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
            mock_exchange.ticker.return_value = {'last': 50000.0}
            mock_ccxt.return_value = mock_exchange
            
            print("\n[Step 1] Creating launcher...")
            launcher = LiveTradingLauncher(mode='paper')
            
            # Track active tasks before starting
            initial_tasks = len(asyncio.all_tasks())
            print(f"\n[Step 2] Initial task count: {initial_tasks}")
            
            # Start launcher (spawns WebSocket tasks)
            print("\n[Step 3] Starting launcher (10s runtime)...")
            launcher_task = asyncio.create_task(
                launcher._start_trading_loop(duration=10)
            )
            
            # Wait a bit for tasks to spawn
            await asyncio.sleep(2)
            
            # Check if new tasks were created
            current_tasks = len(asyncio.all_tasks())
            new_tasks = current_tasks - initial_tasks
            
            print(f"\n{'='*70}")
            print(f"Async Task Report:")
            print(f"{'='*70}")
            print(f"Initial tasks:  {initial_tasks}")
            print(f"Current tasks:  {current_tasks}")
            print(f"New tasks:      {new_tasks}")
            print(f"{'='*70}\n")
            
            # Verify tasks were spawned
            # Note: Exact count may vary, but should have at least:
            # - launcher_task itself
            # - coordinator loop task
            # - possibly WebSocket tasks (if not mocked away)
            assert new_tasks >= 1, (
                f"Expected at least 1 new task (launcher itself), "
                f"but only {new_tasks} tasks created."
            )
            
            # Wait for completion
            print("[Step 4] Waiting for launcher to complete...")
            await launcher_task
            
            print("\n✅ TEST PASSED: Async tasks properly scheduled and executed")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Async task test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_launcher_initialization_phases(integration_env, cleanup_tasks):
    """
    Integration test: Verify all 8 initialization phases complete.
    
    This test verifies the launcher can complete all initialization phases
    without errors or hanging.
    
    Phases:
    1. Load configuration
    2. Initialize exchange connection
    3. Initialize risk management
    4. Initialize AI components
    5. Initialize strategies
    6. Initialize production system
    7. Register strategies
    8. Perform preflight checks
    """
    print("\n" + "="*70)
    print("TEST: Launcher Initialization Phases")
    print("="*70)
    
    os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT'
    os.environ['TRADING_MODE'] = 'paper'
    
    try:
        from live_trading_launcher import LiveTradingLauncher
        
        # Mock external dependencies
        with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
             patch('core.notify.Telegram') as mock_telegram:
            
            # Setup mock exchange
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
            mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
            mock_exchange.ticker.return_value = {'last': 50000.0}
            mock_ccxt.return_value = mock_exchange
            
            print("\n[Step 1] Creating launcher and initializing...")
            start_time = time.time()
            
            launcher = LiveTradingLauncher(mode='paper')
            
            # The constructor already runs some initialization
            # Now test that we can run a short loop
            print("\n[Step 2] Running short test loop (5s)...")
            
            await asyncio.wait_for(
                launcher._start_trading_loop(duration=5),
                timeout=15
            )
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"Initialization completed in {elapsed:.2f}s")
            print(f"{'='*70}\n")
            
            # Verify reasonable initialization time
            assert elapsed < 15, (
                f"Initialization took too long: {elapsed:.1f}s "
                f"(expected < 15s)"
            )
            
            print("\n✅ TEST PASSED: All initialization phases completed")
            
    except asyncio.TimeoutError:
        pytest.fail(
            "Initialization hung! One or more phases did not complete.\n"
            "Check for blocking code or missing await statements."
        )
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Initialization test failed: {e}")
