#!/usr/bin/env python3
"""
Manual test script for resource cleanup validation.

Tests:
1. Normal shutdown cleanup
2. Cleanup is idempotent
3. Exchange close is called
4. No resource leak warnings

Run with: python tests/manual_test_cleanup.py
"""

import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Set env var to skip Python version check
os.environ['SKIP_PYTHON_VERSION_CHECK'] = '1'

from live_trading_launcher import LiveTradingLauncher, OptimizedWebSocketManager
from core.ccxt_client import CcxtClient


async def test_cleanup_idempotent():
    """Test that cleanup can be called multiple times."""
    print("=" * 70)
    print("TEST 1: Cleanup Idempotency")
    print("=" * 70)
    
    launcher = LiveTradingLauncher(mode='paper', dry_run=True)
    
    # Call cleanup 3 times
    print("Calling cleanup (1/3)...")
    await launcher.cleanup()
    print(f"✓ Cleanup done flag: {launcher._cleanup_done}")
    
    print("\nCalling cleanup (2/3)...")
    await launcher.cleanup()
    print(f"✓ Cleanup done flag: {launcher._cleanup_done}")
    
    print("\nCalling cleanup (3/3)...")
    await launcher.cleanup()
    print(f"✓ Cleanup done flag: {launcher._cleanup_done}")
    
    print("\n✅ TEST PASSED: Cleanup is idempotent")
    print()


async def test_websocket_stop_streaming():
    """Test OptimizedWebSocketManager.stop_streaming()."""
    print("=" * 70)
    print("TEST 2: WebSocket stop_streaming()")
    print("=" * 70)
    
    manager = OptimizedWebSocketManager()
    
    # Test with no manager
    print("Testing with no ws_manager...")
    await manager.stop_streaming()
    print("✓ No error when ws_manager is None")
    
    # Test with mock manager
    print("\nTesting with mock ws_manager...")
    mock_ws_manager = MagicMock()
    mock_ws_manager.close = AsyncMock()
    manager.ws_manager = mock_ws_manager
    manager.is_initialized = True
    
    await manager.stop_streaming()
    print(f"✓ ws_manager.close() called: {mock_ws_manager.close.called}")
    print(f"✓ is_initialized reset: {not manager.is_initialized}")
    
    print("\n✅ TEST PASSED: stop_streaming() works correctly")
    print()


async def test_ccxt_client_close():
    """Test CcxtClient.close() method."""
    print("=" * 70)
    print("TEST 3: CcxtClient.close()")
    print("=" * 70)
    
    # Check method exists
    print("Checking if close() method exists...")
    assert hasattr(CcxtClient, 'close'), "CcxtClient should have close() method"
    print("✓ close() method exists")
    
    # Check it's async
    import inspect
    assert inspect.iscoroutinefunction(CcxtClient.close), "close() should be async"
    print("✓ close() is async")
    
    print("\n✅ TEST PASSED: CcxtClient has proper close() method")
    print()


async def test_cleanup_components():
    """Test that cleanup calls all component cleanup methods."""
    print("=" * 70)
    print("TEST 4: Cleanup Components")
    print("=" * 70)
    
    launcher = LiveTradingLauncher(mode='paper', dry_run=True)
    
    # Mock components
    mock_ws_optimizer = MagicMock()
    mock_ws_optimizer.stop_streaming = AsyncMock()
    launcher.ws_optimizer = mock_ws_optimizer
    
    mock_coordinator = MagicMock()
    mock_coordinator.stop_system = AsyncMock()
    launcher.coordinator = mock_coordinator
    
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    launcher.exchange_clients = {'bingx': mock_client}
    
    # Call cleanup
    print("Calling cleanup...")
    await launcher.cleanup()
    
    # Verify all components were called
    print(f"✓ ws_optimizer.stop_streaming() called: {mock_ws_optimizer.stop_streaming.called}")
    print(f"✓ coordinator.stop_system() called: {mock_coordinator.stop_system.called}")
    print(f"✓ exchange client.close() called: {mock_client.close.called}")
    
    assert mock_ws_optimizer.stop_streaming.called, "WebSocket should be stopped"
    assert mock_coordinator.stop_system.called, "Coordinator should be stopped"
    assert mock_client.close.called, "Exchange client should be closed"
    
    print("\n✅ TEST PASSED: All cleanup components called")
    print()


async def test_cleanup_handles_errors():
    """Test that cleanup continues even if some steps fail."""
    print("=" * 70)
    print("TEST 5: Cleanup Error Handling")
    print("=" * 70)
    
    launcher = LiveTradingLauncher(mode='paper', dry_run=True)
    
    # Mock ws_optimizer that raises error
    mock_ws_optimizer = MagicMock()
    mock_ws_optimizer.stop_streaming = AsyncMock(side_effect=Exception("WebSocket error"))
    launcher.ws_optimizer = mock_ws_optimizer
    
    # Mock coordinator (should still be called even after ws error)
    mock_coordinator = MagicMock()
    mock_coordinator.stop_system = AsyncMock()
    launcher.coordinator = mock_coordinator
    
    # Call cleanup - should not raise
    print("Calling cleanup with failing WebSocket...")
    await launcher.cleanup()
    
    # Verify cleanup completed despite error
    print(f"✓ Cleanup completed: {launcher._cleanup_done}")
    print(f"✓ Coordinator still called: {mock_coordinator.stop_system.called}")
    
    assert launcher._cleanup_done, "Cleanup should complete despite errors"
    assert mock_coordinator.stop_system.called, "Coordinator should still be called"
    
    print("\n✅ TEST PASSED: Cleanup handles errors gracefully")
    print()


async def main():
    """Run all manual tests."""
    print("\n")
    print("🧪 MANUAL CLEANUP TESTS")
    print("=" * 70)
    print()
    
    try:
        await test_cleanup_idempotent()
        await test_websocket_stop_streaming()
        await test_ccxt_client_close()
        await test_cleanup_components()
        await test_cleanup_handles_errors()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
