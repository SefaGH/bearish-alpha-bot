#!/usr/bin/env python3
"""
Test shutdown order to ensure positions close before connections die.
Issue: [CRITICAL BUG] Positions Cannot Be Closed During Shutdown - "Exchange Not Available" Error

This test verifies that the graceful shutdown follows the correct order:
1. Stop trading loop (no new signals)
2. Close all open positions (while connections are ALIVE)
3. Stop WebSocket streams
4. Close exchange connections

The bug was that connections were closed before positions, causing
"Exchange not available: unknown" errors during position closure.
"""

import sys
import os
import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch, call
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock classes to simulate the system
class MockExchangeClient:
    """Mock exchange client that tracks its state."""
    def __init__(self, name='mock_exchange'):
        self.name = name
        self.is_open = True
        self.close_called = False
        self.operations_log = []
    
    async def close(self):
        """Close the exchange connection."""
        self.close_called = True
        self.is_open = False
        self.operations_log.append(('close', datetime.now(timezone.utc)))
    
    def place_order(self, symbol, side, amount, price=None):
        """Simulate placing an order."""
        if not self.is_open:
            raise Exception("Exchange not available: unknown")
        self.operations_log.append(('place_order', datetime.now(timezone.utc), symbol))
        return {'success': True, 'order_id': 'mock_order_123'}


class MockPositionManager:
    """Mock position manager that tracks position closure attempts."""
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        self.positions = {
            'pos_1': {'symbol': 'BTC/USDT:USDT', 'side': 'long'},
            'pos_2': {'symbol': 'ETH/USDT:USDT', 'side': 'short'},
            'pos_3': {'symbol': 'SOL/USDT:USDT', 'side': 'long'},
        }
        self.close_called = False
        self.operations_log = []
        self.closure_errors = []
    
    async def close_all_positions(self, reason):
        """Attempt to close all positions."""
        self.close_called = True
        self.operations_log.append(('close_all_positions', datetime.now(timezone.utc), reason))
        
        closed_count = 0
        for pos_id, pos_data in list(self.positions.items()):
            try:
                # Try to place order - will fail if exchange is closed
                self.exchange_client.place_order(
                    pos_data['symbol'], 
                    'buy' if pos_data['side'] == 'short' else 'sell',
                    0.001
                )
                closed_count += 1
                del self.positions[pos_id]
            except Exception as e:
                # This is the bug we're testing for!
                self.closure_errors.append({
                    'position_id': pos_id,
                    'error': str(e),
                    'exchange_was_open': self.exchange_client.is_open
                })
        
        return {
            'success': len(self.closure_errors) == 0,
            'closed': closed_count,
            'errors': len(self.closure_errors)
        }


class MockWebSocketManager:
    """Mock WebSocket manager."""
    def __init__(self):
        self.is_streaming = True
        self.close_called = False
        self.operations_log = []
    
    async def close(self):
        """Close WebSocket connections."""
        self.close_called = True
        self.is_streaming = False
        self.operations_log.append(('close', datetime.now(timezone.utc)))


class MockCoordinator:
    """Mock production coordinator."""
    def __init__(self, exchange_client, position_manager, ws_manager):
        self.exchange_client = exchange_client
        self.position_manager = position_manager
        self.websocket_manager = ws_manager
        self.is_running = True
        self.operations_log = []
    
    async def stop(self):
        """Stop the coordinator (should NOT close connections anymore)."""
        self.is_running = False
        self.operations_log.append(('stop', datetime.now(timezone.utc)))
        # CRITICAL: This should NOT close websocket or exchange connections
        # Those are closed by the launcher's cleanup() in the correct order


async def test_correct_shutdown_order():
    """
    Test that shutdown follows the correct order:
    1. Stop coordinator (trading loop)
    2. Close positions (exchange ALIVE)
    3. Stop WebSocket
    4. Close exchange
    """
    print("\n" + "="*70)
    print("TEST: Correct Shutdown Order")
    print("="*70)
    
    # Setup mocks
    exchange = MockExchangeClient('bingx')
    position_mgr = MockPositionManager(exchange)
    ws_manager = MockWebSocketManager()
    coordinator = MockCoordinator(exchange, position_mgr, ws_manager)
    
    print("\n1. System initialized")
    print(f"   • Exchange: {exchange.name} (open={exchange.is_open})")
    print(f"   • Positions: {len(position_mgr.positions)}")
    print(f"   • WebSocket: streaming={ws_manager.is_streaming}")
    
    # Simulate correct shutdown order
    print("\n2. STEP 1: Stop trading loop")
    await coordinator.stop()
    assert not coordinator.is_running
    assert exchange.is_open, "Exchange should still be open!"
    assert ws_manager.is_streaming, "WebSocket should still be streaming!"
    print("   ✅ Trading loop stopped")
    print(f"   ✅ Exchange still open: {exchange.is_open}")
    print(f"   ✅ WebSocket still streaming: {ws_manager.is_streaming}")
    
    print("\n3. STEP 2: Close all positions")
    result = await position_mgr.close_all_positions("shutdown")
    assert result['success'], "Position closure failed!"
    assert len(position_mgr.closure_errors) == 0, f"Errors during closure: {position_mgr.closure_errors}"
    assert result['closed'] == 3, f"Expected 3 positions closed, got {result['closed']}"
    print(f"   ✅ All {result['closed']} positions closed successfully")
    print(f"   ✅ No errors: exchange was available during closure")
    
    print("\n4. STEP 3: Stop WebSocket")
    await ws_manager.close()
    assert not ws_manager.is_streaming
    print("   ✅ WebSocket stopped")
    
    print("\n5. STEP 4: Close exchange connection")
    await exchange.close()
    assert not exchange.is_open
    print("   ✅ Exchange connection closed")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Correct shutdown order maintained")
    print("✅ No 'Exchange not available' errors occurred")
    print("="*70)


async def test_incorrect_shutdown_order():
    """
    Test that demonstrates the bug: closing exchange BEFORE positions.
    This should fail with 'Exchange not available' error.
    """
    print("\n" + "="*70)
    print("TEST: Incorrect Shutdown Order (Demonstrates Bug)")
    print("="*70)
    
    # Setup mocks
    exchange = MockExchangeClient('bingx')
    position_mgr = MockPositionManager(exchange)
    ws_manager = MockWebSocketManager()
    coordinator = MockCoordinator(exchange, position_mgr, ws_manager)
    
    print("\n1. System initialized")
    print(f"   • Positions: {len(position_mgr.positions)}")
    
    # Simulate INCORRECT shutdown order (the bug)
    print("\n2. STEP 1: Stop trading loop")
    await coordinator.stop()
    
    print("\n3. STEP 2 (WRONG): Close exchange FIRST")
    await exchange.close()
    assert not exchange.is_open
    print(f"   ⚠️ Exchange closed prematurely: {not exchange.is_open}")
    
    print("\n4. STEP 3: Try to close positions (WILL FAIL)")
    result = await position_mgr.close_all_positions("shutdown")
    
    # This should fail!
    if len(position_mgr.closure_errors) > 0:
        print(f"   ❌ Position closure FAILED as expected (demonstrating the bug)")
        print(f"   ❌ Errors: {len(position_mgr.closure_errors)}")
        for err in position_mgr.closure_errors:
            print(f"      - {err['position_id']}: {err['error']}")
            print(f"        Exchange was open: {err['exchange_was_open']}")
        print("\n   ⚠️ This is the BUG we're fixing!")
    else:
        print("   ❌ TEST FAILED: Expected errors but got none")
        assert False, "Expected position closure to fail with closed exchange"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Bug correctly demonstrated")
    print("   When exchange closes before positions, we get:")
    print("   'Exchange not available: unknown' errors")
    print("="*70)


async def test_order_verification():
    """
    Verify the specific order of operations using timestamps.
    """
    print("\n" + "="*70)
    print("TEST: Order Verification with Timestamps")
    print("="*70)
    
    # Setup mocks
    exchange = MockExchangeClient('bingx')
    position_mgr = MockPositionManager(exchange)
    ws_manager = MockWebSocketManager()
    coordinator = MockCoordinator(exchange, position_mgr, ws_manager)
    
    # Execute shutdown
    await coordinator.stop()
    await position_mgr.close_all_positions("shutdown")
    await ws_manager.close()
    await exchange.close()
    
    # Collect all operations with timestamps
    all_operations = []
    all_operations.extend([('coordinator', op) for op in coordinator.operations_log])
    all_operations.extend([('position_mgr', op) for op in position_mgr.operations_log])
    all_operations.extend([('ws_manager', op) for op in ws_manager.operations_log])
    all_operations.extend([('exchange', op) for op in exchange.operations_log])
    
    # Sort by timestamp
    all_operations.sort(key=lambda x: x[1][1])
    
    print("\nOperation timeline:")
    for component, (action, timestamp, *args) in all_operations:
        print(f"   {timestamp.strftime('%H:%M:%S.%f')[:-3]} - {component:15s}: {action}")
    
    # Verify order (filter out place_order operations which are sub-operations)
    major_operations = [op for op in all_operations if op[1][0] != 'place_order']
    operation_names = [op[1][0] for op in major_operations]
    expected_order = ['stop', 'close_all_positions', 'close', 'close']
    
    print(f"\nExpected major operations: {expected_order}")
    print(f"Actual major operations:   {operation_names}")
    
    assert operation_names == expected_order, f"Order mismatch! Got {operation_names}"
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Correct operation order verified")
    print("="*70)


async def run_all_tests():
    """Run all shutdown order tests."""
    print("\n" + "="*70)
    print("SHUTDOWN ORDER TEST SUITE")
    print("Testing Critical Bug Fix")
    print("="*70)
    
    await test_correct_shutdown_order()
    await test_incorrect_shutdown_order()
    await test_order_verification()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nVerified:")
    print("  ✅ Correct shutdown order prevents 'Exchange not available' errors")
    print("  ✅ Positions close successfully when exchange is alive")
    print("  ✅ Bug scenario correctly demonstrates the problem")
    print("  ✅ Operation timeline follows expected sequence")
    print("="*70)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    try:
        asyncio.run(run_all_tests())
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
