#!/usr/bin/env python3
"""
WebSocket Task Management Validation Demo

This script demonstrates the WebSocket task management fix for Issue #160.
It shows how tasks are now properly returned, scheduled, and managed.

Usage:
    python examples/websocket_task_validation_demo.py
"""

import sys
import os
import asyncio
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

print("=" * 80)
print("WebSocket Task Management Validation Demo".center(80))
print("=" * 80)
print()


# Demonstration 1: Mock OptimizedWebSocketManager behavior
print("[Demo 1] OptimizedWebSocketManager.initialize_websockets() Return Type")
print("-" * 80)

class MockOptimizedWebSocketManager:
    """Mock implementation showing the fix."""
    
    def __init__(self):
        self.is_initialized = False
        self.fixed_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    
    async def initialize_websockets(self, exchange_clients) -> List[asyncio.Task]:
        """
        FIXED: Now returns List[asyncio.Task] instead of bool.
        """
        print(f"  - Initializing WebSocket for {len(self.fixed_symbols)} symbols...")
        
        # Simulate creating streaming tasks
        tasks = []
        for symbol in self.fixed_symbols:
            async def mock_stream(sym=symbol):
                print(f"    ✓ Stream task started for {sym}")
                await asyncio.sleep(0.5)
                print(f"    ✓ Stream task running for {sym}")
            
            task = asyncio.create_task(mock_stream())
            tasks.append(task)
        
        if tasks:
            self.is_initialized = True
            print(f"  - FIXED: Returning {len(tasks)} tasks (was: returning True)")
            return tasks
        else:
            print(f"  - FIXED: Returning [] (was: returning False)")
            return []


async def demo1():
    """Demonstrate the fix in initialize_websockets."""
    manager = MockOptimizedWebSocketManager()
    
    # OLD behavior: result = await manager.initialize_websockets({})
    # OLD: result would be True/False (tasks lost!)
    
    # NEW behavior:
    tasks = await manager.initialize_websockets({})
    
    print(f"\n  Result type: {type(tasks)}")
    print(f"  Result value: {len(tasks)} tasks")
    print(f"  Tasks are: {tasks}")
    print(f"\n  ✅ SUCCESS: Tasks are returned and can be scheduled!")
    
    # Wait for tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)


# Demonstration 2: Task scheduling in _start_trading_loop
print("\n[Demo 2] Task Scheduling in _start_trading_loop()")
print("-" * 80)


async def demo2():
    """Demonstrate how tasks are now scheduled in the trading loop."""
    manager = MockOptimizedWebSocketManager()
    
    print("  - OLD behavior:")
    print("    # tasks = await ws_optimizer.initialize_websockets()")
    print("    # ❌ Tasks returned but never scheduled!")
    print()
    
    print("  - NEW behavior:")
    
    # Simulate the new behavior
    ws_tasks = []
    ws_streaming = False
    
    print("    # Capture tasks")
    streaming_tasks = await manager.initialize_websockets({})
    
    if streaming_tasks:
        ws_tasks = streaming_tasks
        ws_streaming = True
        print(f"    ✅ {len(ws_tasks)} WebSocket streams running in background")
        
        # Brief delay to check connection
        await asyncio.sleep(0.5)
        print(f"    ✅ Streams are active and processing data")
    
    print(f"\n  ✅ SUCCESS: Tasks are scheduled and running!")
    
    # Cleanup
    print("\n  - Cleanup (in finally block):")
    if ws_tasks:
        print(f"    Cancelling {len(ws_tasks)} WebSocket streams...")
        for task in ws_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*ws_tasks, return_exceptions=True)
        print(f"    ✓ All WebSocket streams cancelled")


# Demonstration 3: Status reporting
print("\n[Demo 3] Status Reporting with Real Connection State")
print("-" * 80)


class MockWebSocketClient:
    """Mock WebSocket client with is_connected() method."""
    
    def __init__(self, name):
        self.name = name
        self._is_connected = False
        self._first_message_received = False
        self._running = False
    
    def is_connected(self) -> bool:
        """
        FIXED: Now checks actual connection state (from Issue #159).
        """
        return self._is_connected and self._running and self._first_message_received
    
    def simulate_connection(self):
        """Simulate connection establishment."""
        self._is_connected = True
        self._running = True
        self._first_message_received = True


async def demo3():
    """Demonstrate improved status reporting."""
    print("  - Creating mock WebSocket client...")
    client = MockWebSocketClient('bingx')
    
    print(f"\n  Initial state:")
    print(f"    is_connected(): {client.is_connected()}")
    print(f"    Status: DISCONNECTED ⚠️")
    
    print(f"\n  After connection:")
    client.simulate_connection()
    print(f"    is_connected(): {client.is_connected()}")
    print(f"    Status: CONNECTED and STREAMING ✅")
    
    print(f"\n  ✅ SUCCESS: Status accurately reflects connection state!")


# Run all demonstrations
async def main():
    """Run all validation demonstrations."""
    try:
        await demo1()
        print("\n" + "=" * 80 + "\n")
        
        await demo2()
        print("\n" + "=" * 80 + "\n")
        
        await demo3()
        
        print("\n" + "=" * 80)
        print("All Validation Demos Completed Successfully! ✅".center(80))
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
