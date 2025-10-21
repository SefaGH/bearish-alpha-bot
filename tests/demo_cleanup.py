#!/usr/bin/env python3
"""
Demonstration script showing resource cleanup in action.

This script demonstrates:
1. How cleanup is called automatically on exit
2. Cleanup handles errors gracefully
3. Resources are properly released

Run with: python tests/demo_cleanup.py
"""

import sys
import os
import asyncio
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Set env var to skip Python version check
os.environ['SKIP_PYTHON_VERSION_CHECK'] = '1'

# We'll create a minimal mock to demonstrate
print("=" * 70)
print("🔍 RESOURCE CLEANUP DEMONSTRATION")
print("=" * 70)
print()

print("This demonstration shows how the cleanup system works:")
print()
print("1. Resources are tracked during initialization")
print("2. Cleanup is called automatically in finally block")
print("3. Cleanup is idempotent (safe to call multiple times)")
print("4. Cleanup continues even if some steps fail")
print("5. Proper exit codes are returned")
print()
print("=" * 70)


class MockResource:
    """Mock resource that tracks if it was closed."""
    
    def __init__(self, name):
        self.name = name
        self.closed = False
        print(f"  📦 {name} opened")
    
    async def close(self):
        """Close the resource."""
        if self.closed:
            print(f"  ⚠️  {self.name} already closed")
        else:
            print(f"  ✅ {self.name} closed")
            self.closed = True


class MockLauncher:
    """Mock launcher demonstrating cleanup pattern."""
    
    def __init__(self):
        self._cleanup_done = False
        self.resources = []
    
    async def initialize(self):
        """Initialize mock resources."""
        print("\n📋 INITIALIZATION:")
        self.resources = [
            MockResource("Exchange Connection"),
            MockResource("WebSocket Stream"),
            MockResource("Production System"),
        ]
    
    async def cleanup(self):
        """Cleanup all resources (idempotent)."""
        if self._cleanup_done:
            print("\n🧹 CLEANUP: Already completed, skipping")
            return
        
        print("\n🧹 CLEANUP: Starting...")
        
        # Close all resources
        for resource in self.resources:
            try:
                await resource.close()
            except Exception as e:
                print(f"  ⚠️  Error closing {resource.name}: {e}")
        
        self._cleanup_done = True
        print("  ✅ Cleanup completed\n")
    
    async def run(self, scenario="normal"):
        """Run with different scenarios."""
        try:
            await self.initialize()
            
            print(f"\n▶️  RUNNING: {scenario} scenario")
            
            if scenario == "normal":
                print("  ✓ Task 1 completed")
                print("  ✓ Task 2 completed")
                print("  ✓ All tasks completed")
                return 0
            
            elif scenario == "error":
                print("  ✓ Task 1 completed")
                print("  ❌ Task 2 failed!")
                raise RuntimeError("Simulated error")
            
            elif scenario == "interrupt":
                print("  ✓ Task 1 completed")
                print("  ⚠️  Keyboard interrupt!")
                raise KeyboardInterrupt()
        
        except KeyboardInterrupt:
            print("  🛑 Interrupted by user")
            return 130
        
        except Exception as e:
            print(f"  💥 Error: {e}")
            return 1
        
        finally:
            # ✅ CRITICAL: Always cleanup!
            await self.cleanup()


async def demo_scenario(name, scenario):
    """Run a demonstration scenario."""
    print("=" * 70)
    print(f"SCENARIO: {name}")
    print("=" * 70)
    
    launcher = MockLauncher()
    exit_code = await launcher.run(scenario)
    
    print(f"📊 RESULT: Exit code = {exit_code}")
    print()
    
    return exit_code


async def demo_idempotency():
    """Demonstrate cleanup idempotency."""
    print("=" * 70)
    print("SCENARIO: Multiple Cleanup Calls")
    print("=" * 70)
    
    launcher = MockLauncher()
    await launcher.initialize()
    
    print("\n▶️  RUNNING: Testing idempotency")
    print("  ✓ Task completed")
    
    # Call cleanup multiple times
    print("\n🧹 Calling cleanup (1/3)...")
    await launcher.cleanup()
    
    print("🧹 Calling cleanup (2/3)...")
    await launcher.cleanup()
    
    print("🧹 Calling cleanup (3/3)...")
    await launcher.cleanup()
    
    print("📊 RESULT: Cleanup is idempotent - safe to call multiple times")
    print()


async def main():
    """Run all demonstrations."""
    print()
    
    # Demo 1: Normal exit
    await demo_scenario("Normal Exit (Success)", "normal")
    
    # Demo 2: Error during execution
    await demo_scenario("Error During Execution", "error")
    
    # Demo 3: Keyboard interrupt
    await demo_scenario("Keyboard Interrupt (Ctrl+C)", "interrupt")
    
    # Demo 4: Idempotency
    await demo_idempotency()
    
    # Summary
    print("=" * 70)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. ✅ Cleanup runs automatically in finally block")
    print("2. ✅ Cleanup is idempotent (safe to call multiple times)")
    print("3. ✅ Cleanup continues even if some steps fail")
    print("4. ✅ Proper exit codes returned (0=success, 1=error, 130=interrupt)")
    print("5. ✅ Resources always released before exit")
    print()
    print("This prevents:")
    print("  ❌ 'Unclosed client session' warnings")
    print("  ❌ 'requires to release all resources' warnings")
    print("  ❌ Resource leaks that cause freezes")
    print("  ❌ Ports/sockets left occupied")
    print()
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
