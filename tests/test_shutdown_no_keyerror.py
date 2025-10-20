#!/usr/bin/env python
"""
Test shutdown scenario without KeyError.
Issue #136: Verify bot can shutdown gracefully even with no closed positions.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.position_manager import AdvancedPositionManager


# Mock classes
class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    def __init__(self):
        self.cfg = {}
        self.portfolio_state = {}


class MockRiskManager:
    """Mock risk manager for testing."""
    def register_position(self, position_id, position):
        pass
    
    def close_position(self, position_id, exit_price, realized_pnl):
        pass


async def simulate_shutdown_with_no_trades():
    """
    Simulate a shutdown scenario where the bot ran but never executed any trades.
    This was the scenario causing KeyError: 'stop_loss_count'.
    """
    print("\n" + "="*70)
    print("TEST: Shutdown Simulation with No Trades")
    print("="*70)
    
    # Initialize position manager (simulating engine startup)
    portfolio_mgr = MockPortfolioManager()
    risk_mgr = MockRiskManager()
    position_mgr = AdvancedPositionManager(portfolio_mgr, risk_mgr)
    
    print("\n1. Simulating Bot Startup")
    print("-" * 70)
    print("✅ Position manager initialized")
    print("✅ Engine running...")
    print("   (No signals generated or executed)")
    
    print("\n2. Simulating Bot Shutdown")
    print("-" * 70)
    print("Stopping engine...")
    
    # This is what happens in live_trading_engine.stop_live_trading()
    # It should NOT raise KeyError even with no closed positions
    try:
        position_mgr.log_exit_summary()
        print("✅ Exit summary logged successfully")
        print("✅ No KeyError occurred!")
    except KeyError as e:
        print(f"❌ FAILED: KeyError occurred: {e}")
        raise
    
    print("\n✅ Shutdown completed gracefully")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED!")
    print("Issue #136 FIXED: Bot can shutdown without crashing")
    print("="*70)


async def simulate_shutdown_with_open_position():
    """
    Simulate shutdown with an open position (not closed yet).
    Exit stats should still work.
    """
    print("\n" + "="*70)
    print("TEST: Shutdown with Open Position (Not Closed)")
    print("="*70)
    
    portfolio_mgr = MockPortfolioManager()
    risk_mgr = MockRiskManager()
    position_mgr = AdvancedPositionManager(portfolio_mgr, risk_mgr)
    
    print("\n1. Opening a position")
    print("-" * 70)
    
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',
        'entry': 110000.0,
        'stop': 109500.0,
        'target': 111200.0,
        'strategy': 'test_strategy',
        'exchange': 'mock'
    }
    
    execution_result = {
        'success': True,
        'avg_price': 110000.0,
        'filled_amount': 0.001
    }
    
    result = await position_mgr.open_position(signal, execution_result)
    assert result['success'], "Failed to open position"
    print(f"✅ Position opened: {result['position_id']}")
    print("   (Position is still OPEN, not closed)")
    
    print("\n2. Simulating Shutdown with Open Position")
    print("-" * 70)
    
    # Shutdown should work even with open positions
    try:
        position_mgr.log_exit_summary()
        print("✅ Exit summary logged (shows 0 exits)")
        print("✅ No KeyError with open position!")
    except KeyError as e:
        print(f"❌ FAILED: KeyError occurred: {e}")
        raise
    
    print("\n✅ Shutdown completed with open position")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED!")
    print("="*70)


async def run_all_shutdown_tests():
    """Run all shutdown simulation tests."""
    print("\n" + "="*70)
    print("SHUTDOWN SIMULATION TEST SUITE")
    print("Testing Issue #136 Fix")
    print("="*70)
    
    await simulate_shutdown_with_no_trades()
    await simulate_shutdown_with_open_position()
    
    print("\n" + "="*70)
    print("✅ ALL SHUTDOWN TESTS PASSED!")
    print("The bot can now shutdown gracefully in all scenarios:")
    print("  • No trades executed")
    print("  • Open positions (not closed)")
    print("  • Empty closed_positions list")
    print("="*70)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    asyncio.run(run_all_shutdown_tests())
