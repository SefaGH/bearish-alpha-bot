#!/usr/bin/env python3
"""
Test exit statistics with no closed positions.
Issue #136: Fix KeyError when no positions have been closed.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.position_manager import AdvancedPositionManager


# Mock classes for testing
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


async def test_empty_exit_statistics():
    """Test exit statistics when no positions have been closed."""
    print("\n" + "="*70)
    print("TEST: Exit Statistics with Empty Closed Positions")
    print("="*70)
    
    # Initialize position manager
    portfolio_mgr = MockPortfolioManager()
    risk_mgr = MockRiskManager()
    position_mgr = AdvancedPositionManager(portfolio_mgr, risk_mgr)
    
    print("\n1. Testing get_exit_statistics() with no closed positions")
    print("-" * 70)
    
    # Get exit statistics when no positions have been closed
    stats = position_mgr.get_exit_statistics()
    
    print(f"\nExit Statistics (no closed positions):")
    print(f"  Total Exits: {stats['total_exits']}")
    print(f"  Stop Loss Count: {stats['stop_loss_count']}")
    print(f"  Take Profit Count: {stats['take_profit_count']}")
    print(f"  Trailing Stop Count: {stats['trailing_stop_count']}")
    print(f"  Manual Close Count: {stats['manual_close_count']}")
    print(f"  Liquidation Count: {stats['liquidation_count']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Total P&L: ${stats['total_pnl']:+.2f}")
    
    # Assertions - all should be 0 or empty
    assert stats['total_exits'] == 0, "Should have 0 total exits"
    assert stats['stop_loss_count'] == 0, "Should have 0 stop loss exits"
    assert stats['take_profit_count'] == 0, "Should have 0 take profit exits"
    assert stats['trailing_stop_count'] == 0, "Should have 0 trailing stop exits"
    assert stats['manual_close_count'] == 0, "Should have 0 manual close exits"
    assert stats['liquidation_count'] == 0, "Should have 0 liquidation exits"
    assert stats['winning_trades'] == 0, "Should have 0 winning trades"
    assert stats['losing_trades'] == 0, "Should have 0 losing trades"
    assert stats['win_rate'] == 0.0, "Win rate should be 0.0"
    assert stats['total_pnl'] == 0.0, "Total P&L should be 0.0"
    assert stats['exits_by_reason'] == {}, "exits_by_reason should be empty dict"
    
    print("\n✅ All assertions passed for empty closed positions!")
    
    print("\n2. Testing log_exit_summary() with no closed positions")
    print("-" * 70)
    
    # This should NOT raise KeyError
    try:
        position_mgr.log_exit_summary()
        print("\n✅ log_exit_summary() executed without KeyError!")
    except KeyError as e:
        print(f"\n❌ KeyError occurred: {e}")
        raise
    
    print("\n3. Opening and monitoring a position (but not closing it)")
    print("-" * 70)
    
    # Create a mock signal for position opening
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'long',
        'entry': 110000.0,
        'stop': 109500.0,
        'target': 111200.0,
        'strategy': 'test_strategy',
        'exchange': 'mock'
    }
    
    # Mock execution result
    execution_result = {
        'success': True,
        'avg_price': 110000.0,
        'filled_amount': 0.001
    }
    
    # Open position (but don't close it)
    result = await position_mgr.open_position(signal, execution_result)
    assert result['success'], "Failed to open position"
    position_id = result['position_id']
    
    print(f"\nPosition opened: {position_id}")
    print("Position is still OPEN (not closed)")
    
    # Try to get exit statistics again - should still be empty
    stats = position_mgr.get_exit_statistics()
    assert stats['total_exits'] == 0, "Should still have 0 exits (position not closed)"
    
    print("\n✅ Exit statistics still show 0 exits (as expected)")
    
    # Try log_exit_summary again - should work
    try:
        position_mgr.log_exit_summary()
        print("\n✅ log_exit_summary() still works with open position!")
    except KeyError as e:
        print(f"\n❌ KeyError occurred: {e}")
        raise
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("Issue #136 is FIXED - No KeyError on empty closed positions")
    print("="*70)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    asyncio.run(test_empty_exit_statistics())
