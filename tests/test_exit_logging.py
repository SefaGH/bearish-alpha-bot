#!/usr/bin/env python3
"""
Test enhanced exit logging and session summaries.
Issue #134: Validate Exit Logic - Enhanced logging for SL/TP/Trailing Stop.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.position_manager import AdvancedPositionManager, ExitReason


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


async def test_exit_logging():
    """Test enhanced exit event logging."""
    print("\n" + "="*70)
    print("TEST: Enhanced Exit Event Logging")
    print("="*70)
    
    # Initialize position manager
    portfolio_mgr = MockPortfolioManager()
    risk_mgr = MockRiskManager()
    position_mgr = AdvancedPositionManager(portfolio_mgr, risk_mgr)
    
    # Create test positions with different exit scenarios
    test_positions = [
        {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 110000.0,
            'amount': 0.001,
            'stop_loss': 109500.0,
            'take_profit': 111200.0,
            'exit_price': 109500.0,  # Stop loss hit
            'exit_reason': ExitReason.STOP_LOSS.value
        },
        {
            'symbol': 'ETH/USDT:USDT',
            'side': 'long',
            'entry_price': 3500.0,
            'amount': 0.05,
            'stop_loss': 3450.0,
            'take_profit': 3552.5,
            'exit_price': 3552.5,  # Take profit hit
            'exit_reason': ExitReason.TAKE_PROFIT.value
        },
        {
            'symbol': 'SOL/USDT:USDT',
            'side': 'long',
            'entry_price': 145.0,
            'amount': 1.0,
            'stop_loss': 142.0,
            'take_profit': 150.0,
            'exit_price': 148.15,  # Trailing stop hit
            'exit_reason': ExitReason.TRAILING_STOP.value
        },
        {
            'symbol': 'BNB/USDT:USDT',
            'side': 'short',
            'entry_price': 620.0,
            'amount': 0.2,
            'stop_loss': 630.0,
            'take_profit': 610.0,
            'exit_price': 610.0,  # Take profit hit (short)
            'exit_reason': ExitReason.TAKE_PROFIT.value
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'side': 'short',
            'entry_price': 0.65,
            'amount': 500.0,
            'stop_loss': 0.68,
            'take_profit': 0.62,
            'exit_price': 0.68,  # Stop loss hit (short)
            'exit_reason': ExitReason.STOP_LOSS.value
        }
    ]
    
    print("\n1. Creating and Closing Test Positions")
    print("-" * 70)
    
    # Open and close each position
    for i, pos_data in enumerate(test_positions):
        # Create a mock signal for position opening
        signal = {
            'symbol': pos_data['symbol'],
            'side': pos_data['side'],
            'entry': pos_data['entry_price'],
            'stop': pos_data['stop_loss'],
            'target': pos_data['take_profit'],
            'strategy': 'test_strategy',
            'exchange': 'mock'
        }
        
        # Mock execution result
        execution_result = {
            'success': True,
            'avg_price': pos_data['entry_price'],
            'filled_amount': pos_data['amount']
        }
        
        # Open position
        result = await position_mgr.open_position(signal, execution_result)
        assert result['success'], f"Failed to open position {i+1}"
        position_id = result['position_id']
        
        print(f"\nPosition {i+1} opened: {position_id}")
        
        # Close position with specified exit
        close_result = await position_mgr.close_position(
            position_id,
            pos_data['exit_price'],
            pos_data['exit_reason']
        )
        
        assert close_result['success'], f"Failed to close position {i+1}"
        print(f"Position {i+1} closed with reason: {pos_data['exit_reason']}")
    
    print("\n2. Testing Exit Statistics")
    print("-" * 70)
    
    # Get exit statistics
    stats = position_mgr.get_exit_statistics()
    
    print(f"\nExit Statistics:")
    print(f"  Total Exits: {stats['total_exits']}")
    print(f"  Stop Loss Count: {stats['stop_loss_count']}")
    print(f"  Take Profit Count: {stats['take_profit_count']}")
    print(f"  Trailing Stop Count: {stats['trailing_stop_count']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Total P&L: ${stats['total_pnl']:+.2f}")
    
    # Assertions
    assert stats['total_exits'] == 5, "Should have 5 total exits"
    assert stats['stop_loss_count'] == 2, "Should have 2 stop loss exits"
    assert stats['take_profit_count'] == 2, "Should have 2 take profit exits"
    assert stats['trailing_stop_count'] == 1, "Should have 1 trailing stop exit"
    assert stats['winning_trades'] == 3, "Should have 3 winning trades"
    assert stats['losing_trades'] == 2, "Should have 2 losing trades"
    
    print("\n✅ All exit statistics assertions passed!")
    
    print("\n3. Testing Session Summary Logging")
    print("-" * 70)
    
    # Log the session summary (this will print the formatted summary)
    position_mgr.log_exit_summary()
    
    print("\n✅ Session summary logged successfully!")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


if __name__ == '__main__':
    # Configure logging to see our enhanced exit logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    asyncio.run(test_exit_logging())
