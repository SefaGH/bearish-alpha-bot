"""
Unit tests for P&L calculation utilities.

Tests all P&L calculation functions to ensure correct calculations
for both long and short positions.
"""

import pytest
from src.utils.pnl_calculator import (
    calculate_unrealized_pnl,
    calculate_realized_pnl,
    calculate_pnl_percentage,
    calculate_return_percentage,
    calculate_position_value
)


class TestUnrealizedPnL:
    """Test unrealized P&L calculations."""
    
    def test_long_position_profit(self):
        """Test long position with profit."""
        pnl = calculate_unrealized_pnl('long', 50000, 51000, 0.1)
        assert pnl == 100.0
    
    def test_long_position_loss(self):
        """Test long position with loss."""
        pnl = calculate_unrealized_pnl('long', 50000, 49000, 0.1)
        assert pnl == -100.0
    
    def test_short_position_profit(self):
        """Test short position with profit."""
        pnl = calculate_unrealized_pnl('short', 50000, 49000, 0.1)
        assert pnl == 100.0
    
    def test_short_position_loss(self):
        """Test short position with loss."""
        pnl = calculate_unrealized_pnl('short', 50000, 51000, 0.1)
        assert pnl == -100.0
    
    def test_buy_side_alias(self):
        """Test 'buy' side works same as 'long'."""
        pnl_long = calculate_unrealized_pnl('long', 50000, 51000, 0.1)
        pnl_buy = calculate_unrealized_pnl('buy', 50000, 51000, 0.1)
        assert pnl_long == pnl_buy
    
    def test_sell_side_alias(self):
        """Test 'sell' side works same as 'short'."""
        pnl_short = calculate_unrealized_pnl('short', 50000, 49000, 0.1)
        pnl_sell = calculate_unrealized_pnl('sell', 50000, 49000, 0.1)
        assert pnl_short == pnl_sell
    
    def test_zero_price_movement(self):
        """Test P&L when price doesn't move."""
        pnl = calculate_unrealized_pnl('long', 50000, 50000, 0.1)
        assert pnl == 0.0
    
    def test_fractional_amounts(self):
        """Test with fractional position sizes."""
        pnl = calculate_unrealized_pnl('long', 100, 110, 0.5)
        assert pnl == 5.0


class TestRealizedPnL:
    """Test realized P&L calculations."""
    
    def test_long_position_profit(self):
        """Test realized profit on long position."""
        pnl = calculate_realized_pnl('long', 50000, 51000, 0.1)
        assert pnl == 100.0
    
    def test_long_position_loss(self):
        """Test realized loss on long position."""
        pnl = calculate_realized_pnl('long', 50000, 49000, 0.1)
        assert pnl == -100.0
    
    def test_short_position_profit(self):
        """Test realized profit on short position."""
        pnl = calculate_realized_pnl('short', 50000, 49000, 0.1)
        assert pnl == 100.0
    
    def test_short_position_loss(self):
        """Test realized loss on short position."""
        pnl = calculate_realized_pnl('short', 50000, 51000, 0.1)
        assert pnl == -100.0
    
    def test_consistency_with_unrealized(self):
        """Test realized PnL matches unrealized PnL calculation."""
        unrealized = calculate_unrealized_pnl('long', 50000, 51000, 0.1)
        realized = calculate_realized_pnl('long', 50000, 51000, 0.1)
        assert unrealized == realized


class TestPnLPercentage:
    """Test P&L percentage calculations."""
    
    def test_positive_pnl_percentage(self):
        """Test positive P&L percentage."""
        pnl_pct = calculate_pnl_percentage(100, 50000, 0.1)
        assert pnl_pct == 2.0
    
    def test_negative_pnl_percentage(self):
        """Test negative P&L percentage."""
        pnl_pct = calculate_pnl_percentage(-150, 50000, 0.1)
        assert pnl_pct == -3.0
    
    def test_zero_pnl(self):
        """Test zero P&L percentage."""
        pnl_pct = calculate_pnl_percentage(0, 50000, 0.1)
        assert pnl_pct == 0.0
    
    def test_zero_division_safety(self):
        """Test protection against division by zero."""
        pnl_pct = calculate_pnl_percentage(100, 0, 0)
        assert pnl_pct == 0.0
    
    def test_zero_amount(self):
        """Test with zero amount."""
        pnl_pct = calculate_pnl_percentage(100, 50000, 0)
        assert pnl_pct == 0.0
    
    def test_large_percentage(self):
        """Test large percentage gain."""
        pnl_pct = calculate_pnl_percentage(5000, 50000, 0.1)
        assert pnl_pct == 100.0


class TestReturnPercentage:
    """Test return percentage calculations."""
    
    def test_long_position_return(self):
        """Test return on long position."""
        ret_pct = calculate_return_percentage(50000, 51000, 'long')
        assert ret_pct == 2.0
    
    def test_short_position_return(self):
        """Test return on short position."""
        ret_pct = calculate_return_percentage(50000, 49000, 'short')
        assert ret_pct == 2.0
    
    def test_long_position_loss(self):
        """Test loss on long position."""
        ret_pct = calculate_return_percentage(50000, 49000, 'long')
        assert ret_pct == -2.0
    
    def test_short_position_loss(self):
        """Test loss on short position."""
        ret_pct = calculate_return_percentage(50000, 51000, 'short')
        assert ret_pct == -2.0
    
    def test_zero_entry_price(self):
        """Test protection against zero entry price."""
        ret_pct = calculate_return_percentage(0, 51000, 'long')
        assert ret_pct == 0.0
    
    def test_negative_entry_price(self):
        """Test protection against negative entry price."""
        ret_pct = calculate_return_percentage(-50000, 51000, 'long')
        assert ret_pct == 0.0
    
    def test_no_price_change(self):
        """Test when price doesn't change."""
        ret_pct = calculate_return_percentage(50000, 50000, 'long')
        assert ret_pct == 0.0


class TestPositionValue:
    """Test position value calculations."""
    
    def test_basic_position_value(self):
        """Test basic position value calculation."""
        value = calculate_position_value(50000, 0.1)
        assert value == 5000.0
    
    def test_integer_amount(self):
        """Test with integer amount."""
        value = calculate_position_value(100, 5)
        assert value == 500.0
    
    def test_zero_price(self):
        """Test with zero price."""
        value = calculate_position_value(0, 0.1)
        assert value == 0.0
    
    def test_zero_amount(self):
        """Test with zero amount."""
        value = calculate_position_value(50000, 0)
        assert value == 0.0
    
    def test_large_values(self):
        """Test with large position values."""
        value = calculate_position_value(100000, 10)
        assert value == 1000000.0


class TestIntegrationScenarios:
    """Test integrated P&L calculation scenarios."""
    
    def test_complete_long_trade_flow(self):
        """Test complete flow for a profitable long trade."""
        # Entry
        entry_price = 50000
        amount = 0.1
        position_value = calculate_position_value(entry_price, amount)
        assert position_value == 5000.0
        
        # During trade
        current_price = 51000
        unrealized_pnl = calculate_unrealized_pnl('long', entry_price, current_price, amount)
        assert unrealized_pnl == 100.0
        
        unrealized_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
        assert unrealized_pct == 2.0
        
        # Close trade
        exit_price = 51500
        realized_pnl = calculate_realized_pnl('long', entry_price, exit_price, amount)
        assert realized_pnl == 150.0
        
        return_pct = calculate_return_percentage(entry_price, exit_price, 'long')
        assert return_pct == 3.0
    
    def test_complete_short_trade_flow(self):
        """Test complete flow for a profitable short trade."""
        # Entry
        entry_price = 50000
        amount = 0.1
        position_value = calculate_position_value(entry_price, amount)
        assert position_value == 5000.0
        
        # During trade
        current_price = 49000
        unrealized_pnl = calculate_unrealized_pnl('short', entry_price, current_price, amount)
        assert unrealized_pnl == 100.0
        
        unrealized_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
        assert unrealized_pct == 2.0
        
        # Close trade
        exit_price = 48500
        realized_pnl = calculate_realized_pnl('short', entry_price, exit_price, amount)
        assert realized_pnl == 150.0
        
        return_pct = calculate_return_percentage(entry_price, exit_price, 'short')
        assert return_pct == 3.0
    
    def test_losing_trade_calculations(self):
        """Test calculations for a losing trade."""
        entry_price = 50000
        current_price = 48000
        amount = 0.1
        
        # Calculate loss
        unrealized_pnl = calculate_unrealized_pnl('long', entry_price, current_price, amount)
        assert unrealized_pnl == -200.0
        
        # Calculate loss percentage
        pnl_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
        assert pnl_pct == -4.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
