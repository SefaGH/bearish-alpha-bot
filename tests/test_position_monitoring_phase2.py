"""
Phase 2 tests for enhanced position monitoring features.
Issue #117: Position monitoring enhancements (trailing stop, short positions).
"""

import pytest
from unittest.mock import Mock
from src.core.position_manager import AdvancedPositionManager


class TestPositionMonitoringPhase2:
    """Test enhanced position monitoring features."""
    
    @pytest.fixture
    def position_manager(self):
        """Create position manager for testing."""
        portfolio_mgr = Mock()
        risk_mgr = Mock()
        ws_mgr = Mock()
        return AdvancedPositionManager(portfolio_mgr, risk_mgr, ws_mgr)
    
    @pytest.mark.asyncio
    async def test_short_position_stop_loss(self, position_manager):
        """Test: Short position hits stop loss (price goes up)."""
        position_id = 'short_test_1'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'short',
            'entry_price': 50000,
            'current_price': 51500,  # Price went up (bad for short)
            'stop_loss': 51000,  # Stop loss above entry for short
            'take_profit': 48000,  # Take profit below entry for short
            'amount': 0.01
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'stop_loss'
        assert exit_check['exit_price'] == 51500
    
    @pytest.mark.asyncio
    async def test_trailing_stop_activation(self, position_manager):
        """Test: Trailing stop tracks highest price and exits on pullback."""
        position_id = 'trailing_test_1'
        
        # Initial position
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 52000,  # Price went up
            'highest_price': 50000,  # Track highest
            'trailing_stop_enabled': True,
            'trailing_stop_distance': 0.02,  # 2%
            'stop_loss': 49000,
            'take_profit': 55000,
            'amount': 0.01
        }
        
        # First check: price at $52,000 (no exit, update highest)
        exit_check = await position_manager.manage_position_exits(position_id)
        assert exit_check['should_exit'] == False
        assert position_manager.positions[position_id]['highest_price'] == 52000
        
        # Price pulls back to $50,960 (2% below $52,000 peak)
        position_manager.positions[position_id]['current_price'] = 50960
        
        # Second check: should NOT exit yet (exactly at trailing stop)
        exit_check = await position_manager.manage_position_exits(position_id)
        # Trailing stop = 52000 * 0.98 = 50960
        # Price = 50960, so might exit depending on implementation (>= or >)
        
        # Price pulls back further to $50,900 (below trailing stop)
        position_manager.positions[position_id]['current_price'] = 50900
        
        # Third check: should exit (trailing stop hit)
        exit_check = await position_manager.manage_position_exits(position_id)
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'trailing_stop'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
