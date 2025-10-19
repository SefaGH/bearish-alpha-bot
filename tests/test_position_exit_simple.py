"""
Simple tests for position exit logic.
Issue #117: Position monitoring and exit implementation.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.core.position_manager import AdvancedPositionManager


class TestPositionExitSimple:
    """Test position exit logic."""
    
    @pytest.fixture
    def position_manager(self):
        """Create position manager."""
        portfolio_mgr = Mock()
        risk_mgr = Mock()
        ws_mgr = Mock()
        return AdvancedPositionManager(portfolio_mgr, risk_mgr, ws_mgr)
    
    @pytest.mark.asyncio
    async def test_stop_loss_hit_triggers_exit(self, position_manager):
        """Test: Stop loss hit → should exit."""
        position_id = 'pos_test_1'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 48500,  # Below stop loss
            'stop_loss': 49000,
            'take_profit': 52000,
            'amount': 0.01
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'stop_loss'
    
    @pytest.mark.asyncio
    async def test_take_profit_hit_triggers_exit(self, position_manager):
        """Test: Take profit hit → should exit."""
        position_id = 'pos_test_2'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 52500,  # Above take profit
            'stop_loss': 49000,
            'take_profit': 52000,
            'amount': 0.01
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'take_profit'
    
    @pytest.mark.asyncio
    async def test_no_exit_when_price_in_range(self, position_manager):
        """Test: Price between SL and TP → no exit."""
        position_id = 'pos_test_3'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 50500,  # Between SL and TP
            'stop_loss': 49000,
            'take_profit': 52000,
            'amount': 0.01
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
