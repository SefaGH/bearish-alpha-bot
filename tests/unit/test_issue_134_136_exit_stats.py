import pytest
from unittest.mock import MagicMock, patch
from src.core.position_manager import AdvancedPositionManager, PositionStatus

class TestExitStats:
    @pytest.fixture
    def manager(self):
        risk_manager = MagicMock()
        order_manager = MagicMock()
        manager = AdvancedPositionManager(risk_manager, order_manager)
        return manager

    def test_get_exit_statistics_empty(self, manager):
        """Test that empty stats contain all required keys (Issue #136)."""
        stats = manager.get_exit_statistics()
        
        required_keys = [
            'total_exits', 'exits_by_reason', 'stop_loss_count', 
            'take_profit_count', 'trailing_stop_count', 'manual_close_count',
            'liquidation_count', 'winning_trades', 'losing_trades',
            'win_rate', 'total_pnl', 'total_win_pnl', 'total_loss_pnl',
            'avg_win', 'avg_loss'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
            
        assert stats['total_exits'] == 0
        assert stats['win_rate'] == 0.0

    def test_get_exit_statistics_populated(self, manager):
        """Test stats calculation with closed positions."""
        # Add some closed positions
        manager.closed_positions = [
            {
                'position_id': '1', 'exit_reason': 'stop_loss', 
                'realized_pnl': -10.0, 'return_pct': -1.0
            },
            {
                'position_id': '2', 'exit_reason': 'take_profit', 
                'realized_pnl': 20.0, 'return_pct': 2.0
            },
            {
                'position_id': '3', 'exit_reason': 'manual', 
                'realized_pnl': 5.0, 'return_pct': 0.5
            }
        ]
        
        stats = manager.get_exit_statistics()
        
        assert stats['total_exits'] == 3
        assert stats['stop_loss_count'] == 1
        assert stats['take_profit_count'] == 1
        assert stats['manual_close_count'] == 1
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert stats['total_pnl'] == 15.0
        assert stats['win_rate'] == pytest.approx(66.67, 0.1)

    def test_log_exit_summary_no_crash(self, manager):
        """Test that logging summary doesn't crash (Issue #134/136)."""
        # Mock logger to avoid actual output
        with patch('src.core.position_manager.logger') as mock_logger:
            manager.log_exit_summary()
            
            # Verify some logs were made
            assert mock_logger.info.called

    def test_log_individual_trade_history_no_crash(self, manager):
        """Test that logging individual history doesn't crash."""
        manager.closed_positions = [
            {
                'position_id': '1', 'strategy': 'test', 'side': 'long',
                'entry_price': 100, 'exit_price': 110, 'realized_pnl': 10,
                'return_pct': 10.0, 'exit_reason': 'take_profit',
                'opened_at': None, 'closed_at': None # Missing dates shouldn't crash
            }
        ]
        
        with patch('src.core.position_manager.logger') as mock_logger:
            manager.log_individual_trade_history()
            assert mock_logger.info.called

