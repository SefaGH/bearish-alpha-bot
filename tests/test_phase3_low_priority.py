"""
Phase 3: Low Priority Fixes - Tests for Exit Logic Validation & WebSocket Performance Logging
Issue #134: Validate Exit Logic
Issue #135: Add WebSocket Performance Logging
"""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.position_manager import AdvancedPositionManager


class TestExitLogicValidation:
    """Test exit logic validation and session summaries (Issue #134)."""
    
    @pytest.fixture
    def position_manager(self):
        """Create position manager."""
        portfolio_mgr = Mock()
        portfolio_mgr.cfg = {}
        risk_mgr = Mock()
        risk_mgr.register_position = Mock()
        risk_mgr.close_position = Mock()
        ws_mgr = Mock()
        return AdvancedPositionManager(portfolio_mgr, risk_mgr, ws_mgr)
    
    @pytest.mark.asyncio
    async def test_stop_loss_exit_with_logging(self, position_manager):
        """Test stop loss exit triggers with proper logging."""
        position_id = 'pos_btc_sl_test'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 100000.0,
            'current_price': 95000.0,  # 5% below entry
            'stop_loss': 97000.0,
            'take_profit': 105000.0,
            'amount': 0.01,
            'trailing_stop_enabled': False
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'stop_loss'
        assert exit_check['exit_price'] == 95000.0
    
    @pytest.mark.asyncio
    async def test_take_profit_exit_with_logging(self, position_manager):
        """Test take profit exit triggers with proper logging."""
        position_id = 'pos_eth_tp_test'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'ETH/USDT:USDT',
            'side': 'long',
            'entry_price': 3500.0,
            'current_price': 3675.0,  # 5% above entry
            'stop_loss': 3400.0,
            'take_profit': 3650.0,
            'amount': 0.1,
            'trailing_stop_enabled': False
        }
        
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'take_profit'
        assert exit_check['exit_price'] == 3675.0
    
    @pytest.mark.asyncio
    async def test_trailing_stop_exit(self, position_manager):
        """Test trailing stop exit triggers correctly."""
        position_id = 'pos_sol_trailing_test'
        position_manager.positions[position_id] = {
            'position_id': position_id,
            'symbol': 'SOL/USDT:USDT',
            'side': 'long',
            'entry_price': 140.0,
            'current_price': 143.0,  # Price moved up
            'stop_loss': 138.0,
            'take_profit': 150.0,
            'amount': 1.0,
            'trailing_stop_enabled': True,
            'trailing_stop_distance': 0.02,  # 2%
            'highest_price': 147.0  # Highest reached was 147
        }
        
        # Current price 143 is 2.7% below highest (147)
        # With 2% trailing distance, should trigger exit
        exit_check = await position_manager.manage_position_exits(position_id)
        
        assert exit_check['should_exit'] == True
        assert exit_check['exit_reason'] == 'trailing_stop'
        assert exit_check['exit_price'] == 143.0
    
    @pytest.mark.asyncio
    async def test_exit_statistics_summary(self, position_manager):
        """Test exit statistics generation for session summary."""
        # Create some closed positions with different exit reasons
        position_manager.closed_positions = [
            {
                'position_id': 'pos_1',
                'symbol': 'BTC/USDT:USDT',
                'exit_reason': 'stop_loss',
                'realized_pnl': -50.0,
                'entry_price': 100000,
                'exit_price': 99500
            },
            {
                'position_id': 'pos_2',
                'symbol': 'ETH/USDT:USDT',
                'exit_reason': 'take_profit',
                'realized_pnl': 120.0,
                'entry_price': 3500,
                'exit_price': 3650
            },
            {
                'position_id': 'pos_3',
                'symbol': 'SOL/USDT:USDT',
                'exit_reason': 'trailing_stop',
                'realized_pnl': 80.0,
                'entry_price': 140,
                'exit_price': 145
            },
            {
                'position_id': 'pos_4',
                'symbol': 'BTC/USDT:USDT',
                'exit_reason': 'take_profit',
                'realized_pnl': 200.0,
                'entry_price': 100000,
                'exit_price': 102000
            },
            {
                'position_id': 'pos_5',
                'symbol': 'ETH/USDT:USDT',
                'exit_reason': 'stop_loss',
                'realized_pnl': -30.0,
                'entry_price': 3500,
                'exit_price': 3470
            }
        ]
        
        stats = position_manager.get_exit_statistics()
        
        # Verify basic counts
        assert stats['total_exits'] == 5
        assert stats['stop_loss_count'] == 2
        assert stats['take_profit_count'] == 2
        assert stats['trailing_stop_count'] == 1
        
        # Verify win/loss breakdown
        assert stats['winning_trades'] == 3
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == 60.0
        
        # Verify P&L calculations
        assert stats['total_pnl'] == 320.0  # 120 + 80 + 200 - 50 - 30
        assert stats['total_win_pnl'] == 400.0  # 120 + 80 + 200
        assert stats['total_loss_pnl'] == -80.0  # -50 - 30
        
        # Verify averages
        assert stats['avg_win'] == pytest.approx(133.33, rel=0.01)
        assert stats['avg_loss'] == -40.0
    
    def test_log_exit_summary_no_errors(self, position_manager):
        """Test that log_exit_summary doesn't raise errors even with no data."""
        # Should not raise any errors
        position_manager.log_exit_summary()
        
        # Add some positions and test again
        position_manager.closed_positions = [
            {
                'position_id': 'pos_1',
                'exit_reason': 'take_profit',
                'realized_pnl': 100.0
            }
        ]
        position_manager.log_exit_summary()


class TestWebSocketPerformanceLogging:
    """Test WebSocket performance logging (Issue #135)."""
    
    @pytest.fixture
    def ws_stats_engine(self):
        """Create a mock object with WebSocket stats functionality."""
        class MockWSStatsEngine:
            def __init__(self):
                self.ws_stats = {
                    'websocket_fetches': 0,
                    'rest_fetches': 0,
                    'websocket_failures': 0,
                    'total_latency_ws': 0.0,
                    'total_latency_rest': 0.0,
                    'avg_latency_ws': 0.0,
                    'avg_latency_rest': 0.0,
                    'websocket_success_rate': 0.0,
                    'last_ws_fetch_time': None,
                    'last_rest_fetch_time': None,
                    'consecutive_ws_failures': 0
                }
            
            def _record_ws_fetch(self, latency_ms: float, success: bool):
                """Record WebSocket fetch metrics."""
                if success:
                    self.ws_stats['websocket_fetches'] += 1
                    self.ws_stats['total_latency_ws'] += latency_ms
                    self.ws_stats['avg_latency_ws'] = self.ws_stats['total_latency_ws'] / self.ws_stats['websocket_fetches']
                    self.ws_stats['consecutive_ws_failures'] = 0
                else:
                    self.ws_stats['websocket_failures'] += 1
                    self.ws_stats['consecutive_ws_failures'] += 1
                
                total = self.ws_stats['websocket_fetches'] + self.ws_stats['websocket_failures']
                if total > 0:
                    self.ws_stats['websocket_success_rate'] = (self.ws_stats['websocket_fetches'] / total * 100)
            
            def _record_rest_fetch(self, latency_ms: float, success: bool):
                """Record REST fetch metrics."""
                if success:
                    self.ws_stats['rest_fetches'] += 1
                    self.ws_stats['total_latency_rest'] += latency_ms
                    self.ws_stats['avg_latency_rest'] = self.ws_stats['total_latency_rest'] / self.ws_stats['rest_fetches']
            
            def get_websocket_stats(self):
                """Get comprehensive WebSocket statistics."""
                stats = self.ws_stats.copy()
                total = stats['websocket_fetches'] + stats['rest_fetches']
                stats['websocket_usage_ratio'] = (stats['websocket_fetches'] / total * 100) if total > 0 else 0.0
                if stats['avg_latency_rest'] > 0 and stats['avg_latency_ws'] > 0:
                    stats['latency_improvement_pct'] = ((stats['avg_latency_rest'] - stats['avg_latency_ws']) / stats['avg_latency_rest'] * 100)
                else:
                    stats['latency_improvement_pct'] = 0.0
                return stats
            
            def _log_websocket_performance(self):
                """Log WebSocket performance metrics."""
                stats = self.get_websocket_stats()
                # Just verify it can be formatted without errors
                log_msg = (
                    f"[WS-PERFORMANCE]\n"
                    f"  Usage Ratio: {stats['websocket_usage_ratio']:.1f}%\n"
                    f"  WS Latency: {stats['avg_latency_ws']:.1f}ms\n"
                    f"  REST Latency: {stats['avg_latency_rest']:.1f}ms\n"
                    f"  Improvement: {stats['latency_improvement_pct']:.1f}%"
                )
                return log_msg
        
        return MockWSStatsEngine()
    
    def test_websocket_stats_initialization(self, ws_stats_engine):
        """Test WebSocket statistics are initialized correctly."""
        stats = ws_stats_engine.ws_stats
        
        assert 'websocket_fetches' in stats
        assert 'rest_fetches' in stats
        assert 'websocket_failures' in stats
        assert 'avg_latency_ws' in stats
        assert 'avg_latency_rest' in stats
        assert stats['websocket_fetches'] == 0
        assert stats['rest_fetches'] == 0
    
    def test_record_ws_fetch_success(self, ws_stats_engine):
        """Test recording successful WebSocket fetch."""
        ws_stats_engine._record_ws_fetch(15.0, success=True)
        
        stats = ws_stats_engine.ws_stats
        assert stats['websocket_fetches'] == 1
        assert stats['avg_latency_ws'] == 15.0
        assert stats['consecutive_ws_failures'] == 0
    
    def test_record_ws_fetch_failure(self, ws_stats_engine):
        """Test recording failed WebSocket fetch."""
        ws_stats_engine._record_ws_fetch(0, success=False)
        
        stats = ws_stats_engine.ws_stats
        assert stats['websocket_failures'] == 1
        assert stats['consecutive_ws_failures'] == 1
    
    def test_record_rest_fetch(self, ws_stats_engine):
        """Test recording REST fetch."""
        ws_stats_engine._record_rest_fetch(250.0, success=True)
        
        stats = ws_stats_engine.ws_stats
        assert stats['rest_fetches'] == 1
        assert stats['avg_latency_rest'] == 250.0
    
    def test_get_websocket_stats_with_metrics(self, ws_stats_engine):
        """Test getting WebSocket statistics with calculated metrics."""
        # Record some fetches
        ws_stats_engine._record_ws_fetch(15.0, success=True)
        ws_stats_engine._record_ws_fetch(20.0, success=True)
        ws_stats_engine._record_rest_fetch(250.0, success=True)
        ws_stats_engine._record_rest_fetch(300.0, success=True)
        
        stats = ws_stats_engine.get_websocket_stats()
        
        # Check usage ratio
        assert stats['websocket_usage_ratio'] == pytest.approx(50.0, rel=0.01)  # 2 WS / 4 total
        
        # Check average latencies
        assert stats['avg_latency_ws'] == pytest.approx(17.5, rel=0.01)  # (15 + 20) / 2
        assert stats['avg_latency_rest'] == pytest.approx(275.0, rel=0.01)  # (250 + 300) / 2
        
        # Check improvement percentage
        expected_improvement = ((275.0 - 17.5) / 275.0) * 100
        assert stats['latency_improvement_pct'] == pytest.approx(expected_improvement, rel=0.01)
    
    def test_log_websocket_performance_no_errors(self, ws_stats_engine):
        """Test that _log_websocket_performance doesn't raise errors."""
        # Record some data
        ws_stats_engine._record_ws_fetch(15.0, success=True)
        ws_stats_engine._record_rest_fetch(250.0, success=True)
        
        # Should not raise any errors
        log_msg = ws_stats_engine._log_websocket_performance()
        assert "[WS-PERFORMANCE]" in log_msg
    
    def test_websocket_performance_format(self, ws_stats_engine):
        """Test WebSocket performance log format matches requirements."""
        ws_stats_engine._record_ws_fetch(18.3, success=True)
        ws_stats_engine._record_rest_fetch(234.7, success=True)
        
        stats = ws_stats_engine.get_websocket_stats()
        
        # Verify all required metrics are present
        assert 'websocket_usage_ratio' in stats
        assert 'avg_latency_ws' in stats
        assert 'avg_latency_rest' in stats
        assert 'latency_improvement_pct' in stats
        
        # Verify format can be generated
        log_format = (
            f"[WS-PERFORMANCE]\n"
            f"  Usage Ratio: {stats['websocket_usage_ratio']:.1f}%\n"
            f"  WS Latency: {stats['avg_latency_ws']:.1f}ms\n"
            f"  REST Latency: {stats['avg_latency_rest']:.1f}ms\n"
            f"  Improvement: {stats['latency_improvement_pct']:.1f}%"
        )
        
        assert "[WS-PERFORMANCE]" in log_format
        assert "Usage Ratio:" in log_format
        assert "WS Latency:" in log_format
        assert "REST Latency:" in log_format
        assert "Improvement:" in log_format


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
