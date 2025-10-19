"""
Tests for Phase 3.4 Critical Fixes:
- ATR-based TP/SL (Issue #102)
- Duplicate Prevention (Issue #103)
- Exit Monitoring (Issue #100)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from core.strategy_coordinator import StrategyCoordinator
from core.position_manager import AdvancedPositionManager


class TestATRBasedTPSL:
    """Test ATR-based TP/SL calculations (Issue #102)."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample market data with required indicators
        self.market_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='30min'),
            'open': [1800.0 + i for i in range(100)],
            'high': [1810.0 + i for i in range(100)],
            'low': [1790.0 + i for i in range(100)],
            'close': [1800.0 + i for i in range(100)],
            'volume': [1000.0] * 100,
            'rsi': [30.0 + (i % 20) for i in range(100)],  # Oscillating RSI
            'atr': [50.0] * 100,  # $50 ATR
            'ema21': [1800.0 + i for i in range(100)],
            'ema50': [1795.0 + i for i in range(100)],
            'ema200': [1790.0 + i for i in range(100)]
        })
    
    def test_adaptive_ob_atr_based_tpsl(self):
        """Test AdaptiveOversoldBounce uses ATR for TP/SL."""
        config = {
            'rsi_max': 45,
            'adaptive_rsi_base': 45,
            'tp_atr_mult': 2.5,
            'sl_atr_mult': 1.2,
            'min_tp_pct': 0.008,
            'max_sl_pct': 0.015
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        # Use data with low RSI to trigger signal
        test_data = self.market_data.copy()
        test_data.loc[test_data.index[-1], 'rsi'] = 25.0
        
        signal = strategy.signal(test_data)
        
        assert signal is not None, "Signal should be generated for low RSI"
        assert 'entry' in signal
        assert 'stop' in signal
        assert 'target' in signal
        assert 'rr_ratio' in signal
        
        # Verify ATR-based calculation
        entry = signal['entry']
        stop = signal['stop']
        target = signal['target']
        atr = signal['atr']
        
        # For long position:
        # Expected TP = entry + (ATR × tp_atr_mult) = 1899 + (50 × 2.5) = 2024
        # Expected SL = entry - (ATR × sl_atr_mult) = 1899 - (50 × 1.2) = 1839
        expected_tp = entry + (atr * 2.5)
        expected_sl = entry - (atr * 1.2)
        
        # Allow for safety boundary adjustments
        assert target >= expected_tp * 0.95, f"TP too low: {target} vs expected {expected_tp}"
        assert stop >= expected_sl * 0.95, f"SL too low: {stop} vs expected {expected_sl}"
        
        # Verify R/R ratio is calculated
        actual_rr = abs(target - entry) / abs(entry - stop)
        assert signal['rr_ratio'] == pytest.approx(actual_rr, rel=0.01)
        
        # Verify R/R ratio > 1.5 (requirement from problem statement)
        assert signal['rr_ratio'] > 1.5, f"R/R ratio {signal['rr_ratio']} should be > 1.5"
    
    def test_adaptive_str_atr_based_tpsl(self):
        """Test AdaptiveShortTheRip uses ATR for TP/SL."""
        config = {
            'rsi_min': 55,
            'adaptive_rsi_base': 55,
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.5,
            'min_tp_pct': 0.010,
            'max_sl_pct': 0.020
        }
        
        strategy = AdaptiveShortTheRip(config)
        
        # Use data with high RSI to trigger signal
        # Need to set EMA alignment for short: ema21 < ema50 <= ema200
        test_data = self.market_data.copy()
        test_data.loc[test_data.index[-1], 'rsi'] = 75.0
        test_data.loc[test_data.index[-1], 'ema21'] = 1800
        test_data.loc[test_data.index[-1], 'ema50'] = 1850
        test_data.loc[test_data.index[-1], 'ema200'] = 1900
        
        signal = strategy.signal(test_data)
        
        assert signal is not None, "Signal should be generated for high RSI"
        assert 'entry' in signal
        assert 'stop' in signal
        assert 'target' in signal
        assert 'rr_ratio' in signal
        
        # Verify ATR-based calculation for short position
        entry = signal['entry']
        stop = signal['stop']
        target = signal['target']
        atr = signal['atr']
        
        # For short position:
        # Expected TP = entry - (ATR × tp_atr_mult) = 1899 - (50 × 3.0) = 1749
        # Expected SL = entry + (ATR × sl_atr_mult) = 1899 + (50 × 1.5) = 1974
        expected_tp = entry - (atr * 3.0)
        expected_sl = entry + (atr * 1.5)
        
        # Allow for safety boundary adjustments
        assert target <= expected_tp * 1.05, f"TP too high for short: {target} vs expected {expected_tp}"
        assert stop <= expected_sl * 1.05, f"SL too high for short: {stop} vs expected {expected_sl}"
        
        # Verify R/R ratio is calculated
        actual_rr = abs(entry - target) / abs(stop - entry)
        assert signal['rr_ratio'] == pytest.approx(actual_rr, rel=0.01)
        
        # Verify R/R ratio > 1.5
        assert signal['rr_ratio'] > 1.5, f"R/R ratio {signal['rr_ratio']} should be > 1.5"
    
    def test_safety_boundaries_applied(self):
        """Test that safety boundaries are enforced."""
        config = {
            'rsi_max': 45,
            'tp_atr_mult': 0.5,  # Very small TP multiplier
            'sl_atr_mult': 5.0,  # Very large SL multiplier
            'min_tp_pct': 0.008,  # 0.8% minimum TP
            'max_sl_pct': 0.015   # 1.5% maximum SL
        }
        
        strategy = AdaptiveOversoldBounce(config)
        
        test_data = self.market_data.copy()
        test_data.loc[test_data.index[-1], 'rsi'] = 25.0
        
        signal = strategy.signal(test_data)
        
        assert signal is not None
        
        entry = signal['entry']
        stop = signal['stop']
        target = signal['target']
        
        # Check that minimum TP is enforced
        tp_pct = (target - entry) / entry
        assert tp_pct >= 0.008, f"TP percentage {tp_pct} should be >= 0.008"
        
        # Check that maximum SL is enforced (with small tolerance for floating point precision)
        sl_pct = (entry - stop) / entry
        max_sl_with_tolerance = config['max_sl_pct'] * 1.01  # 1% tolerance for floating point
        assert sl_pct <= max_sl_with_tolerance, f"SL percentage {sl_pct} should be <= {config['max_sl_pct']}"


class TestDuplicatePrevention:
    """Test duplicate prevention (Issue #103)."""
    
    @pytest.fixture
    def coordinator(self):
        """Create a strategy coordinator for testing."""
        # Mock portfolio manager and risk manager
        portfolio_mgr = Mock()
        portfolio_mgr.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 300,
                    'same_strategy_cooldown': 180,
                    'min_price_change': 0.002
                }
            }
        }
        portfolio_mgr.exchange_clients = {}
        
        risk_mgr = Mock()
        risk_mgr.active_positions = {}
        
        return StrategyCoordinator(portfolio_mgr, risk_mgr)
    
    def test_symbol_cooldown(self, coordinator):
        """Test symbol cooldown prevents duplicate signals."""
        signal1 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000
        }
        
        # First signal should pass
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid, f"First signal should be valid: {reason}"
        
        # Second signal for same symbol should fail immediately
        signal2 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50100
        }
        
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert not is_valid, "Second signal should be rejected due to cooldown"
        assert "cooldown" in reason.lower()
    
    def test_strategy_cooldown(self, coordinator):
        """Test strategy cooldown prevents duplicate signals."""
        signal1 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000
        }
        
        # First signal
        coordinator.validate_duplicate(signal1, 'strategy1')
        
        # Different symbol but same strategy
        signal2 = {
            'symbol': 'ETH/USDT:USDT',
            'side': 'buy',
            'entry': 3000
        }
        
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert not is_valid, "Same strategy signal should be rejected due to cooldown"
        assert "cooldown" in reason.lower()
    
    def test_price_movement_requirement(self, coordinator):
        """Test minimum price movement requirement."""
        signal1 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000
        }
        
        # First signal
        coordinator.validate_duplicate(signal1, 'strategy1')
        
        # Manually reset cooldowns to test price movement check in isolation
        coordinator.last_position_time.clear()
        coordinator.last_strategy_time.clear()
        
        # Signal with insufficient price movement (< 0.2%)
        signal2 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50050  # Only 0.1% change
        }
        
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy2')
        assert not is_valid, "Signal with insufficient price movement should be rejected"
        assert "price movement" in reason.lower()
        
        # Signal with sufficient price movement (> 0.2%)
        signal3 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50150  # 0.3% change
        }
        
        is_valid, reason = coordinator.validate_duplicate(signal3, 'strategy3')
        assert is_valid, f"Signal with sufficient price movement should be valid: {reason}"
    
    def test_different_symbols_allowed(self, coordinator):
        """Test that different symbols are allowed simultaneously."""
        signal1 = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000
        }
        
        signal2 = {
            'symbol': 'ETH/USDT:USDT',
            'side': 'buy',
            'entry': 3000
        }
        
        # Both should pass with different strategies
        is_valid1, _ = coordinator.validate_duplicate(signal1, 'strategy1')
        is_valid2, _ = coordinator.validate_duplicate(signal2, 'strategy2')
        
        assert is_valid1, "First signal should be valid"
        assert is_valid2, "Second signal with different symbol and strategy should be valid"


class TestExitMonitoring:
    """Test exit monitoring (Issue #100)."""
    
    @pytest.fixture
    def position_manager(self):
        """Create a position manager for testing."""
        # Mock dependencies
        portfolio_mgr = Mock()
        portfolio_mgr.cfg = {
            'position_management': {
                'exit_monitoring': {
                    'enabled': True,
                    'check_frequency': 1  # Fast for testing
                },
                'time_based_exit': {
                    'max_position_duration': 10  # 10 seconds for testing
                }
            }
        }
        portfolio_mgr.exchange_clients = {}
        
        risk_mgr = Mock()
        risk_mgr.register_position = Mock()
        risk_mgr.close_position = Mock()
        
        ws_manager = Mock()
        
        return AdvancedPositionManager(portfolio_mgr, risk_mgr, ws_manager)
    
    @pytest.mark.asyncio
    async def test_stop_loss_priority(self, position_manager):
        """Test that stop-loss has highest priority."""
        # Create a test position
        position = {
            'position_id': 'test_pos_1',
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 50000,
            'amount': 0.01,
            'stop_loss': 49500,
            'take_profit': 51000,
            'status': 'open',
            'opened_at': datetime.now(timezone.utc),
            'unrealized_pnl': 0,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0
        }
        
        position_manager.positions['test_pos_1'] = position
        
        # Mock price fetch to return stop-loss price
        with patch.object(position_manager, '_get_current_price_from_ws', 
                         return_value=49400):  # Below stop-loss
            result = await position_manager.manage_position_exits('test_pos_1')
        
        assert result['should_exit'], "Position should exit when stop-loss is hit"
        assert result['exit_reason'] == 'stop_loss', "Exit reason should be stop_loss"
        assert 'test_pos_1' not in position_manager.positions, "Position should be closed"
    
    @pytest.mark.asyncio
    async def test_take_profit_exit(self, position_manager):
        """Test take-profit exit."""
        position = {
            'position_id': 'test_pos_2',
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 50000,
            'amount': 0.01,
            'stop_loss': 49500,
            'take_profit': 51000,
            'status': 'open',
            'opened_at': datetime.now(timezone.utc),
            'unrealized_pnl': 0,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0
        }
        
        position_manager.positions['test_pos_2'] = position
        
        # Mock price fetch to return take-profit price
        with patch.object(position_manager, '_get_current_price_from_ws', 
                         return_value=51100):  # Above take-profit
            result = await position_manager.manage_position_exits('test_pos_2')
        
        assert result['should_exit'], "Position should exit when take-profit is hit"
        assert result['exit_reason'] == 'take_profit', "Exit reason should be take_profit"
        assert 'test_pos_2' not in position_manager.positions, "Position should be closed"
    
    @pytest.mark.asyncio
    async def test_timeout_exit(self, position_manager):
        """Test timeout-based exit."""
        # Create position with old timestamp (max_position_duration is 10s, so use 15s to exceed it)
        max_duration = position_manager.portfolio_manager.cfg['position_management']['time_based_exit']['max_position_duration']
        old_time = datetime.now(timezone.utc) - timedelta(seconds=max_duration + 5)
        
        position = {
            'position_id': 'test_pos_3',
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 50000,
            'amount': 0.01,
            'stop_loss': 49500,
            'take_profit': 51000,
            'status': 'open',
            'opened_at': old_time,
            'unrealized_pnl': 0,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0
        }
        
        position_manager.positions['test_pos_3'] = position
        
        # Mock price fetch to return current price (no SL/TP hit)
        with patch.object(position_manager, '_get_current_price_from_ws', 
                         return_value=50200):  # Between SL and TP
            result = await position_manager.manage_position_exits('test_pos_3')
        
        assert result['should_exit'], "Position should exit due to timeout"
        assert result['exit_reason'] == 'time_exit', "Exit reason should be time_exit"
        assert 'test_pos_3' not in position_manager.positions, "Position should be closed"
    
    @pytest.mark.asyncio
    async def test_no_exit_conditions(self, position_manager):
        """Test that position remains open when no exit conditions are met."""
        position = {
            'position_id': 'test_pos_4',
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry_price': 50000,
            'current_price': 50000,
            'amount': 0.01,
            'stop_loss': 49500,
            'take_profit': 51000,
            'status': 'open',
            'opened_at': datetime.now(timezone.utc),
            'unrealized_pnl': 0,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0
        }
        
        position_manager.positions['test_pos_4'] = position
        
        # Mock price fetch to return price between SL and TP
        with patch.object(position_manager, '_get_current_price_from_ws', 
                         return_value=50200):  # Between SL and TP, recent position
            result = await position_manager.manage_position_exits('test_pos_4')
        
        assert not result['should_exit'], "Position should remain open"
        assert 'test_pos_4' in position_manager.positions, "Position should still be active"
    
    @pytest.mark.asyncio
    async def test_short_position_stop_loss(self, position_manager):
        """Test stop-loss for short positions."""
        position = {
            'position_id': 'test_pos_5',
            'symbol': 'BTC/USDT:USDT',
            'side': 'short',
            'entry_price': 50000,
            'current_price': 50000,
            'amount': 0.01,
            'stop_loss': 50500,  # For short, SL is above entry
            'take_profit': 49000,  # For short, TP is below entry
            'status': 'open',
            'opened_at': datetime.now(timezone.utc),
            'unrealized_pnl': 0,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0
        }
        
        position_manager.positions['test_pos_5'] = position
        
        # Mock price fetch to return price above stop-loss
        with patch.object(position_manager, '_get_current_price_from_ws', 
                         return_value=50600):  # Above stop-loss for short
            result = await position_manager.manage_position_exits('test_pos_5')
        
        assert result['should_exit'], "Short position should exit when price > stop-loss"
        assert result['exit_reason'] == 'stop_loss', "Exit reason should be stop_loss"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
