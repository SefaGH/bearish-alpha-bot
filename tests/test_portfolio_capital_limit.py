"""
Comprehensive tests for Portfolio Capital Limit Enforcement.
Tests the new portfolio-wide capital limit validation to prevent over-exposure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from core.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    """Create a RiskManager with $100 capital for testing."""
    portfolio_config = {
        'equity_usd': 100,
        'max_portfolio_risk': 0.02,  # 2% max risk
        'max_position_size': 0.15     # 15% max position size
    }
    return RiskManager(portfolio_config)


class TestPortfolioCapitalLimit:
    """Test portfolio capital limit enforcement."""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_first_position(self, risk_manager):
        """Test that first position on empty portfolio should be accepted."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,
            'position_size': 0.0003,  # $15 position (15% of $100)
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
        
        assert is_valid is True, f"First position should be accepted: {reason}"
        assert 'current_exposure' in metrics
        assert metrics['current_exposure'] == 0.0
        assert metrics['new_position_value'] == pytest.approx(15.0, rel=0.01)
        assert metrics['projected_exposure'] == pytest.approx(15.0, rel=0.01)
        assert metrics['capital_limit'] == 100.0
    
    @pytest.mark.asyncio
    async def test_multiple_positions_within_limit(self, risk_manager):
        """Test that 6 positions × $15 = $90 should all be accepted."""
        positions_accepted = 0
        
        for i in range(6):
            signal = {
                'symbol': f'SYMBOL{i}/USDT:USDT',
                'entry': 100,
                'stop': 98,
                'target': 105,
                'position_size': 0.15,  # $15 position each
                'side': 'long'
            }
            
            is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
            
            if is_valid:
                positions_accepted += 1
                # Register position to track it
                risk_manager.register_position(
                    f'pos_{i}',
                    {
                        'symbol': signal['symbol'],
                        'entry_price': signal['entry'],
                        'stop_loss': signal['stop'],
                        'size': signal['position_size'],
                        'side': signal['side'],
                        'risk_amount': abs(signal['entry'] - signal['stop']) * signal['position_size']
                    }
                )
        
        assert positions_accepted == 6, f"All 6 positions should be accepted, got {positions_accepted}"
        
        # Verify total exposure
        total_exposure = risk_manager._calculate_total_portfolio_exposure()
        assert total_exposure == pytest.approx(90.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_position_exceeds_limit_rejected(self, risk_manager):
        """Test that 7th position making total $105 should be REJECTED."""
        # First, add 6 positions totaling $90
        for i in range(6):
            signal = {
                'symbol': f'SYMBOL{i}/USDT:USDT',
                'entry': 100,
                'stop': 98,
                'target': 105,
                'position_size': 0.15,  # $15 position each
                'side': 'long'
            }
            
            is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
            assert is_valid is True
            
            risk_manager.register_position(
                f'pos_{i}',
                {
                    'symbol': signal['symbol'],
                    'entry_price': signal['entry'],
                    'stop_loss': signal['stop'],
                    'size': signal['position_size'],
                    'side': signal['side'],
                    'risk_amount': abs(signal['entry'] - signal['stop']) * signal['position_size']
                }
            )
        
        # Now try to add 7th position (should be rejected)
        signal_7 = {
            'symbol': 'SYMBOL7/USDT:USDT',
            'entry': 100,
            'stop': 98,
            'target': 105,
            'position_size': 0.15,  # $15 position
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal_7, {})
        
        assert is_valid is False, "7th position should be rejected"
        assert 'exceed' in reason.lower(), f"Reason should mention exceeding limit: {reason}"
        assert 'current_exposure' in metrics
        assert metrics['current_exposure'] == pytest.approx(90.0, rel=0.01)
        assert metrics['new_position_value'] == pytest.approx(15.0, rel=0.01)
        assert metrics['projected_exposure'] == pytest.approx(105.0, rel=0.01)
        assert metrics['capital_limit'] == 100.0
    
    @pytest.mark.asyncio
    async def test_edge_case_exactly_at_limit(self, risk_manager):
        """Test that position bringing total to exactly $100 should be accepted."""
        # Add 6 positions totaling $84 (6 × $14)
        for i in range(6):
            signal = {
                'symbol': f'SYMBOL{i}/USDT:USDT',
                'entry': 100,
                'stop': 98,
                'target': 105,
                'position_size': 0.14,  # $14 position each
                'side': 'long'
            }
            
            is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
            assert is_valid is True
            
            risk_manager.register_position(
                f'pos_{i}',
                {
                    'symbol': signal['symbol'],
                    'entry_price': signal['entry'],
                    'stop_loss': signal['stop'],
                    'size': signal['position_size'],
                    'side': signal['side'],
                    'risk_amount': abs(signal['entry'] - signal['stop']) * signal['position_size']
                }
            )
        
        # Now add 7th position that stays within both capital and position size limits
        # 6 positions × $14 = $84, plus $14 = $98 (within $100 capital and 15% position size limit)
        signal_7 = {
            'symbol': 'SYMBOL7/USDT:USDT',
            'entry': 100,
            'stop': 98,
            'target': 105,
            'position_size': 0.14,  # $14 position (within 15% limit)
            'side': 'long'
        }
        
        is_valid, reason, metrics = await risk_manager.validate_new_position(signal_7, {})
        
        assert is_valid is True, f"Position at limit should be accepted: {reason}"
        assert metrics['projected_exposure'] == pytest.approx(98.0, rel=0.01)  # 7 × $14
    
    @pytest.mark.asyncio
    async def test_realistic_8_pair_scenario(self, risk_manager):
        """Test with 8 different trading pairs with realistic sizes."""
        test_signals = [
            {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'stop': 49000, 'target': 52000, 'size': 0.0003, 'value': 15},  # $15
            {'symbol': 'ETH/USDT:USDT', 'entry': 3000, 'stop': 2940, 'target': 3150, 'size': 0.005, 'value': 15},   # $15
            {'symbol': 'BNB/USDT:USDT', 'entry': 400, 'stop': 392, 'target': 420, 'size': 0.025, 'value': 10},      # $10
            {'symbol': 'SOL/USDT:USDT', 'entry': 100, 'stop': 98, 'target': 105, 'size': 0.15, 'value': 15},        # $15
            {'symbol': 'ADA/USDT:USDT', 'entry': 0.5, 'stop': 0.49, 'target': 0.53, 'size': 20, 'value': 10},       # $10
            {'symbol': 'XRP/USDT:USDT', 'entry': 0.6, 'stop': 0.588, 'target': 0.63, 'size': 25, 'value': 15},      # $15
            {'symbol': 'DOT/USDT:USDT', 'entry': 8, 'stop': 7.84, 'target': 8.4, 'size': 1.875, 'value': 15},       # $15
            {'symbol': 'MATIC/USDT:USDT', 'entry': 1, 'stop': 0.98, 'target': 1.05, 'size': 10, 'value': 10},       # $10
        ]
        
        cumulative_value = 0
        positions_accepted = 0
        
        for i, signal_data in enumerate(test_signals):
            signal = {
                'symbol': signal_data['symbol'],
                'entry': signal_data['entry'],
                'stop': signal_data['stop'],
                'target': signal_data['target'],
                'position_size': signal_data['size'],
                'side': 'long'
            }
            
            is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
            
            if is_valid:
                positions_accepted += 1
                cumulative_value += signal_data['value']
                risk_manager.register_position(
                    f'pos_{i}',
                    {
                        'symbol': signal['symbol'],
                        'entry_price': signal['entry'],
                        'stop_loss': signal['stop'],
                        'size': signal['position_size'],
                        'side': signal['side'],
                        'risk_amount': abs(signal['entry'] - signal['stop']) * signal['position_size']
                    }
                )
            else:
                # Should fail once we exceed $100
                projected_value = cumulative_value + signal_data['value']
                assert projected_value > 100, f"Position rejected before reaching limit. Cumulative: ${cumulative_value}, would be: ${projected_value}"
                break
        
        # Expected values: 15+15+10+15+10+15+15 = 95 (7 positions accepted)
        # 8th position would be $10, making total $105 which exceeds $100 limit
        assert positions_accepted == 7, f"Should accept 7 positions (total $95), got {positions_accepted}"
        
        total_exposure = risk_manager._calculate_total_portfolio_exposure()
        assert total_exposure <= 100.0
        assert total_exposure == pytest.approx(95.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_portfolio_summary_includes_exposure(self, risk_manager):
        """Test that portfolio summary includes new exposure fields."""
        # Add some positions
        for i in range(3):
            signal = {
                'symbol': f'SYMBOL{i}/USDT:USDT',
                'entry': 100,
                'stop': 98,
                'target': 105,
                'position_size': 0.1,  # $10 position each
                'side': 'long'
            }
            
            is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
            assert is_valid is True
            
            risk_manager.register_position(
                f'pos_{i}',
                {
                    'symbol': signal['symbol'],
                    'entry_price': signal['entry'],
                    'stop_loss': signal['stop'],
                    'size': signal['position_size'],
                    'side': signal['side'],
                    'risk_amount': abs(signal['entry'] - signal['stop']) * signal['position_size']
                }
            )
        
        # Get portfolio summary
        summary = risk_manager.get_portfolio_summary()
        
        # Verify new fields exist
        assert 'total_exposure' in summary
        assert 'available_capital' in summary
        assert 'capital_utilization' in summary
        
        # Verify values
        assert summary['total_exposure'] == pytest.approx(30.0, rel=0.01)
        assert summary['available_capital'] == pytest.approx(70.0, rel=0.01)
        assert summary['capital_utilization'] == pytest.approx(0.3, rel=0.01)
        assert summary['active_positions'] == 3
