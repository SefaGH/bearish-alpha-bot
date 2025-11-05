"""
Test the new position sizing architecture refactor.
Tests the two-stage position sizing: risk-based + capital percentage caps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, AsyncMock
from core.position_sizing import AdvancedPositionSizing


@pytest.fixture
def mock_risk_manager():
    """Create a mock RiskManager with portfolio summary."""
    risk_manager = Mock()
    risk_manager.config = {
        'per_trade_risk_pct': 0.01,  # 1%
        'risk_usd_cap': 5,
        'max_notional_pct_per_trade': 0.20,  # 20%
        'max_margin_pct_per_trade': 0.20,  # 20%
        'leverage_default': 5
    }
    
    # Mock portfolio summary
    def get_portfolio_summary():
        return {
            'portfolio_value': 100,
            'available_capital': 100,
            'total_exposure': 0
        }
    
    risk_manager.get_portfolio_summary = get_portfolio_summary
    return risk_manager


@pytest.fixture
def position_sizing(mock_risk_manager):
    """Create AdvancedPositionSizing instance."""
    return AdvancedPositionSizing(mock_risk_manager)


class TestPositionSizingRefactor:
    """Test the new two-stage position sizing with capital percentage caps."""
    
    @pytest.mark.asyncio
    async def test_normal_position_sizing(self, position_sizing):
        """Test normal position sizing with 1% stop and $100 capital."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49500,  # 1% stop
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        # With 1% stop and 1% risk, risk-based notional would be $100
        # But should be capped at 20% of capital = $20
        assert 'amount' in sized_signal
        assert 'notional' in sized_signal
        assert 'sizing_meta' in sized_signal
        
        notional = sized_signal['notional']
        assert notional == pytest.approx(20, rel=0.01), f"Expected $20 cap, got ${notional}"
        
        meta = sized_signal['sizing_meta']
        assert meta['capped'] is True, "Position should be capped"
        assert meta['calculations']['risk_based_notional'] == pytest.approx(100, rel=0.01)
        assert meta['calculations']['final_notional'] == pytest.approx(20, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_tight_stop_loss_scenario(self, position_sizing):
        """Test the bug fix: tight stop (0.1%) that would give $1000 position."""
        signal = {
            'symbol': 'ETH/USDT:USDT',
            'entry': 3000,
            'stop': 2997,  # 0.1% stop (tight!)
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        notional = sized_signal['notional']
        # Risk-based would be $1/$0.001 = $1000
        # Should be capped at 20% = $20
        assert notional == pytest.approx(20, rel=0.01), f"Tight stop should be capped at $20, got ${notional}"
        
        meta = sized_signal['sizing_meta']
        assert meta['capped'] is True
        assert meta['calculations']['risk_based_notional'] > 100, "Risk-based should be very high"
        assert meta['position_pct'] == pytest.approx(20, rel=0.1), "Position should be 20% of capital"
    
    @pytest.mark.asyncio
    async def test_futures_with_leverage(self, position_sizing):
        """Test position sizing for futures with 5x leverage."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49500,  # 1% stop
            'leverage': 5
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        notional = sized_signal['notional']
        # With leverage, margin cap = 20% * 5 = 100, but exposure cap is still 20
        # So should be capped at $20
        assert notional == pytest.approx(20, rel=0.01)
        
        meta = sized_signal['sizing_meta']
        assert meta['calculations']['margin_cap'] == pytest.approx(100, rel=0.01)
        assert meta['calculations']['exposure_cap'] == pytest.approx(20, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_large_stop_loss(self, position_sizing):
        """Test position sizing with large stop (5%) - not capped."""
        signal = {
            'symbol': 'SOL/USDT:USDT',
            'entry': 100,
            'stop': 95,  # 5% stop
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        notional = sized_signal['notional']
        # Risk-based: $1 / 0.05 = $20
        # Cap: $20
        # Not capped because they're equal
        assert notional == pytest.approx(20, rel=0.01)
        
        meta = sized_signal['sizing_meta']
        # May or may not be marked as capped due to equality
        assert meta['calculations']['risk_based_notional'] == pytest.approx(20, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_risk_cap_applied(self, position_sizing):
        """Test that risk_usd_cap is applied correctly."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49000,  # 2% stop
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        notional = sized_signal['notional']
        meta = sized_signal['sizing_meta']
        
        # Base risk: $100 * 1% = $1, capped at $5
        # Risk-based notional: $5 / 0.02 = $250
        # Cap: $20
        # Final: $20
        assert notional == pytest.approx(20, rel=0.01)
        assert meta['calculations']['base_risk_usd'] == pytest.approx(1, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_sizing_meta_structure(self, position_sizing):
        """Test that sizing metadata has correct structure."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49500,
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        meta = sized_signal['sizing_meta']
        
        # Check structure
        assert 'method' in meta
        assert 'capital' in meta
        assert 'risk_pct' in meta
        assert 'stop_pct' in meta
        assert 'calculations' in meta
        assert 'position_pct' in meta
        assert 'capped' in meta
        
        # Check calculations sub-structure
        calc = meta['calculations']
        assert 'base_risk_usd' in calc
        assert 'risk_based_notional' in calc
        assert 'exposure_cap' in calc
        assert 'margin_cap' in calc
        assert 'final_notional' in calc
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, position_sizing):
        """Test backward compatibility with position_size field."""
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'entry': 50000,
            'stop': 49500,
            'leverage': 1
        }
        
        sized_signal = await position_sizing.calculate_optimal_size(signal)
        
        # Both amount and position_size should be set
        assert 'amount' in sized_signal
        assert 'position_size' in sized_signal
        assert sized_signal['amount'] == sized_signal['position_size']


class TestCapitalLimitRuleMarginSupport:
    """Test the enhanced CapitalLimitRule with margin support."""
    
    def test_spot_trading_validation(self):
        """Test spot trading (leverage=1) capital validation."""
        from core.risk_rules import CapitalLimitRule
        
        rule = CapitalLimitRule()
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'notional': 20,  # $20 position
            'leverage': 1
        }
        
        # Mock portfolio_manager as dict (test compatibility)
        portfolio_manager = {
            'portfolio_value': 100,
            'total_exposure': 0
        }
        
        # Inject portfolio values into signal (fallback mechanism)
        signal['portfolio_value'] = 100
        signal['current_exposure'] = 0
        
        is_valid, reason = rule.validate(signal, portfolio_manager)
        
        assert is_valid is True, f"Spot position should pass: {reason}"
        assert "spot" in reason.lower()
    
    def test_futures_trading_validation(self):
        """Test futures trading (leverage>1) margin validation."""
        from core.risk_rules import CapitalLimitRule
        
        rule = CapitalLimitRule()
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'notional': 100,  # $100 position
            'leverage': 5  # 5x leverage
        }
        
        # Inject portfolio values
        signal['portfolio_value'] = 100
        signal['current_exposure'] = 0
        
        # Required margin: $100 / 5 = $20
        # Available: $100
        # Should pass
        is_valid, reason = rule.validate(signal, {})
        
        assert is_valid is True, f"Futures position should pass: {reason}"
        assert "margin" in reason.lower() or "leverage" in reason.lower()
    
    def test_futures_insufficient_margin(self):
        """Test futures rejection when insufficient margin."""
        from core.risk_rules import CapitalLimitRule
        
        rule = CapitalLimitRule()
        
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'notional': 600,  # $600 position
            'leverage': 5  # 5x leverage
        }
        
        # Inject portfolio values
        signal['portfolio_value'] = 100
        signal['current_exposure'] = 0
        
        # Required margin: $600 / 5 = $120
        # Available: $100
        # Should fail
        is_valid, reason = rule.validate(signal, {})
        
        assert is_valid is False, "Should reject due to insufficient margin"
        assert "margin" in reason.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
