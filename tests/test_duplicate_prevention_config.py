"""
Test duplicate prevention config optimization.
Issue #129: Test new signals.duplicate_prevention config format
"""

import pytest
from unittest.mock import Mock
from src.core.strategy_coordinator import StrategyCoordinator


class TestDuplicatePreventionConfig:
    """Test new signals.duplicate_prevention config format (Issue #129)."""
    
    @pytest.fixture
    def coordinator_new_config(self):
        """Create coordinator with new signals.duplicate_prevention config."""
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'signals': {
                'duplicate_prevention': {
                    'enabled': True,
                    'min_price_change_pct': 0.05,  # 0.05% threshold
                    'cooldown_seconds': 20          # 20 second cooldown
                }
            }
        }
        risk_manager = Mock()
        return StrategyCoordinator(portfolio_manager, risk_manager)
    
    @pytest.fixture
    def coordinator_old_config(self):
        """Create coordinator with old monitoring.duplicate_prevention config (fallback)."""
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 60,
                    'price_delta_bypass_threshold': 0.0015,  # 0.15%
                    'price_delta_bypass_enabled': True
                }
            }
        }
        risk_manager = Mock()
        return StrategyCoordinator(portfolio_manager, risk_manager)
    
    def test_new_config_reads_threshold(self, coordinator_new_config):
        """Test that new config format reads 0.05% threshold correctly."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,030 (0.06% change > 0.05%) → should ACCEPT via bypass
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50030, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Signal with 0.06% change should be accepted (> 0.05% threshold), got: {reason}"
        assert ('bypass' in reason.lower() or 'OK' in reason), f"Reason should indicate bypass or OK, got: {reason}"
    
    def test_new_config_rejects_below_threshold(self, coordinator_new_config):
        """Test that new config rejects signals below 0.05% threshold."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,020 (0.04% change < 0.05%) → should REJECT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50020, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal2, 'strategy1')
        assert is_valid is False, f"Signal with 0.04% change should be rejected (< 0.05% threshold), got: {reason}"
        assert 'cooldown' in reason.lower(), f"Reason should mention cooldown, got: {reason}"
    
    def test_old_config_fallback_still_works(self, coordinator_old_config):
        """Test that old monitoring.duplicate_prevention config still works as fallback."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator_old_config.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,100 (0.2% change > 0.15%) → should ACCEPT via bypass
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50100, 'side': 'long'}
        is_valid, reason = coordinator_old_config.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Signal with 0.2% change should be accepted (> 0.15% threshold), got: {reason}"
    
    def test_new_config_more_lenient_than_old(self, coordinator_new_config, coordinator_old_config):
        """Test that new config (0.05%) is more lenient than old config (0.15%)."""
        # Test with 0.1% price change - should be accepted by new, rejected by old
        
        # New config: 0.1% change should be accepted
        signal1_new = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        coordinator_new_config.validate_duplicate(signal1_new, 'strategy1')
        signal2_new = {'symbol': 'BTC/USDT:USDT', 'entry': 50050, 'side': 'long'}  # 0.1% change
        is_valid_new, reason_new = coordinator_new_config.validate_duplicate(signal2_new, 'strategy1')
        
        # Old config: 0.1% change should be rejected
        signal1_old = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        coordinator_old_config.validate_duplicate(signal1_old, 'strategy1')
        signal2_old = {'symbol': 'BTC/USDT:USDT', 'entry': 50050, 'side': 'long'}  # 0.1% change
        is_valid_old, reason_old = coordinator_old_config.validate_duplicate(signal2_old, 'strategy1')
        
        assert is_valid_new is True, f"New config (0.05%) should accept 0.1% change, got: {reason_new}"
        assert is_valid_old is False, f"Old config (0.15%) should reject 0.1% change, got: {reason_old}"
    
    def test_edge_case_exact_new_threshold(self, coordinator_new_config):
        """Test that exactly 0.05% price change bypasses cooldown with new config."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,025 (exactly 0.05% change) → should ACCEPT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50025, 'side': 'long'}
        is_valid, reason = coordinator_new_config.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Exact threshold (>= 0.05%) should be accepted, got: is_valid={is_valid}, reason={reason}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
