"""
Test enhanced duplicate prevention with price delta bypass.
Phase 3.4 - Issue #118: Enhanced Duplicate Prevention
"""

import pytest
import time
from unittest.mock import Mock
from src.core.strategy_coordinator import StrategyCoordinator


class TestDuplicatePreventionEnhanced:
    """Test enhanced duplicate prevention with price delta bypass functionality."""
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator with test configuration."""
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 2,  # 2 seconds for testing
                    'price_delta_bypass_threshold': 0.0015,  # 0.15%
                    'price_delta_bypass_enabled': True
                }
            }
        }
        risk_manager = Mock()
        return StrategyCoordinator(portfolio_manager, risk_manager)
    
    def test_cooldown_reject_small_price_change(self, coordinator):
        """Test that small price changes (<0.15%) are rejected during cooldown."""
        # First signal: BTC @ $50,000 with strategy1 → should accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,050 (0.1% change < 0.15%) → should REJECT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50050, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is False, f"Second signal should be rejected, got: is_valid={is_valid}"
        assert 'cooldown' in reason.lower(), f"Reason should mention cooldown, got: {reason}"
        assert ('0.10%' in reason or '0.1%' in reason), f"Reason should show 0.1% price change, got: {reason}"
    
    def test_cooldown_bypass_large_price_change(self, coordinator):
        """Test that large price changes (>0.15%) bypass cooldown."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,100 (0.2% change > 0.15%) → should ACCEPT via bypass
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50100, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Second signal should be accepted via bypass, got: is_valid={is_valid}, reason={reason}"
        assert ('bypass' in reason.lower() or 'OK' in reason), f"Reason should indicate bypass or OK, got: {reason}"
    
    def test_outside_cooldown_always_accept(self, coordinator):
        """Test that signals outside cooldown are always accepted."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Sleep 2.1 seconds (cooldown expires)
        time.sleep(2.1)
        
        # Second signal: BTC @ $50,010 (0.02% change) → should ACCEPT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50010, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Signal after cooldown should be accepted, got: is_valid={is_valid}, reason={reason}"
    
    def test_different_strategy_same_symbol(self, coordinator):
        """Test that different strategies on same symbol are independent."""
        # Signal 1: strategy1 on BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Signal 2: strategy2 on BTC @ $50,010 (immediately) → should ACCEPT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50010, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy2')
        assert is_valid is True, f"Different strategy should be accepted, got: is_valid={is_valid}, reason={reason}"
    
    def test_edge_case_exact_threshold(self, coordinator):
        """Test that exactly 0.15% price change bypasses cooldown."""
        # First signal: BTC @ $50,000 → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal should be accepted, got: {reason}"
        
        # Second signal: BTC @ $50,075 (exactly 0.15% change) → should ACCEPT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50075, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is True, f"Exact threshold (>= 0.15%) should be accepted, got: is_valid={is_valid}, reason={reason}"
    
    def test_realistic_market_scenario(self, coordinator):
        """Test realistic volatile market scenario with multiple price drops."""
        # Simulate BTC drops: [50000, 49800, 49500, 49200, 49000] (each ~0.4% drop)
        prices = [50000, 49800, 49500, 49200, 49000]
        
        for i, price in enumerate(prices):
            signal = {'symbol': 'BTC/USDT:USDT', 'entry': price, 'side': 'long'}
            is_valid, reason = coordinator.validate_duplicate(signal, 'strategy1')
            
            if i == 0:
                # First signal should always be accepted
                assert is_valid is True, f"First signal should be accepted, got: {reason}"
            else:
                # All subsequent signals should be accepted due to large price changes
                assert is_valid is True, f"Signal {i+1} @ ${price} should be accepted via bypass, got: is_valid={is_valid}, reason={reason}"
                assert ('bypass' in reason.lower() or 'OK' in reason), f"Reason should indicate bypass, got: {reason}"
    
    def test_no_price_history_reject(self, coordinator):
        """Test that signals without price history are rejected during cooldown."""
        # First signal: entry=0 (no price) → accept
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 0, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal1, 'strategy1')
        assert is_valid is True, f"First signal (no price) should be accepted, got: {reason}"
        
        # Second signal: entry=50000 (within cooldown, no history to compare) → should REJECT
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(signal2, 'strategy1')
        assert is_valid is False, f"Signal without price history should be rejected, got: is_valid={is_valid}"
        assert 'cooldown' in reason.lower(), f"Reason should mention cooldown, got: {reason}"
    
    def test_multiple_symbols_independent(self, coordinator):
        """Test that multiple symbols are tracked independently."""
        # BTC signal 1 @ $50,000 → accept
        btc_signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(btc_signal1, 'strategy1')
        assert is_valid is True, f"BTC signal 1 should be accepted, got: {reason}"
        
        # ETH signal 1 @ $3,000 → accept (different symbol)
        eth_signal1 = {'symbol': 'ETH/USDT:USDT', 'entry': 3000, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(eth_signal1, 'strategy1')
        assert is_valid is True, f"ETH signal 1 should be accepted, got: {reason}"
        
        # BTC signal 2 @ $50,020 (0.04% change) → reject
        btc_signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50020, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(btc_signal2, 'strategy1')
        assert is_valid is False, f"BTC signal 2 should be rejected (small change), got: is_valid={is_valid}"
        
        # ETH signal 2 @ $3,010 (0.33% change) → accept via bypass
        eth_signal2 = {'symbol': 'ETH/USDT:USDT', 'entry': 3010, 'side': 'long'}
        is_valid, reason = coordinator.validate_duplicate(eth_signal2, 'strategy1')
        assert is_valid is True, f"ETH signal 2 should be accepted via bypass, got: is_valid={is_valid}, reason={reason}"
    
    def test_statistics_tracking(self, coordinator):
        """Test that statistics are properly tracked."""
        # First signal: accepted
        signal1 = {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'}
        coordinator.validate_duplicate(signal1, 'strategy1')
        
        # Second signal: bypass
        signal2 = {'symbol': 'BTC/USDT:USDT', 'entry': 50100, 'side': 'long'}
        coordinator.validate_duplicate(signal2, 'strategy1')
        
        # Third signal: rejected
        signal3 = {'symbol': 'BTC/USDT:USDT', 'entry': 50120, 'side': 'long'}
        coordinator.validate_duplicate(signal3, 'strategy1')
        
        # Get stats
        stats = coordinator.get_duplicate_prevention_stats()
        
        # Verify stats structure
        assert 'cooldown_bypasses' in stats, "Stats should include cooldown_bypasses"
        assert 'rejected_by_cooldown' in stats, "Stats should include rejected_by_cooldown"
        assert 'rejected_by_price_delta' in stats, "Stats should include rejected_by_price_delta"
        assert 'bypass_rate' in stats, "Stats should include bypass_rate"
        assert 'avg_bypass_price_delta' in stats, "Stats should include avg_bypass_price_delta"
        assert 'last_bypass_time' in stats, "Stats should include last_bypass_time"
        
        # Verify bypass was tracked
        assert stats['cooldown_bypasses'] >= 1, f"Should have at least 1 bypass, got: {stats['cooldown_bypasses']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
