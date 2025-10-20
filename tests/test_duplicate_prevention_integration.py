"""
Integration test for duplicate prevention optimization (Issue #129).
Simulates paper trading session to verify signal acceptance rate.
"""

import pytest
import time
from unittest.mock import Mock
from src.core.strategy_coordinator import StrategyCoordinator


class TestDuplicatePreventionIntegration:
    """Test signal acceptance rate with optimized duplicate prevention (Issue #129)."""
    
    @pytest.fixture
    def coordinator_optimized(self):
        """Create coordinator with optimized config (0.05%, 20s cooldown)."""
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'signals': {
                'duplicate_prevention': {
                    'enabled': True,
                    'min_price_change_pct': 0.05,  # Optimized threshold
                    'cooldown_seconds': 0.5         # Short cooldown for testing
                }
            }
        }
        risk_manager = Mock()
        return StrategyCoordinator(portfolio_manager, risk_manager)
    
    @pytest.fixture
    def coordinator_aggressive(self):
        """Create coordinator with old aggressive config (0.15%, 30s cooldown)."""
        portfolio_manager = Mock()
        portfolio_manager.cfg = {
            'monitoring': {
                'duplicate_prevention': {
                    'enabled': True,
                    'same_symbol_cooldown': 1,  # Short cooldown for testing
                    'price_delta_bypass_threshold': 0.0015,  # 0.15%
                    'price_delta_bypass_enabled': True
                }
            }
        }
        risk_manager = Mock()
        return StrategyCoordinator(portfolio_manager, risk_manager)
    
    def test_signal_acceptance_rate_optimized(self, coordinator_optimized):
        """
        Test that optimized config achieves >70% signal acceptance rate.
        Simulates 10 signals with realistic price movements in a low-volatility market.
        """
        # Simulate 10 signals with small price movements (0.01-0.10%)
        # Base price: $50,000
        signals = [
            {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'},   # Signal 1: $50,000 → Accept (first)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50030, 'side': 'long'},   # Signal 2: +0.06% → Accept (> 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50055, 'side': 'long'},   # Signal 3: +0.05% → Accept (>= 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50070, 'side': 'long'},   # Signal 4: +0.03% → Reject (< 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50100, 'side': 'long'},   # Signal 5: +0.06% → Accept (> 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50125, 'side': 'long'},   # Signal 6: +0.05% → Accept (>= 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50140, 'side': 'long'},   # Signal 7: +0.03% → Reject (< 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50170, 'side': 'long'},   # Signal 8: +0.06% → Accept (> 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50195, 'side': 'long'},   # Signal 9: +0.05% → Accept (>= 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50210, 'side': 'long'},   # Signal 10: +0.03% → Reject (< 0.05%)
        ]
        
        accepted = 0
        rejected = 0
        
        for i, signal in enumerate(signals):
            is_valid, reason = coordinator_optimized.validate_duplicate(signal, 'strategy1')
            
            if is_valid:
                accepted += 1
                print(f"Signal {i+1}: ACCEPTED - ${signal['entry']} - {reason}")
            else:
                rejected += 1
                print(f"Signal {i+1}: REJECTED - ${signal['entry']} - {reason}")
            
            # Small delay to simulate time passing
            time.sleep(0.1)
        
        acceptance_rate = (accepted / len(signals)) * 100
        
        print(f"\n📊 Results:")
        print(f"   Accepted: {accepted}/{len(signals)} ({acceptance_rate:.0f}%)")
        print(f"   Rejected: {rejected}/{len(signals)}")
        
        # Issue #129 requirement: At least 70% acceptance rate
        assert acceptance_rate >= 70, f"Expected ≥70% acceptance rate, got {acceptance_rate:.0f}%"
    
    def test_signal_acceptance_rate_comparison(self, coordinator_optimized, coordinator_aggressive):
        """
        Compare signal acceptance rates between optimized (0.05%) and aggressive (0.15%) configs.
        Verify that optimized config accepts more signals in low-volatility scenarios.
        """
        # Test with signals having 0.06-0.10% price changes
        # These should be accepted by optimized (0.05%) but rejected by aggressive (0.15%)
        signals = [
            {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'},  # Base
            {'symbol': 'BTC/USDT:USDT', 'entry': 50030, 'side': 'long'},  # +0.06%
            {'symbol': 'BTC/USDT:USDT', 'entry': 50060, 'side': 'long'},  # +0.06%
            {'symbol': 'BTC/USDT:USDT', 'entry': 50090, 'side': 'long'},  # +0.06%
            {'symbol': 'BTC/USDT:USDT', 'entry': 50120, 'side': 'long'},  # +0.06%
        ]
        
        # Test optimized config
        accepted_optimized = 0
        for signal in signals:
            is_valid, _ = coordinator_optimized.validate_duplicate(signal, 'strategy1')
            if is_valid:
                accepted_optimized += 1
            time.sleep(0.1)
        
        # Test aggressive config
        accepted_aggressive = 0
        for signal in signals:
            is_valid, _ = coordinator_aggressive.validate_duplicate(signal, 'strategy1')
            if is_valid:
                accepted_aggressive += 1
            time.sleep(0.1)
        
        rate_optimized = (accepted_optimized / len(signals)) * 100
        rate_aggressive = (accepted_aggressive / len(signals)) * 100
        
        print(f"\n📊 Comparison:")
        print(f"   Optimized (0.05%): {accepted_optimized}/{len(signals)} ({rate_optimized:.0f}%)")
        print(f"   Aggressive (0.15%): {accepted_aggressive}/{len(signals)} ({rate_aggressive:.0f}%)")
        
        # Optimized should accept significantly more signals
        assert accepted_optimized > accepted_aggressive, \
            f"Optimized config should accept more signals than aggressive config"
        
        # Optimized should meet the 70% requirement
        assert rate_optimized >= 70, f"Optimized config should achieve ≥70% acceptance rate"
    
    def test_no_spam_trades_with_optimized_config(self, coordinator_optimized):
        """
        Verify that optimized config doesn't create spam trades.
        Even with lower threshold, duplicate signals should still be blocked.
        """
        # Try to spam same price multiple times - should be rejected after first
        signals = [
            {'symbol': 'BTC/USDT:USDT', 'entry': 50000, 'side': 'long'},  # Accept
            {'symbol': 'BTC/USDT:USDT', 'entry': 50005, 'side': 'long'},  # Reject (0.01% < 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50010, 'side': 'long'},  # Reject (0.01% < 0.05% from last accepted)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50015, 'side': 'long'},  # Reject (0.01% < 0.05%)
            {'symbol': 'BTC/USDT:USDT', 'entry': 50020, 'side': 'long'},  # Reject (0.04% < 0.05%)
        ]
        
        accepted = 0
        rejected = 0
        
        for signal in signals:
            is_valid, _ = coordinator_optimized.validate_duplicate(signal, 'strategy1')
            if is_valid:
                accepted += 1
            else:
                rejected += 1
        
        # Should accept only the first signal (no price movement), reject spam
        assert accepted == 1, f"Should accept only 1 signal, got {accepted}"
        assert rejected == 4, f"Should reject 4 spam signals, got {rejected}"
        
        print(f"✅ No spam trades: {accepted} accepted, {rejected} rejected")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
