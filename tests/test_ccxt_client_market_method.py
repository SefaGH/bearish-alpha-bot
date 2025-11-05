#!/usr/bin/env python3
"""
DEPRECATED: Test for CcxtClient market() Method

⚠️ ARCHITECTURAL CHANGE NOTICE ⚠️

The market() method and related helpers have been REMOVED from CcxtClient
as part of architectural refactoring to improve separation of concerns.

Market metadata retrieval is now handled by MarketDataPipeline.get_market_metadata()

See: tests/test_market_metadata_pipeline.py for current tests

This test file is kept for historical reference but tests are now SKIPPED.

Original Purpose:
This test verified that the CcxtClient.market() method correctly handled:
1. Symbol normalization (BTC/USDT, BTC/USDT:USDT, BTC-USDT)
2. Cached market data retrieval
3. Fallback to safe default values when no market data is available
4. Compatibility with "no market load" optimization mode
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccxt_client import CcxtClient


@pytest.mark.skip(reason="Methods removed from CcxtClient - see test_market_metadata_pipeline.py")
class TestCcxtClientMarketMethod:
    """DEPRECATED: Test CcxtClient market() method and related helpers."""
    
    @pytest.fixture
    def mock_ccxt(self):
        """Create a mock ccxt exchange."""
        with patch('core.ccxt_client.ccxt') as mock_ccxt:
            # Setup basic exchange mock
            mock_exchange = MagicMock()
            mock_exchange.markets = {}
            mock_ccxt.bingx = Mock(return_value=mock_exchange)
            yield mock_ccxt
    
    @pytest.fixture
    def client_with_markets(self, mock_ccxt):
        """Create a CcxtClient with pre-loaded market data."""
        client = CcxtClient('bingx', creds={'apiKey': 'test', 'secret': 'test'})
        
        # Inject market data
        client.ex.markets = {
            'BTC/USDT:USDT': {
                'id': 'BTC-USDT',
                'symbol': 'BTC/USDT:USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'active': True,
                'type': 'swap',
                'precision': {'amount': 6, 'price': 2},
                'limits': {
                    'amount': {'min': 0.000001, 'max': 1000000},
                    'price': {'min': 0.01, 'max': None},
                    'cost': {'min': 10, 'max': None},
                }
            },
            'ETH/USDT:USDT': {
                'id': 'ETH-USDT',
                'symbol': 'ETH/USDT:USDT',
                'base': 'ETH',
                'quote': 'USDT',
                'active': True,
                'type': 'swap',
                'precision': {'amount': 6, 'price': 2},
                'limits': {
                    'amount': {'min': 0.000001, 'max': 1000000},
                    'price': {'min': 0.01, 'max': None},
                    'cost': {'min': 5, 'max': None},
                }
            }
        }
        
        return client
    
    @pytest.fixture
    def client_no_markets(self, mock_ccxt):
        """Create a CcxtClient with no market data (no market load mode)."""
        client = CcxtClient('bingx', creds={'apiKey': 'test', 'secret': 'test'})
        client.ex.markets = {}
        return client
    
    # Test 1: Symbol Normalization
    def test_normalize_symbol_keys_standard(self, client_with_markets):
        """Test symbol normalization with standard format."""
        variants = client_with_markets._normalize_symbol_keys('BTC/USDT')
        
        assert 'BTC/USDT' in variants
        assert 'BTC/USDT:USDT' in variants
        assert 'BTC-USDT' in variants
        assert 'BTCUSDT' in variants
    
    def test_normalize_symbol_keys_perpetual(self, client_with_markets):
        """Test symbol normalization with perpetual format."""
        variants = client_with_markets._normalize_symbol_keys('ETH/USDT:USDT')
        
        assert 'ETH/USDT' in variants
        assert 'ETH/USDT:USDT' in variants
        assert 'ETH-USDT' in variants
        assert 'ETHUSDT' in variants
    
    def test_normalize_symbol_keys_native(self, client_with_markets):
        """Test symbol normalization with BingX native format."""
        variants = client_with_markets._normalize_symbol_keys('BTC-USDT')
        
        assert 'BTC/USDT' in variants
        assert 'BTC/USDT:USDT' in variants
        assert 'BTC-USDT' in variants
    
    # Test 2: Cached Market Retrieval
    def test_get_cached_market_exact_match(self, client_with_markets):
        """Test retrieving market data with exact symbol match."""
        market = client_with_markets._get_cached_market('BTC/USDT:USDT')
        
        assert market is not None
        assert market['symbol'] == 'BTC/USDT:USDT'
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
    
    def test_get_cached_market_variant_match(self, client_with_markets):
        """Test retrieving market data with symbol variant."""
        # Try to get with standard format when stored as perpetual
        market = client_with_markets._get_cached_market('BTC/USDT')
        
        assert market is not None
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
    
    def test_get_cached_market_not_found(self, client_with_markets):
        """Test retrieving non-existent market returns None."""
        market = client_with_markets._get_cached_market('NONEXISTENT/USDT')
        
        assert market is None
    
    # Test 3: Main market() Method with Cached Data
    def test_market_with_cached_data(self, client_with_markets):
        """Test market() returns cached data when available."""
        market = client_with_markets.market('BTC/USDT:USDT')
        
        assert market is not None
        assert market['symbol'] == 'BTC/USDT:USDT'
        assert market['limits']['cost']['min'] == 10
    
    def test_market_with_variant_symbol(self, client_with_markets):
        """Test market() works with symbol variants."""
        # Request with standard format
        market = client_with_markets.market('ETH/USDT')
        
        assert market is not None
        assert market['base'] == 'ETH'
        assert market['quote'] == 'USDT'
    
    # Test 4: Fallback Mechanism
    def test_market_fallback_no_data(self, client_no_markets):
        """Test market() returns safe fallback when no data available."""
        market = client_no_markets.market('BTC/USDT')
        
        assert market is not None
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
        assert market['active'] is True
        assert market['type'] == 'swap'
        
        # Check fallback values
        assert market['limits']['cost']['min'] == 5  # Default min cost
        assert market['limits']['amount']['min'] == 0.000001
        assert market['precision']['amount'] == 6
        assert market['precision']['price'] == 2
        
        # Check metadata
        assert market['info']['source'] == 'fallback'
        assert 'timestamp' in market['info']
    
    def test_market_fallback_perpetual_format(self, client_no_markets):
        """Test market() fallback handles perpetual format correctly."""
        market = client_no_markets.market('ETH/USDT:USDT')
        
        assert market is not None
        assert market['base'] == 'ETH'
        assert market['quote'] == 'USDT'
        assert market['symbol'] == 'ETH/USDT'
    
    def test_market_fallback_native_format(self, client_no_markets):
        """Test market() fallback handles native format correctly."""
        market = client_no_markets.market('SOL-USDT')
        
        assert market is not None
        assert market['base'] == 'SOL'
        assert market['quote'] == 'USDT'
    
    # Test 5: Compatibility with OrderManager
    def test_market_provides_required_fields_for_order_manager(self, client_with_markets):
        """Test that market() provides all fields needed by SmartOrderManager."""
        market = client_with_markets.market('BTC/USDT')
        
        # Fields that OrderManager might access
        assert 'precision' in market
        assert 'amount' in market['precision']
        assert 'price' in market['precision']
        
        assert 'limits' in market
        assert 'amount' in market['limits']
        assert 'price' in market['limits']
        assert 'cost' in market['limits']
        
        assert 'min' in market['limits']['cost']
    
    def test_market_fallback_provides_required_fields(self, client_no_markets):
        """Test that fallback also provides all required fields."""
        market = client_no_markets.market('BTC/USDT')
        
        # Same checks as above, but with fallback data
        assert 'precision' in market
        assert 'amount' in market['precision']
        assert 'price' in market['precision']
        
        assert 'limits' in market
        assert 'amount' in market['limits']
        assert 'price' in market['limits']
        assert 'cost' in market['limits']
        
        assert 'min' in market['limits']['cost']
        assert market['limits']['cost']['min'] > 0  # Should be positive
    
    # Test 6: Timestamp Helper
    def test_timestamp_returns_milliseconds(self, client_with_markets):
        """Test that timestamp() returns current time in milliseconds."""
        timestamp = client_with_markets.timestamp()
        
        assert isinstance(timestamp, int)
        assert timestamp > 0
        
        # Should be reasonably close to current time (within last minute)
        import time
        current_time_ms = int(time.time() * 1000)
        assert abs(timestamp - current_time_ms) < 60000  # Within 60 seconds
    
    # Test 7: No Market Load Mode
    def test_no_market_load_mode_compatibility(self, client_no_markets):
        """Test that market() works in 'no market load' optimization mode."""
        # Set up "no market load" mode
        client_no_markets._skip_market_load = True
        client_no_markets._required_symbols_only = {'BTC/USDT:USDT', 'ETH/USDT:USDT'}
        
        # Should still work, returning fallback
        market = client_no_markets.market('BTC/USDT')
        
        assert market is not None
        assert market['base'] == 'BTC'
        assert market['info']['source'] == 'fallback'
    
    # Test 8: Multiple Exchange Formats
    def test_market_handles_multiple_exchanges(self, mock_ccxt):
        """Test that different exchanges can use market() method."""
        # Test with different exchange (e.g., kucoinfutures)
        mock_ccxt.kucoinfutures = Mock(return_value=MagicMock())
        
        client = CcxtClient('kucoinfutures', creds={'apiKey': 'test', 'secret': 'test', 'password': 'test'})
        client.ex.markets = {}
        
        # Should work and return fallback
        market = client.market('BTC/USDT:USDT')
        
        assert market is not None
        assert market['base'] == 'BTC'


@pytest.mark.skip(reason="Methods removed from CcxtClient - see test_market_metadata_pipeline.py")
class TestMarketMethodIntegration:
    """DEPRECATED: Integration tests for market() method with SmartOrderManager scenario."""
    
    @pytest.fixture
    def mock_ccxt(self):
        """Create a mock ccxt exchange for integration tests."""
        with patch('core.ccxt_client.ccxt') as mock_ccxt:
            # Setup basic exchange mock
            mock_exchange = MagicMock()
            mock_exchange.markets = {}
            mock_ccxt.bingx = Mock(return_value=mock_exchange)
            yield mock_ccxt
    
    @pytest.fixture
    def client_for_integration(self, mock_ccxt):
        """Create a client similar to production setup."""
        mock_ccxt.bingx = Mock(return_value=MagicMock())
        client = CcxtClient('bingx', creds={'apiKey': 'test', 'secret': 'test'})
        
        # Simulate "no market load" mode (common in production)
        client._skip_market_load = True
        client._required_symbols_only = {'BTC/USDT:USDT', 'ETH/USDT:USDT'}
        client.ex.markets = {}
        
        return client
    
    def test_order_manager_can_call_market_method(self, client_for_integration):
        """
        Test that SmartOrderManager can successfully call client.market().
        
        This is the exact scenario from the bug report.
        """
        symbol = 'BTC/USDT:USDT'
        
        # Simulate what OrderManager does:
        # market_info = self.client.market(symbol)
        market_info = client_for_integration.market(symbol)
        
        # Should not raise AttributeError
        assert market_info is not None
        
        # Should have required fields for order validation
        min_cost = market_info['limits']['cost']['min']
        assert min_cost > 0
        
        # Successful test - min_cost validation works
    
    def test_min_cost_validation(self, client_for_integration):
        """Test that market() provides usable min_cost for validation."""
        market = client_for_integration.market('BTC/USDT')
        
        # Simulate notional value check
        notional_value = 100.0  # $100 order
        min_notional = market['limits']['cost']['min']
        
        # Should be able to validate
        is_valid = notional_value >= min_notional
        
        assert isinstance(is_valid, bool)
        # Validation logic works correctly


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
