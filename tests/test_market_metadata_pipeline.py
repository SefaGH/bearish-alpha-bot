#!/usr/bin/env python3
"""
Test for MarketDataPipeline get_market_metadata() Method

This test verifies that the MarketDataPipeline correctly handles market metadata
retrieval with proper caching and symbol normalization.

This addresses the architectural refactoring where market data responsibility
was moved from CcxtClient to MarketDataPipeline.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.market_data_pipeline import MarketDataPipeline
from core.ccxt_client import CcxtClient


class TestMarketMetadataRetrieval:
    """Test MarketDataPipeline market metadata retrieval."""
    
    @pytest.fixture
    def mock_ccxt_client(self):
        """Create a mock CcxtClient."""
        client = Mock(spec=CcxtClient)
        client.name = 'bingx'
        
        # Mock load_markets to return market data
        client.load_markets = Mock(return_value={
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
        })
        
        return client
    
    @pytest.fixture
    def pipeline(self, mock_ccxt_client):
        """Create a MarketDataPipeline instance."""
        exchanges = {'bingx': mock_ccxt_client}
        return MarketDataPipeline(exchanges=exchanges, config={})
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_basic(self, pipeline, mock_ccxt_client):
        """Test basic market metadata retrieval."""
        symbol = 'BTC/USDT:USDT'
        exchange_id = 'bingx'
        
        market = await pipeline.get_market_metadata(symbol, exchange_id)
        
        assert market is not None
        assert market['symbol'] == 'BTC/USDT:USDT'
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
        assert market['limits']['cost']['min'] == 10
        
        # Verify load_markets was called
        mock_ccxt_client.load_markets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_caching(self, pipeline, mock_ccxt_client):
        """Test that market metadata is cached after first retrieval."""
        symbol = 'ETH/USDT:USDT'
        exchange_id = 'bingx'
        
        # First call
        market1 = await pipeline.get_market_metadata(symbol, exchange_id)
        assert market1 is not None
        
        # Second call should use cache
        market2 = await pipeline.get_market_metadata(symbol, exchange_id)
        assert market2 is not None
        assert market1 == market2
        
        # load_markets should only be called once (for the first request)
        assert mock_ccxt_client.load_markets.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_invalid_exchange(self, pipeline):
        """Test error handling for invalid exchange."""
        with pytest.raises(ValueError, match="Exchange 'invalid' not available"):
            await pipeline.get_market_metadata('BTC/USDT', 'invalid')
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_invalid_symbol(self, pipeline, mock_ccxt_client):
        """Test error handling for invalid symbol."""
        # Mock load_markets to return empty dict for this test
        mock_ccxt_client.load_markets = Mock(return_value={})
        
        with pytest.raises(ValueError, match="Symbol 'INVALID/USDT' not found"):
            await pipeline.get_market_metadata('INVALID/USDT', 'bingx')
    
    @pytest.mark.asyncio
    async def test_normalize_symbol_variants(self, pipeline):
        """Test symbol variant normalization."""
        # Test standard format
        variants = pipeline._normalize_symbol_variants('BTC/USDT')
        assert 'BTC/USDT' in variants
        assert 'BTC/USDT:USDT' in variants
        assert 'BTC-USDT' in variants
        assert 'BTCUSDT' in variants
        
        # Test perpetual format
        variants = pipeline._normalize_symbol_variants('ETH/USDT:USDT')
        assert 'ETH/USDT:USDT' in variants
        assert 'ETH/USDT' in variants
        assert 'ETH-USDT' in variants
        assert 'ETHUSDT' in variants
        
        # Test native format
        variants = pipeline._normalize_symbol_variants('SOL-USDT')
        assert 'SOL-USDT' in variants
        assert 'SOL/USDT' in variants
        assert 'SOL/USDT:USDT' in variants
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_with_variant(self, pipeline, mock_ccxt_client):
        """Test that symbol variants are tried when exact match fails."""
        # Request with standard format, but only perpetual is available in markets
        symbol = 'BTC/USDT'
        exchange_id = 'bingx'
        
        market = await pipeline.get_market_metadata(symbol, exchange_id)
        
        # Should find BTC/USDT:USDT as a variant
        assert market is not None
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
    
    @pytest.mark.asyncio
    async def test_market_metadata_cache_isolation(self, pipeline, mock_ccxt_client):
        """Test that cache properly isolates different exchange/symbol combinations."""
        # Get metadata for two different symbols
        market1 = await pipeline.get_market_metadata('BTC/USDT:USDT', 'bingx')
        market2 = await pipeline.get_market_metadata('ETH/USDT:USDT', 'bingx')
        
        # They should be different
        assert market1 != market2
        assert market1['symbol'] == 'BTC/USDT:USDT'
        assert market2['symbol'] == 'ETH/USDT:USDT'
        
        # Check cache keys are different
        assert 'bingx:BTC/USDT:USDT' in pipeline._market_metadata_cache
        assert 'bingx:ETH/USDT:USDT' in pipeline._market_metadata_cache


class TestMarketMetadataIntegration:
    """Integration tests for market metadata with OrderManager scenario."""
    
    @pytest.fixture
    def mock_ccxt_client_for_integration(self):
        """Create a mock CcxtClient for integration tests."""
        client = Mock(spec=CcxtClient)
        client.name = 'bingx'
        
        # Mock load_markets with realistic data
        client.load_markets = Mock(return_value={
            'BTC/USDT:USDT': {
                'id': 'BTC-USDT',
                'symbol': 'BTC/USDT:USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'active': True,
                'type': 'swap',
                'precision': {'amount': 0.001, 'price': 0.1},
                'limits': {
                    'amount': {'min': 0.001, 'max': 100},
                    'cost': {'min': 10, 'max': None},
                },
            }
        })
        
        return client
    
    @pytest.fixture
    def pipeline_for_integration(self, mock_ccxt_client_for_integration):
        """Create a pipeline for integration tests."""
        exchanges = {'bingx': mock_ccxt_client_for_integration}
        return MarketDataPipeline(exchanges=exchanges, config={})
    
    @pytest.mark.asyncio
    async def test_order_manager_integration_scenario(self, pipeline_for_integration):
        """
        Test the exact scenario OrderManager uses to get market metadata.
        
        This simulates what happens in _limit_order_execution:
        market = await self.market_data_pipeline.get_market_metadata(symbol, exchange)
        """
        symbol = 'BTC/USDT:USDT'
        exchange = 'bingx'
        
        # This is what OrderManager does
        market = await pipeline_for_integration.get_market_metadata(symbol, exchange)
        
        # Should not raise any errors
        assert market is not None
        
        # Should have required fields for order validation
        assert 'precision' in market
        assert 'limits' in market
        assert 'cost' in market['limits']
        assert 'min' in market['limits']['cost']
        
        min_notional = market['limits']['cost']['min']
        assert min_notional > 0
        
        # Simulate notional value check (as in OrderManager)
        amount = 0.01
        limit_price = 50000.0
        notional_value = amount * limit_price  # $500
        
        is_valid = notional_value >= min_notional
        assert is_valid is True  # Should pass validation
    
    @pytest.mark.asyncio
    async def test_min_cost_validation_flow(self, pipeline_for_integration):
        """Test complete flow from metadata retrieval to validation."""
        market = await pipeline_for_integration.get_market_metadata('BTC/USDT:USDT', 'bingx')
        
        # Extract validation data
        min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        price_precision = market.get('precision', {}).get('price', 2)
        
        assert min_cost > 0
        assert min_amount > 0
        assert price_precision > 0
        
        # All required fields for OrderManager are present
        assert isinstance(min_cost, (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
