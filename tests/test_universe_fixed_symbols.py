#!/usr/bin/env python3
"""
Test fixed symbol list optimization for universe.py.
Verifies that fixed symbol mode skips market loading.
"""
import sys
import os
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from universe import build_universe


class TestUniverseFixedSymbols:
    """Test suite for fixed symbol list optimization."""
    
    def test_fixed_symbols_no_market_loading(self):
        """Test that fixed symbols mode doesn't call markets() or tickers()."""
        # Create mock exchange clients
        mock_client = Mock()
        mock_client.markets = Mock(return_value={})
        mock_client.tickers = Mock(return_value={})
        
        exchanges = {
            'bingx': mock_client,
            'kucoinfutures': mock_client
        }
        
        # Config with fixed symbols and auto_select=false
        fixed_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT'
        ]
        
        config = {
            'universe': {
                'fixed_symbols': fixed_symbols,
                'auto_select': False
            }
        }
        
        # Build universe
        result = build_universe(exchanges, config)
        
        # Verify markets() and tickers() were NOT called
        mock_client.markets.assert_not_called()
        mock_client.tickers.assert_not_called()
        
        # Verify all exchanges got the same symbol list
        assert len(result) == 2
        assert 'bingx' in result
        assert 'kucoinfutures' in result
        assert result['bingx'] == fixed_symbols
        assert result['kucoinfutures'] == fixed_symbols
    
    def test_auto_select_mode_calls_markets(self):
        """Test that auto_select=true still calls markets()."""
        # Create mock exchange clients with proper market data
        mock_markets = {
            'BTC/USDT:USDT': {
                'symbol': 'BTC/USDT:USDT',
                'active': True,
                'quote': 'USDT',
                'type': 'swap',
                'linear': True,
                'swap': True
            }
        }
        
        mock_tickers = {
            'BTC/USDT:USDT': {
                'symbol': 'BTC/USDT:USDT',
                'quoteVolume': 5000000
            }
        }
        
        mock_client = Mock()
        mock_client.markets = Mock(return_value=mock_markets)
        mock_client.tickers = Mock(return_value=mock_tickers)
        
        exchanges = {'bingx': mock_client}
        
        # Config with auto_select=true
        config = {
            'universe': {
                'fixed_symbols': [],
                'auto_select': True,
                'min_quote_volume_usdt': 1000000,
                'top_n_per_exchange': 5
            }
        }
        
        # Build universe
        result = build_universe(exchanges, config)
        
        # Verify markets() and tickers() WERE called
        mock_client.markets.assert_called_once()
        mock_client.tickers.assert_called_once()
    
    def test_empty_fixed_symbols_falls_back_to_auto_select(self):
        """Test that empty fixed_symbols list falls back to auto_select."""
        mock_markets = {
            'BTC/USDT:USDT': {
                'symbol': 'BTC/USDT:USDT',
                'active': True,
                'quote': 'USDT',
                'type': 'swap',
                'linear': True,
                'swap': True
            }
        }
        
        mock_client = Mock()
        mock_client.markets = Mock(return_value=mock_markets)
        mock_client.tickers = Mock(return_value={})
        
        exchanges = {'bingx': mock_client}
        
        # Config with empty fixed_symbols
        config = {
            'universe': {
                'fixed_symbols': [],
                'auto_select': False
            }
        }
        
        # Build universe - should fall back to auto_select
        result = build_universe(exchanges, config)
        
        # Should call markets() because fixed_symbols is empty
        mock_client.markets.assert_called()
    
    def test_fixed_symbols_correct_count(self):
        """Test that fixed symbols returns correct symbol count."""
        mock_client = Mock()
        exchanges = {'bingx': mock_client}
        
        fixed_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'AVAX/USDT:USDT', 'MATIC/USDT:USDT', 'LINK/USDT:USDT',
            'LTC/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT',
            'XRP/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT'
        ]
        
        config = {
            'universe': {
                'fixed_symbols': fixed_symbols,
                'auto_select': False
            }
        }
        
        result = build_universe(exchanges, config)
        
        # Verify correct count
        assert len(result['bingx']) == 15
        assert result['bingx'] == fixed_symbols


class TestCcxtClientFixedSymbols:
    """Test suite for CcxtClient fixed symbol optimization."""
    
    def test_set_required_symbols_enables_skip_mode(self):
        """Test that set_required_symbols enables market skip mode."""
        from core.ccxt_client import CcxtClient
        
        # Create client without credentials (sandbox mode)
        with patch('core.ccxt_client.ccxt') as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binance = MagicMock(return_value=mock_exchange)
            
            client = CcxtClient('binance', None)
            
            # Set required symbols
            symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
            client.set_required_symbols(symbols)
            
            # Verify skip mode is enabled
            assert client._skip_market_load == True
            assert client._required_symbols_only == set(symbols)
    
    def test_markets_skips_loading_in_fixed_mode(self):
        """Test that markets() returns minimal data without loading."""
        from core.ccxt_client import CcxtClient
        
        with patch('core.ccxt_client.ccxt') as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.load_markets = MagicMock(return_value={})
            mock_ccxt.binance = MagicMock(return_value=mock_exchange)
            
            client = CcxtClient('binance', None)
            
            # Enable skip mode
            symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
            client.set_required_symbols(symbols)
            
            # Call markets()
            markets = client.markets()
            
            # Verify load_markets was NOT called
            mock_exchange.load_markets.assert_not_called()
            
            # Verify we got minimal market data
            assert 'BTC/USDT:USDT' in markets
            assert 'ETH/USDT:USDT' in markets
            assert markets['BTC/USDT:USDT']['active'] == True
            assert markets['BTC/USDT:USDT']['quote'] == 'USDT'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
