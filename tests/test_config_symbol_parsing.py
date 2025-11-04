"""
Test configuration symbol parsing fixes.

This test suite validates that trading symbols are correctly parsed from
environment variables, whether they are single symbols or comma-separated lists.

Related Issue: SefaGH/bearish-alpha-bot#278
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.live_trading_config import LiveTradingConfiguration


class TestConfigSymbolParsing:
    """Test that trading symbols are correctly parsed from environment variables."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Clear any existing env vars that might interfere
        self.original_env = {}
        env_vars_to_clear = ['TRADING_SYMBOLS']
        for var in env_vars_to_clear:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]
        
        # Clear the singleton cache before each test
        import config.live_trading_config
        config.live_trading_config._config_instance = None
    
    def teardown_method(self):
        """Restore original environment."""
        # Restore original env vars
        for var, value in self.original_env.items():
            os.environ[var] = value
        
        # Clear test env vars
        if 'TRADING_SYMBOLS' in os.environ:
            del os.environ['TRADING_SYMBOLS']
        
        # Clear the singleton cache after each test
        import config.live_trading_config
        config.live_trading_config._config_instance = None
    
    def test_single_trading_symbol(self):
        """Test that a single trading symbol is correctly parsed as a list."""
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT'
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None, "fixed_symbols should not be None"
        assert isinstance(symbols, list), f"Expected list, got {type(symbols)}"
        assert len(symbols) == 1, f"Expected 1 symbol, got {len(symbols)}"
        assert symbols[0] == 'BTC/USDT', f"Expected 'BTC/USDT', got '{symbols[0]}'"
    
    def test_single_trading_symbol_with_quote(self):
        """Test that a single trading symbol with quote suffix is correctly parsed."""
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT'
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None
        assert isinstance(symbols, list)
        assert len(symbols) == 1
        assert symbols[0] == 'BTC/USDT:USDT'
    
    def test_multiple_trading_symbols(self):
        """Test that multiple comma-separated trading symbols are correctly parsed."""
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT,ETH/USDT,SOL/USDT'
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None
        assert isinstance(symbols, list)
        assert len(symbols) == 3, f"Expected 3 symbols, got {len(symbols)}"
        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
        assert 'SOL/USDT' in symbols
    
    def test_multiple_trading_symbols_with_spaces(self):
        """Test that symbols with extra spaces are correctly trimmed."""
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT, ETH/USDT , SOL/USDT'
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None
        assert isinstance(symbols, list)
        assert len(symbols) == 3
        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
        assert 'SOL/USDT' in symbols
        # Ensure no extra spaces
        assert all(' ' not in s or s.count(' ') == 0 for s in symbols if '/' in s)
    
    def test_multiple_trading_symbols_with_quotes(self):
        """Test that multiple symbols with quote suffixes are correctly parsed."""
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT'
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None
        assert isinstance(symbols, list)
        assert len(symbols) == 3
        assert 'BTC/USDT:USDT' in symbols
        assert 'ETH/USDT:USDT' in symbols
        assert 'SOL/USDT:USDT' in symbols
    
    def test_no_env_uses_yaml_default(self):
        """Test that when no environment variable is set, YAML default is used."""
        # Don't set TRADING_SYMBOLS env var
        
        config = LiveTradingConfiguration.load()
        symbols = config.get('universe', {}).get('fixed_symbols')
        
        assert symbols is not None
        assert isinstance(symbols, list)
        # The default in config.example.yaml should be parsed as a list
        assert len(symbols) > 0


class TestCastValueMethod:
    """Test the _cast_value static method directly."""
    
    def test_cast_single_symbol(self):
        """Test casting a single trading symbol."""
        result = LiveTradingConfiguration._cast_value('BTC/USDT', str)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 'BTC/USDT'
    
    def test_cast_multiple_symbols(self):
        """Test casting multiple comma-separated symbols."""
        result = LiveTradingConfiguration._cast_value('BTC/USDT,ETH/USDT', str)
        assert isinstance(result, list)
        assert len(result) == 2
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result
    
    def test_cast_symbol_with_quote(self):
        """Test casting symbol with quote suffix."""
        result = LiveTradingConfiguration._cast_value('BTC/USDT:USDT', str)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 'BTC/USDT:USDT'
    
    def test_cast_boolean(self):
        """Test casting boolean values."""
        assert LiveTradingConfiguration._cast_value('true', bool) is True
        assert LiveTradingConfiguration._cast_value('false', bool) is False
        assert LiveTradingConfiguration._cast_value('1', bool) is True
        assert LiveTradingConfiguration._cast_value('0', bool) is False
    
    def test_cast_integer(self):
        """Test casting integer values."""
        result = LiveTradingConfiguration._cast_value('42', int)
        assert result == 42
        assert isinstance(result, int)
    
    def test_cast_float(self):
        """Test casting float values."""
        result = LiveTradingConfiguration._cast_value('3.14', float)
        assert result == 3.14
        assert isinstance(result, float)
    
    def test_cast_list_without_slash(self):
        """Test casting regular comma-separated list without trading symbol format."""
        result = LiveTradingConfiguration._cast_value('item1,item2,item3', list)
        assert isinstance(result, list)
        assert len(result) == 3
        assert 'item1' in result
        assert 'item2' in result
        assert 'item3' in result
    
    def test_cast_plain_string(self):
        """Test casting a plain string without special characters."""
        result = LiveTradingConfiguration._cast_value('simple_value', str)
        assert result == 'simple_value'
        assert isinstance(result, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
