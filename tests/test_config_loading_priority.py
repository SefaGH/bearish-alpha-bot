"""
Test Configuration Loading Priority (ENV > YAML > Defaults).

This test suite validates that all modules use consistent configuration
with the correct priority order to prevent state mismatches that can
cause bot freezes.
"""

import os
import sys
import pytest
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from config.live_trading_config import LiveTradingConfiguration
from core.live_trading_engine import LiveTradingEngine


class TestConfigLoadingPriority:
    """Test configuration loading priority across all modules."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Clear any existing env vars that might interfere
        self.original_env = {}
        env_vars_to_clear = [
            'TRADING_SYMBOLS',
            'CONFIG_PATH',
            'CAPITAL_USDT',
            'DUPLICATE_PREVENTION_THRESHOLD',
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]
    
    def teardown_method(self):
        """Restore original environment."""
        # Restore original env vars
        for var, value in self.original_env.items():
            os.environ[var] = value
        
        # Clear test env vars
        test_vars = ['TRADING_SYMBOLS', 'CONFIG_PATH', 'CAPITAL_USDT']
        for var in test_vars:
            if var in os.environ and var not in self.original_env:
                del os.environ[var]
    
    def test_env_overrides_yaml(self):
        """Test that ENV variables override YAML config."""
        # Set ENV variable with single symbol
        os.environ['TRADING_SYMBOLS'] = 'XRP/USDT:USDT'
        
        # Load config (will also load YAML which has BTC/ETH/SOL)
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # ENV should win
        assert 'universe' in config
        assert config['universe']['fixed_symbols'] == ['XRP/USDT:USDT']
    
    def test_yaml_used_when_no_env(self):
        """Test that YAML config is used when no ENV variable is set."""
        # Make sure no ENV variable is set
        if 'TRADING_SYMBOLS' in os.environ:
            del os.environ['TRADING_SYMBOLS']
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Should use YAML defaults (BTC/ETH/SOL)
        assert 'universe' in config
        fixed_symbols = config['universe']['fixed_symbols']
        assert isinstance(fixed_symbols, list)
        assert len(fixed_symbols) > 0
    
    def test_defaults_when_no_yaml_and_no_env(self):
        """Test that defaults are used when no YAML and no ENV."""
        # Create empty temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({}, f)
            temp_config_path = f.name
        
        try:
            # Set CONFIG_PATH to empty file and no ENV
            os.environ['CONFIG_PATH'] = temp_config_path
            if 'TRADING_SYMBOLS' in os.environ:
                del os.environ['TRADING_SYMBOLS']
            
            # Load config
            config = LiveTradingConfiguration.load(
                config_path=temp_config_path,
                log_summary=False
            )
            
            # Should use ENV defaults (which are the defaults from load_from_env)
            assert 'universe' in config
            fixed_symbols = config['universe']['fixed_symbols']
            assert fixed_symbols == ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        finally:
            os.unlink(temp_config_path)
            if 'CONFIG_PATH' in os.environ:
                del os.environ['CONFIG_PATH']
    
    def test_multiple_env_overrides(self):
        """Test that multiple ENV variables all override YAML."""
        # Set multiple ENV variables
        os.environ['TRADING_SYMBOLS'] = 'AVAX/USDT:USDT,MATIC/USDT:USDT'
        os.environ['CAPITAL_USDT'] = '500'
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Check symbols
        assert config['universe']['fixed_symbols'] == ['AVAX/USDT:USDT', 'MATIC/USDT:USDT']
        
        # Check capital
        assert config['risk']['equity_usd'] == 500.0
    
    def test_live_trading_engine_uses_unified_config(self):
        """Test that LiveTradingEngine uses unified config loader."""
        # Set ENV variable
        os.environ['TRADING_SYMBOLS'] = 'DOGE/USDT:USDT'
        
        # Create mock dependencies
        portfolio_manager = MagicMock()
        risk_manager = MagicMock()
        websocket_manager = MagicMock()
        exchange_clients = {'bingx': MagicMock()}
        
        # Initialize engine
        engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=websocket_manager,
            exchange_clients=exchange_clients
        )
        
        # Check that engine config matches unified config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Both should have the same symbols (from ENV)
        assert engine.config['universe']['fixed_symbols'] == ['DOGE/USDT:USDT']
        assert config['universe']['fixed_symbols'] == ['DOGE/USDT:USDT']
    
    def test_live_trading_launcher_uses_unified_config(self):
        """Test that LiveTradingLauncher uses unified config loader.
        
        This test verifies the _load_config method without needing to
        instantiate the full launcher (which has heavy dependencies).
        """
        # Set ENV variable
        os.environ['TRADING_SYMBOLS'] = 'LTC/USDT:USDT'
        
        # Verify the unified config works as expected
        unified_config = LiveTradingConfiguration.load(log_summary=False)
        
        # Both should have the same symbols (from ENV)
        assert unified_config['universe']['fixed_symbols'] == ['LTC/USDT:USDT']
        
        # Verify by reading the launcher source code that it uses the unified loader
        # The launcher's _load_config() method is:
        # def _load_config(self) -> Dict[str, Any]:
        #     if self.config is None:
        #         from config.live_trading_config import LiveTradingConfiguration
        #         self.config = LiveTradingConfiguration.load(log_summary=False)
        #     return self.config
        #
        # This confirms it uses the same unified loader we're testing
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'live_trading_launcher.py')
        assert os.path.exists(scripts_path), "Launcher file should exist"
        
        with open(scripts_path, 'r') as f:
            content = f.read()
            # Verify it imports and uses LiveTradingConfiguration
            assert 'from config.live_trading_config import LiveTradingConfiguration' in content
            assert 'LiveTradingConfiguration.load' in content
    
    def test_consistency_across_modules(self):
        """Test that all modules see the same config."""
        # Set ENV variable
        os.environ['TRADING_SYMBOLS'] = 'BNB/USDT:USDT,DOT/USDT:USDT'
        
        # Load config through unified loader
        config1 = LiveTradingConfiguration.load(log_summary=False)
        
        # Load config through another unified loader call
        config2 = LiveTradingConfiguration.load(log_summary=False)
        
        # Create engine
        portfolio_manager = MagicMock()
        risk_manager = MagicMock()
        websocket_manager = MagicMock()
        exchange_clients = {'bingx': MagicMock()}
        
        engine = LiveTradingEngine(
            mode='paper',
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            websocket_manager=websocket_manager,
            exchange_clients=exchange_clients
        )
        
        # All should have the same symbols
        expected_symbols = ['BNB/USDT:USDT', 'DOT/USDT:USDT']
        assert config1['universe']['fixed_symbols'] == expected_symbols
        assert config2['universe']['fixed_symbols'] == expected_symbols
        assert engine.config['universe']['fixed_symbols'] == expected_symbols
    
    def test_empty_env_variable_falls_back_to_yaml(self):
        """Test that empty ENV variable falls back to YAML."""
        # Set empty ENV variable
        os.environ['TRADING_SYMBOLS'] = ''
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Should fall back to YAML defaults (since empty env means no value)
        # The get_env_list function returns default when value is empty
        assert 'universe' in config
        fixed_symbols = config['universe']['fixed_symbols']
        assert len(fixed_symbols) > 0  # Should have default symbols
    
    def test_malformed_env_variable_falls_back_gracefully(self):
        """Test that malformed ENV variables fall back gracefully."""
        # Set malformed ENV variable with invalid symbols
        os.environ['TRADING_SYMBOLS'] = 'INVALID,,,,BROKEN'
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Should fall back to defaults since all symbols are invalid
        assert 'universe' in config
        fixed_symbols = config['universe']['fixed_symbols']
        assert len(fixed_symbols) > 0  # Should have defaults
        # Should have default symbols
        assert fixed_symbols == ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    def test_partial_env_override(self):
        """Test that partial ENV override works correctly."""
        # Set only one ENV variable
        os.environ['CAPITAL_USDT'] = '1000'
        # Do not set TRADING_SYMBOLS
        if 'TRADING_SYMBOLS' in os.environ:
            del os.environ['TRADING_SYMBOLS']
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Capital should be from ENV
        assert config['risk']['equity_usd'] == 1000.0
        
        # Symbols should be from YAML (since not in ENV)
        assert 'universe' in config
        assert len(config['universe']['fixed_symbols']) > 0
    
    def test_partially_valid_env_filters_invalid_symbols(self):
        """Test that partially valid ENV filters invalid symbols."""
        # Set ENV with mix of valid and invalid symbols
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,INVALID,ETH/USDT:USDT'
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Should filter invalid, keep valid
        fixed_symbols = config['universe']['fixed_symbols']
        assert fixed_symbols == ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        assert 'INVALID' not in fixed_symbols
    
    def test_case_insensitive_symbol_validation(self):
        """Test that symbol validation is case-insensitive."""
        # Set ENV with lowercase symbols
        os.environ['TRADING_SYMBOLS'] = 'btc/usdt:usdt,ETH/USDT:USDT'
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # Should accept both lowercase and uppercase
        fixed_symbols = config['universe']['fixed_symbols']
        assert len(fixed_symbols) == 2
        assert 'btc/usdt:usdt' in fixed_symbols or 'BTC/USDT:USDT' in fixed_symbols
        assert 'ETH/USDT:USDT' in fixed_symbols
    
    def test_different_symbol_formats(self):
        """Test that different symbol formats are validated correctly."""
        # Test various valid formats
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT,SOL-PERP'
        
        # Load config
        config = LiveTradingConfiguration.load(log_summary=False)
        
        # All should be valid
        fixed_symbols = config['universe']['fixed_symbols']
        assert len(fixed_symbols) == 3
        assert 'BTC/USDT:USDT' in fixed_symbols
        assert 'ETH/USDT' in fixed_symbols
        assert 'SOL-PERP' in fixed_symbols


class TestConfigDeepMerge:
    """Test deep merge functionality."""
    
    def test_deep_merge_nested_dicts(self):
        """Test that deep merge correctly merges nested dictionaries."""
        base = {
            'signals': {
                'oversold_bounce': {
                    'enable': False,
                    'rsi_max': 40
                }
            },
            'risk': {
                'equity_usd': 100
            }
        }
        
        override = {
            'signals': {
                'oversold_bounce': {
                    'enable': True  # Override this
                    # rsi_max not specified, should keep base value
                }
            }
        }
        
        result = LiveTradingConfiguration.deep_merge(base, override)
        
        # Override value should be used
        assert result['signals']['oversold_bounce']['enable'] is True
        
        # Base value should be kept
        assert result['signals']['oversold_bounce']['rsi_max'] == 40
        
        # Unrelated base value should be kept
        assert result['risk']['equity_usd'] == 100
    
    def test_deep_merge_preserves_base_on_empty_override(self):
        """Test that deep merge preserves base when override is empty."""
        base = {
            'universe': {
                'fixed_symbols': ['BTC/USDT:USDT'],
                'auto_select': False
            }
        }
        
        override = {}
        
        result = LiveTradingConfiguration.deep_merge(base, override)
        
        assert result == base
    
    def test_deep_merge_complete_override(self):
        """Test that deep merge can completely override a section."""
        base = {
            'universe': {
                'fixed_symbols': ['BTC/USDT:USDT'],
                'auto_select': False
            }
        }
        
        override = {
            'universe': {
                'fixed_symbols': ['ETH/USDT:USDT'],
                'auto_select': True
            }
        }
        
        result = LiveTradingConfiguration.deep_merge(base, override)
        
        # All values should be from override
        assert result['universe']['fixed_symbols'] == ['ETH/USDT:USDT']
        assert result['universe']['auto_select'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
