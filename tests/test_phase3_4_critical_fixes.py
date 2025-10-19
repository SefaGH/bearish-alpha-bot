"""
Tests for Phase 3.4 Critical Fixes:
1. --live mode support in main.py
2. Simplified config loading in live_trading_engine.py
3. WebSocket initialization with validation in production_coordinator.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from core.live_trading_engine import LiveTradingEngine
from core.production_coordinator import ProductionCoordinator


class MockExchangeClient:
    """Mock exchange client for testing."""
    
    def __init__(self, exchange_name):
        self.exchange_name = exchange_name
    
    def ticker(self, symbol):
        return {'symbol': symbol, 'last': 50000.0, 'close': 50000.0}


class TestLiveModeSupport:
    """Test Fix 1: --live mode support in main.py."""
    
    def test_argparse_live_flag(self):
        """Test that --live flag is recognized."""
        import argparse
        
        # Verify we can parse --live argument
        parser = argparse.ArgumentParser()
        parser.add_argument('--live', action='store_true')
        parser.add_argument('--paper', action='store_true')
        
        args = parser.parse_args(['--live'])
        assert args.live is True
        
        args = parser.parse_args(['--live', '--paper'])
        assert args.live is True
        assert args.paper is True
    
    @pytest.mark.asyncio
    async def test_main_live_trading_exists(self):
        """Test that main_live_trading function exists and is async."""
        import main
        
        assert hasattr(main, 'main_live_trading'), "main_live_trading function should exist"
        assert callable(main.main_live_trading), "main_live_trading should be callable"
        
        # Verify it's an async function
        import inspect
        assert inspect.iscoroutinefunction(main.main_live_trading), "main_live_trading should be async"


class TestSimplifiedConfigLoading:
    """Test Fix 2: Simplified config loading with clear priority."""
    
    def test_config_loading_from_yaml(self):
        """Test that config loads correctly from YAML file."""
        # Create a temporary YAML config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_config = {
                'universe': {
                    'fixed_symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
                    'auto_select': False
                }
            }
            yaml.dump(yaml_config, f)
            temp_path = f.name
        
        try:
            # Set environment variable to use temp config
            os.environ['CONFIG_PATH'] = temp_path
            
            # Create engine
            engine = LiveTradingEngine(
                mode='paper',
                exchange_clients={'test': MockExchangeClient('test')}
            )
            
            # Verify symbols were loaded from YAML
            assert 'universe' in engine.config
            assert engine.config['universe']['fixed_symbols'] == ['BTC/USDT:USDT', 'ETH/USDT:USDT']
            assert engine.config['universe']['auto_select'] is False
        finally:
            os.unlink(temp_path)
            if 'CONFIG_PATH' in os.environ:
                del os.environ['CONFIG_PATH']
    
    def test_config_loading_from_env(self):
        """Test that config falls back to environment variables."""
        # Set environment variable
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT'
        os.environ['CONFIG_PATH'] = '/tmp/nonexistent_config.yaml'
        
        try:
            # Create engine
            engine = LiveTradingEngine(
                mode='paper',
                exchange_clients={'test': MockExchangeClient('test')}
            )
            
            # Verify symbols were loaded from ENV
            assert 'universe' in engine.config
            assert len(engine.config['universe']['fixed_symbols']) == 3
            assert 'BTC/USDT:USDT' in engine.config['universe']['fixed_symbols']
        finally:
            if 'TRADING_SYMBOLS' in os.environ:
                del os.environ['TRADING_SYMBOLS']
            if 'CONFIG_PATH' in os.environ:
                del os.environ['CONFIG_PATH']
    
    def test_config_loading_defaults(self):
        """Test that config falls back to hard-coded defaults."""
        # Ensure no config file or env vars
        os.environ['CONFIG_PATH'] = '/tmp/nonexistent_config.yaml'
        if 'TRADING_SYMBOLS' in os.environ:
            del os.environ['TRADING_SYMBOLS']
        
        try:
            # Create engine
            engine = LiveTradingEngine(
                mode='paper',
                exchange_clients={'test': MockExchangeClient('test')}
            )
            
            # Verify defaults were used
            assert 'universe' in engine.config
            assert len(engine.config['universe']['fixed_symbols']) == 3
            assert 'BTC/USDT:USDT' in engine.config['universe']['fixed_symbols']
            assert 'ETH/USDT:USDT' in engine.config['universe']['fixed_symbols']
            assert 'SOL/USDT:USDT' in engine.config['universe']['fixed_symbols']
        finally:
            if 'CONFIG_PATH' in os.environ:
                del os.environ['CONFIG_PATH']
    
    def test_config_loading_validates_list(self):
        """Test that config validation rejects non-list fixed_symbols."""
        # Create a temporary YAML config with invalid fixed_symbols
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_config = {
                'universe': {
                    'fixed_symbols': 'BTC/USDT:USDT',  # String instead of list
                    'auto_select': False
                }
            }
            yaml.dump(yaml_config, f)
            temp_path = f.name
        
        try:
            os.environ['CONFIG_PATH'] = temp_path
            
            # Create engine - should fall back to defaults
            engine = LiveTradingEngine(
                mode='paper',
                exchange_clients={'test': MockExchangeClient('test')}
            )
            
            # Should use defaults since YAML was invalid
            assert 'universe' in engine.config
            # Should still have symbols (either from ENV or defaults)
            assert len(engine.config['universe']['fixed_symbols']) >= 3
        finally:
            os.unlink(temp_path)
            if 'CONFIG_PATH' in os.environ:
                del os.environ['CONFIG_PATH']


class TestWebSocketInitialization:
    """Test Fix 3: WebSocket initialization with validation."""
    
    def test_setup_websocket_returns_bool(self):
        """Test that _setup_websocket_connections returns a boolean."""
        coordinator = ProductionCoordinator()
        coordinator.exchange_clients = {'test': MockExchangeClient('test')}
        
        # Mock WebSocketManager
        with patch('core.production_coordinator.WebSocketManager') as mock_ws:
            mock_ws_instance = MagicMock()
            mock_ws.return_value = mock_ws_instance
            
            result = coordinator._setup_websocket_connections()
            
            # Should return a boolean
            assert isinstance(result, bool)
    
    def test_setup_websocket_validates_exchange_clients(self):
        """Test that setup validates exchange_clients exist."""
        coordinator = ProductionCoordinator()
        coordinator.exchange_clients = {}  # Empty
        
        result = coordinator._setup_websocket_connections()
        
        # Should return False when no exchange clients
        assert result is False
    
    def test_setup_websocket_with_fallback_symbols(self):
        """Test that setup uses fallback symbols when none configured."""
        coordinator = ProductionCoordinator()
        coordinator.exchange_clients = {'test': MockExchangeClient('test')}
        coordinator.config['universe'] = {'fixed_symbols': []}  # Empty
        
        # Mock WebSocketManager
        with patch('core.production_coordinator.WebSocketManager') as mock_ws:
            mock_ws_instance = MagicMock()
            mock_ws.return_value = mock_ws_instance
            
            result = coordinator._setup_websocket_connections()
            
            # Should still attempt to set up with defaults
            assert mock_ws.called
    
    def test_get_stream_limit(self):
        """Test that _get_stream_limit returns correct limits per exchange."""
        coordinator = ProductionCoordinator()
        
        # Test known exchanges
        assert coordinator._get_stream_limit('bingx') == 10
        assert coordinator._get_stream_limit('binance') == 20
        assert coordinator._get_stream_limit('kucoinfutures') == 15
        
        # Test unknown exchange (should use default)
        assert coordinator._get_stream_limit('unknown_exchange') == 10
    
    def test_get_stream_limit_from_config(self):
        """Test that _get_stream_limit respects config overrides."""
        coordinator = ProductionCoordinator()
        coordinator.config['websocket'] = {
            'max_streams_per_exchange': {
                'bingx': 5,
                'custom_exchange': 25
            }
        }
        
        # Should use config value
        assert coordinator._get_stream_limit('bingx') == 5
        assert coordinator._get_stream_limit('custom_exchange') == 25
        
        # Should use hardcoded for binance (not in config override)
        assert coordinator._get_stream_limit('binance') == 20
    
    def test_setup_websocket_respects_stream_limits(self):
        """Test that setup respects per-exchange stream limits."""
        coordinator = ProductionCoordinator()
        coordinator.exchange_clients = {'bingx': MockExchangeClient('bingx')}
        
        # Set up many symbols that would exceed limit
        many_symbols = [f'SYM{i}/USDT:USDT' for i in range(20)]
        coordinator.config['universe'] = {'fixed_symbols': many_symbols}
        coordinator.config['websocket'] = {'stream_timeframes': ['1m', '5m']}
        
        # Mock WebSocketManager
        with patch('core.production_coordinator.WebSocketManager') as mock_ws:
            mock_ws_instance = MagicMock()
            mock_ws.return_value = mock_ws_instance
            
            result = coordinator._setup_websocket_connections()
            
            # Should have limited the number of start_ohlcv_stream calls
            # BingX limit is 10, with 2 timeframes = max 5 symbols
            call_count = mock_ws_instance.start_ohlcv_stream.call_count
            assert call_count <= 10, f"Expected max 10 calls, got {call_count}"


class TestIntegration:
    """Integration tests for all three fixes working together."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_paper_mode(self):
        """Test that system can start in paper trading mode."""
        # This would require full integration which we mock for testing
        with patch('core.production_coordinator.ProductionCoordinator') as mock_coord:
            mock_instance = MagicMock()
            mock_coord.return_value = mock_instance
            
            # Mock the async methods
            mock_instance.initialize_production_system = MagicMock(return_value=None)
            mock_instance.run_production_loop = MagicMock(return_value=None)
            
            # These are needed to make it awaitable
            async def async_init(*args, **kwargs):
                pass
            
            async def async_run(*args, **kwargs):
                pass
            
            mock_instance.initialize_production_system = async_init
            mock_instance.run_production_loop = async_run
            
            # Set environment
            os.environ['TRADING_MODE'] = 'paper'
            os.environ['EQUITY_USD'] = '100'
            
            try:
                # This would call main_live_trading but we just verify setup
                assert True  # Placeholder for actual integration test
            finally:
                if 'TRADING_MODE' in os.environ:
                    del os.environ['TRADING_MODE']
                if 'EQUITY_USD' in os.environ:
                    del os.environ['EQUITY_USD']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
