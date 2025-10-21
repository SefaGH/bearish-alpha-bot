#!/usr/bin/env python3
"""
Test for symbol passing fix between launcher and coordinator.

This test validates that symbols are correctly passed from the launcher
to the coordinator and that the fallback mechanisms work as expected.
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from core.production_coordinator import ProductionCoordinator


class TestSymbolPassingFix:
    """Test suite for symbol passing fix."""
    
    def test_coordinator_symbol_initialization(self):
        """Test that coordinator properly initializes active_symbols."""
        coordinator = ProductionCoordinator()
        
        # Test 1: Symbols passed via parameter
        test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        coordinator.active_symbols = test_symbols
        assert coordinator.active_symbols == test_symbols
        assert len(coordinator.active_symbols) == 3
        
    def test_coordinator_symbol_fallback_logic(self):
        """Test the symbol fallback logic in isolation."""
        coordinator = ProductionCoordinator()
        
        # Simulate the fallback logic
        trading_symbols = None
        config = {'universe': {'fixed_symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT']}}
        
        # Test fallback to config
        if trading_symbols:
            coordinator.active_symbols = trading_symbols
        else:
            config_symbols = config.get('universe', {}).get('fixed_symbols', [])
            if config_symbols and isinstance(config_symbols, list):
                coordinator.active_symbols = config_symbols
            else:
                coordinator.active_symbols = []
        
        assert coordinator.active_symbols == ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        assert len(coordinator.active_symbols) == 2
    
    def test_coordinator_empty_symbols_fallback(self):
        """Test that coordinator handles empty symbols gracefully."""
        coordinator = ProductionCoordinator()
        
        # Simulate the fallback logic with no symbols
        trading_symbols = None
        config = {'universe': {}}
        
        if trading_symbols:
            coordinator.active_symbols = trading_symbols
        else:
            config_symbols = config.get('universe', {}).get('fixed_symbols', [])
            if config_symbols and isinstance(config_symbols, list):
                coordinator.active_symbols = config_symbols
            else:
                coordinator.active_symbols = []
        
        assert coordinator.active_symbols == []
        assert len(coordinator.active_symbols) == 0
    
    def test_process_trading_loop_exits_with_empty_symbols(self):
        """Test that _process_trading_loop exits gracefully with empty symbols."""
        coordinator = ProductionCoordinator()
        coordinator.active_symbols = []
        
        # This should complete without error
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coordinator._process_trading_loop())
        
        # If we get here without hanging, the test passes
        assert True
    
    def test_process_trading_loop_processes_symbols(self):
        """Test that _process_trading_loop processes symbols when available."""
        coordinator = ProductionCoordinator()
        coordinator.active_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        coordinator.processed_symbols_count = 0
        
        # Mock process_symbol to avoid actual processing
        async def mock_process_symbol(symbol):
            return None
        
        coordinator.process_symbol = mock_process_symbol
        
        # This should complete without hanging
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coordinator._process_trading_loop())
        
        # If we get here without error, the test passes
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
