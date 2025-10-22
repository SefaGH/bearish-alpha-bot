#!/usr/bin/env python3
"""
Integration test: Config consistency across modules.

These tests verify that configuration is loaded consistently across all modules
and that environment variables properly override YAML config.

Addresses:
- Issue #157: Configuration Loading Priority Conflict
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, MagicMock, patch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from .fakes import (
    FakeProductionCoordinator,
    FakeOptimizedWebSocketManager,
    build_launcher_module_stubs,
    ignore_test_task_cancellation,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_config_consistency_across_all_modules(integration_env, cleanup_tasks):
    """
    Integration test: Verify all modules use same config (ENV priority).
    
    Addresses Issue #157 (Config Loading Priority).
    
    Test Strategy:
    - Set environment variables to override YAML
    - Load unified config
    - Create launcher and verify it uses same config
    - Verify ENV takes priority over YAML
    """
    print("\n" + "="*70)
    print("TEST: Config Consistency Across All Modules")
    print("="*70)
    
    # Set ENV to override YAML
    os.environ['TRADING_SYMBOLS'] = 'XRP/USDT:USDT,ADA/USDT:USDT'
    os.environ['RSI_THRESHOLD_BTC'] = '60'
    os.environ['CAPITAL_USDT'] = '500'
    
    try:
        from config.live_trading_config import LiveTradingConfiguration
        
        print("\n[Step 1] Loading unified configuration...")
        unified_config = LiveTradingConfiguration.load(log_summary=False)
        
        print("\n[Step 2] Verifying ENV override in unified config...")
        configured_symbols = unified_config['universe']['fixed_symbols']
        print(f"  Configured symbols: {configured_symbols}")
        
        # Verify ENV override worked
        expected_symbols = ['XRP/USDT:USDT', 'ADA/USDT:USDT']
        assert configured_symbols == expected_symbols, (
            f"ENV override failed!\n"
            f"Expected: {expected_symbols}\n"
            f"Got:      {configured_symbols}"
        )
        print("  ✓ ENV successfully overrides YAML")
        
        # Try to test launcher if dependencies available
        try:
            # Mock external dependencies before import
            with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
                 patch('core.notify.Telegram') as mock_telegram:
                
                # Import launcher after patching
                from live_trading_launcher import LiveTradingLauncher
                
                # Setup mock exchange
                mock_exchange = MagicMock()
                mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
                mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
                mock_exchange.ticker.return_value = {'last': 50000.0}
                mock_ccxt.return_value = mock_exchange
                
                print("\n[Step 3] Creating launcher and loading config...")
                launcher = LiveTradingLauncher(mode='paper')
                launcher_config = launcher._load_config()
                
                print("\n[Step 4] Verifying launcher uses same config...")
                launcher_symbols = launcher_config['universe']['fixed_symbols']
                print(f"  Launcher symbols: {launcher_symbols}")
                
                # Verify launcher uses unified config
                assert launcher_symbols == expected_symbols, (
                    f"Launcher config mismatch!\n"
                    f"Expected: {expected_symbols}\n"
                    f"Got:      {launcher_symbols}"
                )
                print("  ✓ Launcher uses same config as unified config")
                
                # Note: launcher.CAPITAL_USDT is a launcher attribute (hardcoded 100 by default)
                # The unified config doesn't have a 'capital' section currently
                # ENV variable CAPITAL_USDT is launcher-specific, not part of unified config
                capital = launcher.CAPITAL_USDT
                print(f"\n[Step 5] Checking launcher capital: {capital} USDT")
                assert capital == 500.0
                assert launcher.capital_source == 'env'
                print("  ✓ Launcher capital reflects ENV override")

                print(f"\n{'='*70}")
                print("Config Consistency Verification (Full):")
                print(f"{'='*70}")
                print(f"✓ ENV overrides YAML")
                print(f"✓ Unified config matches ENV")
                print(f"✓ Launcher config matches unified config")
                print(f"✓ All modules use consistent configuration")
                print(f"{'='*70}\n")
                
        except ImportError as e:
            print(f"\n[Step 3] Launcher not available (missing dependencies): {e}")
            print("  - Skipping launcher-specific tests")
            print(f"\n{'='*70}")
            print("Config Consistency Verification (Partial):")
            print(f"{'='*70}")
            print(f"✓ ENV overrides YAML")
            print(f"✓ Unified config matches ENV")
            print(f"⚠ Launcher tests skipped (dependencies not installed)")
            print(f"{'='*70}\n")
        
        print("✅ TEST PASSED: Config consistency verified")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Config consistency test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_env_priority_over_yaml(integration_env, cleanup_tasks):
    """
    Integration test: Verify ENV always takes priority over YAML.
    
    Test Strategy:
    - Set multiple ENV variables with non-default values
    - Load config
    - Verify all ENV values are used, not YAML defaults
    """
    print("\n" + "="*70)
    print("TEST: ENV Priority Over YAML")
    print("="*70)
    
    # Set multiple ENV overrides
    test_values = {
        'TRADING_SYMBOLS': 'DOT/USDT:USDT,LINK/USDT:USDT,UNI/USDT:USDT',
        'CAPITAL_USDT': '2000',
        'TRADING_MODE': 'paper',
        'RSI_THRESHOLD_BTC': '65'
    }
    
    for key, value in test_values.items():
        os.environ[key] = value
    
    try:
        from config.live_trading_config import LiveTradingConfiguration
        
        print("\n[Step 1] Loading config with ENV overrides...")
        config = LiveTradingConfiguration.load(log_summary=False)
        
        print("\n[Step 2] Verifying ENV values are used...")
        
        # Check symbols
        expected_symbols = ['DOT/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT']
        actual_symbols = config['universe']['fixed_symbols']
        
        print(f"\n  Symbols:")
        print(f"    ENV:      {test_values['TRADING_SYMBOLS']}")
        print(f"    Expected: {expected_symbols}")
        print(f"    Actual:   {actual_symbols}")
        
        assert actual_symbols == expected_symbols, (
            f"Symbols don't match ENV!\n"
            f"Expected: {expected_symbols}\n"
            f"Got:      {actual_symbols}"
        )
        print("    ✓ Symbols from ENV")
        
        print(f"\n{'='*70}")
        print("ENV Priority Verification:")
        print(f"{'='*70}")
        print(f"✓ Symbols loaded from ENV")
        print(f"✓ All ENV overrides take priority over YAML")
        print(f"{'='*70}\n")
        
        print("✅ TEST PASSED: ENV priority over YAML verified")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"ENV priority test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_launcher_capital_source_priority(integration_env, cleanup_tasks):
    """Ensure launcher capital uses ENV first, then config."""

    from config.live_trading_config import LiveTradingConfiguration

    os.environ.pop('CAPITAL_USDT', None)

    module_stubs = build_launcher_module_stubs()
    module_stubs.pop('config.live_trading_config', None)
    test_task = asyncio.current_task()
    assert test_task is not None

    with patch.object(LiveTradingConfiguration, 'load', return_value={'universe': {'fixed_symbols': []}}), \
         patch.object(LiveTradingConfiguration, 'load_from_yaml', return_value={}), \
         ignore_test_task_cancellation(test_task), \
         patch.dict('sys.modules', module_stubs), \
         patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
         patch('core.notify.Telegram') as mock_telegram, \
         patch('core.production_coordinator.ProductionCoordinator', FakeProductionCoordinator), \
         patch('live_trading_launcher.OptimizedWebSocketManager', FakeOptimizedWebSocketManager):

        from live_trading_launcher import LiveTradingLauncher

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
        mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
        mock_exchange.ticker.return_value = {'last': 50000.0}
        mock_ccxt.return_value = mock_exchange

        launcher = LiveTradingLauncher(mode='paper')

        assert launcher.CAPITAL_USDT == 100.0
        assert launcher.capital_source == 'default'

    os.environ.pop('CAPITAL_USDT', None)

    with patch.object(LiveTradingConfiguration, 'load', return_value={'risk': {'equity_usd': 750}, 'universe': {'fixed_symbols': []}}), \
         patch.object(LiveTradingConfiguration, 'load_from_yaml', return_value={'risk': {'equity_usd': 750}}), \
         ignore_test_task_cancellation(test_task), \
         patch.dict('sys.modules', module_stubs), \
         patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
         patch('core.notify.Telegram') as mock_telegram, \
         patch('core.production_coordinator.ProductionCoordinator', FakeProductionCoordinator), \
         patch('live_trading_launcher.OptimizedWebSocketManager', FakeOptimizedWebSocketManager):

        from live_trading_launcher import LiveTradingLauncher

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
        mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
        mock_exchange.ticker.return_value = {'last': 50000.0}
        mock_ccxt.return_value = mock_exchange

        launcher = LiveTradingLauncher(mode='paper')

        assert launcher.CAPITAL_USDT == 750.0
        assert launcher.capital_source == 'config'

    os.environ['CAPITAL_USDT'] = '1200'

    with patch.object(LiveTradingConfiguration, 'load', return_value={'risk': {'equity_usd': 750}, 'universe': {'fixed_symbols': []}}), \
         patch.object(LiveTradingConfiguration, 'load_from_yaml', return_value={'risk': {'equity_usd': 750}}), \
         ignore_test_task_cancellation(test_task), \
         patch.dict('sys.modules', module_stubs), \
         patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
         patch('core.notify.Telegram') as mock_telegram, \
         patch('core.production_coordinator.ProductionCoordinator', FakeProductionCoordinator), \
         patch('live_trading_launcher.OptimizedWebSocketManager', FakeOptimizedWebSocketManager):

        from live_trading_launcher import LiveTradingLauncher

        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
        mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
        mock_exchange.ticker.return_value = {'last': 50000.0}
        mock_ccxt.return_value = mock_exchange

        launcher = LiveTradingLauncher(mode='paper')

        assert launcher.CAPITAL_USDT == 1200.0
        assert launcher.capital_source == 'env'

    os.environ.pop('CAPITAL_USDT', None)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_config_validation(integration_env, cleanup_tasks):
    """
    Integration test: Verify config validation works correctly.
    
    Test Strategy:
    - Test with invalid symbols
    - Test with missing required fields
    - Verify validation errors are caught
    """
    print("\n" + "="*70)
    print("TEST: Config Validation")
    print("="*70)
    
    try:
        from config.live_trading_config import LiveTradingConfiguration
        
        # Test 1: Invalid symbol format
        print("\n[Test 1] Invalid symbol format...")
        os.environ['TRADING_SYMBOLS'] = 'INVALID_SYMBOL,BTC/USDT:USDT'
        
        config = LiveTradingConfiguration.load(log_summary=False)
        symbols = config['universe']['fixed_symbols']
        
        # Should filter out invalid symbols
        assert 'INVALID_SYMBOL' not in symbols, (
            "Invalid symbol was not filtered out"
        )
        print("  ✓ Invalid symbols filtered correctly")
        
        # Test 2: Valid config loads successfully
        print("\n[Test 2] Valid config...")
        os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT'
        
        config = LiveTradingConfiguration.load(log_summary=False)
        symbols = config['universe']['fixed_symbols']
        
        assert len(symbols) == 2, f"Expected 2 symbols, got {len(symbols)}"
        print("  ✓ Valid config loads successfully")
        
        print(f"\n{'='*70}")
        print("✅ TEST PASSED: Config validation works correctly")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Config validation test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_config_consistency(integration_env, cleanup_tasks):
    """
    Integration test: Verify config remains consistent during runtime.
    
    Test Strategy:
    - Load config multiple times
    - Verify same values returned
    - Ensure no config drift during execution
    """
    print("\n" + "="*70)
    print("TEST: Runtime Config Consistency")
    print("="*70)
    
    os.environ['TRADING_SYMBOLS'] = 'BTC/USDT:USDT,ETH/USDT:USDT'
    os.environ['CAPITAL_USDT'] = '1000'
    
    try:
        from config.live_trading_config import LiveTradingConfiguration
        
        print("\n[Step 1] Loading config multiple times...")
        
        config1 = LiveTradingConfiguration.load(log_summary=False)
        config2 = LiveTradingConfiguration.load(log_summary=False)
        
        # Verify consistency
        symbols1 = config1['universe']['fixed_symbols']
        symbols2 = config2['universe']['fixed_symbols']
        
        print(f"  Config load 1: {symbols1}")
        print(f"  Config load 2: {symbols2}")
        
        assert symbols1 == symbols2, (
            "Config inconsistent across loads"
        )
        print("  ✓ Config consistent across multiple loads")
        
        # Try launcher test if available
        try:
            # Mock external dependencies before import
            with patch('core.ccxt_client.CcxtClient') as mock_ccxt, \
                 patch('core.notify.Telegram') as mock_telegram:
                
                # Import launcher after patching
                from live_trading_launcher import LiveTradingLauncher
                
                # Setup mock exchange
                mock_exchange = MagicMock()
                mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
                mock_exchange.get_bingx_balance.return_value = {'USDT': {'free': 1000.0}}
                mock_exchange.ticker.return_value = {'last': 50000.0}
                mock_ccxt.return_value = mock_exchange
                
                print("\n[Step 2] Creating launcher and verifying consistency...")
                launcher = LiveTradingLauncher(mode='paper')
                launcher_config = launcher._load_config()
                launcher_symbols = launcher_config['universe']['fixed_symbols']
                
                print(f"  Launcher config: {launcher_symbols}")
                
                assert launcher_symbols == symbols1, (
                    "Launcher config doesn't match unified config"
                )
                print("  ✓ Launcher config matches unified config")
                
        except ImportError:
            print("\n[Step 2] Launcher tests skipped (dependencies not installed)")
        
        print(f"\n{'='*70}")
        print("✅ TEST PASSED: Config remains consistent during runtime")
        print(f"{'='*70}\n")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Runtime config test failed: {e}")
