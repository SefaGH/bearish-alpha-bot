"""
Test debug logging functionality added to universe.py and live_trading_engine.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from io import StringIO
from contextlib import redirect_stdout


def test_universe_debug_logging():
    """Test that universe building produces debug logs."""
    from universe import _is_usdt_candidate
    
    # Create test market data
    test_market = {
        'symbol': 'BTC/USDT:USDT',
        'active': True,
        'quote': 'USDT',
        'swap': True,
        'linear': True
    }
    
    # Capture stdout
    captured_output = StringIO()
    with redirect_stdout(captured_output):
        result = _is_usdt_candidate(test_market, only_linear=True)
    
    # Verify result
    assert result == True, "BTC/USDT:USDT should be accepted as USDT candidate"
    
    # Verify debug logging
    output = captured_output.getvalue()
    assert '[UNIVERSE]' in output, "Debug log should contain [UNIVERSE] tag"
    assert 'BTC/USDT:USDT' in output, "Debug log should contain symbol name"
    assert 'active=True' in output, "Debug log should contain active status"
    assert 'quote=USDT' in output, "Debug log should contain quote currency"
    
    print("✓ Universe debug logging test passed")


def test_universe_rejects_non_usdt():
    """Test that non-USDT markets are rejected with logging."""
    from universe import _is_usdt_candidate
    
    test_market = {
        'symbol': 'BTC/USD',
        'active': True,
        'quote': 'USD',  # Not USDT
        'swap': True,
        'linear': True
    }
    
    captured_output = StringIO()
    with redirect_stdout(captured_output):
        result = _is_usdt_candidate(test_market, only_linear=True)
    
    assert result == False, "BTC/USD should be rejected (not USDT quote)"
    output = captured_output.getvalue()
    assert 'BTC/USD' in output, "Debug log should show rejected symbol"
    
    print("✓ Universe rejection logging test passed")


def test_universe_bingx_perpetual_logging():
    """Test that BingX perpetuals are logged correctly."""
    from universe import _is_usdt_candidate
    
    # BingX perpetual characteristics
    bingx_perpetual = {
        'symbol': 'ETH/USDT:USDT',
        'active': True,
        'quote': 'USDT',
        'swap': True,
        'linear': True
    }
    
    captured_output = StringIO()
    with redirect_stdout(captured_output):
        result = _is_usdt_candidate(bingx_perpetual, only_linear=True)
    
    assert result == True, "BingX perpetual should be accepted"
    output = captured_output.getvalue()
    assert '✅' in output or 'accepted' in output.lower(), "Should log acceptance"
    assert 'linear' in output.lower() or 'perpetual' in output.lower(), "Should mention linear/perpetual"
    
    print("✓ BingX perpetual logging test passed")


def test_live_trading_engine_import():
    """Test that live trading engine can be imported without errors."""
    try:
        from core.live_trading_engine import LiveTradingEngine
        print("✓ LiveTradingEngine import successful")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import LiveTradingEngine: {e}")


if __name__ == '__main__':
    # Run tests directly
    print("Running debug logging tests...\n")
    test_universe_debug_logging()
    test_universe_rejects_non_usdt()
    test_universe_bingx_perpetual_logging()
    test_live_trading_engine_import()
    print("\n✓ All debug logging tests passed!")
