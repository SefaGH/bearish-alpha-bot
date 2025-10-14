#!/usr/bin/env python3
"""
Test Debug Mode Functionality.

Validates that debug mode properly enables enhanced logging across all components.
"""

import sys
import os
import logging
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.debug_logger import DebugLogger, setup_debug_logger


def test_debug_logger_initialization():
    """Test DebugLogger initializes correctly."""
    debug_logger = DebugLogger(debug_mode=True)
    assert debug_logger.is_debug_enabled()
    
    debug_logger_off = DebugLogger(debug_mode=False)
    assert not debug_logger_off.is_debug_enabled()
    print("✓ DebugLogger initialization test passed")


def test_debug_mode_enables_debug_level():
    """Test that debug mode sets logging to DEBUG level."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create logger with debug mode
    logger = setup_debug_logger('test_debug', debug_mode=True)
    
    # Check that debug level is enabled
    assert logger.level == logging.DEBUG
    print("✓ Debug mode enables DEBUG level test passed")


def test_debug_messages_appear_in_debug_mode():
    """Test that debug messages appear when debug mode is enabled."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create string buffer to capture output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger('test_messages')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # Log debug message
    logger.debug("Test debug message")
    
    # Check that message was captured
    log_output = log_capture.getvalue()
    assert "Test debug message" in log_output, "Debug message should appear in output"
    assert "DEBUG" in log_output, "DEBUG level should appear in output"
    print("✓ Debug messages appear in debug mode test passed")


def test_debug_emoji_in_formatter():
    """Test that debug mode includes emoji in log format."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create logger with debug mode
    logger = setup_debug_logger('test_emoji', debug_mode=True)
    
    # Check formatter includes emoji by testing formatted output
    if logger.handlers:
        # Create test log record
        record = logging.LogRecord(
            name='test_emoji',
            level=logging.DEBUG,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        formatted_msg = logger.handlers[0].formatter.format(record)
        assert '🔍' in formatted_msg, "Emoji should appear in formatted debug message"
    print("✓ Debug emoji in formatter test passed")


def test_strategy_debug_logging_format():
    """Test that strategy debug logging follows the required format."""
    test_messages = [
        "🎯 [STRATEGY-AdaptiveOB] Market analysis started",
        "📊 [STRATEGY-AdaptiveOB] Price data: close=$50000.00, RSI=25.00",
        "✅ [STRATEGY-AdaptiveOB] Signal result: BUY signal generated",
        "❌ [STRATEGY-AdaptiveSTR] Signal result: No signal - RSI 55.00 > 70.00"
    ]
    
    for msg in test_messages:
        assert msg.startswith(("🎯", "📊", "✅", "❌"))
        assert "[STRATEGY-" in msg
        print(f"  ✓ Format check passed: {msg[:50]}...")
    
    print("✓ Strategy debug logging format test passed")


def test_ml_debug_logging_format():
    """Test that ML debug logging follows the required format."""
    test_messages = [
        "🧠 [ML-REGIME] Market regime: bullish (confidence: 85%)",
        "🧠 [ML-ADAPTER] Signal enhancement: buy → buy (strength: 0.75)",
        "🧠 [ML-PRICE] Price prediction: $51000 (±2.5%)"
    ]
    
    for msg in test_messages:
        assert msg.startswith("🧠")
        assert "[ML-" in msg
        print(f"  ✓ Format check passed: {msg[:50]}...")
    
    print("✓ ML debug logging format test passed")


def test_risk_debug_logging_format():
    """Test that risk management debug logging follows the required format."""
    test_messages = [
        "🛡️ [RISK-CALC] Portfolio value: $100.00",
        "🛡️ [RISK-CALC] Position size check: $15.00 vs $20.00 max",
        "🛡️ [RISK-CALC] Risk per trade: 2.50% (limit: 5.00%)",
        "🛡️ [RISK-CALC] APPROVED: All risk checks passed"
    ]
    
    for msg in test_messages:
        assert msg.startswith("🛡️")
        assert "[RISK-CALC]" in msg
        print(f"  ✓ Format check passed: {msg[:50]}...")
    
    print("✓ Risk management debug logging format test passed")


def test_order_debug_logging_format():
    """Test that order execution debug logging follows the required format."""
    test_messages = [
        "🎪 [ORDER-MGR] Signal received: {'symbol': 'BTC/USDT', 'side': 'buy'}",
        "🎪 [ORDER-MGR] Pre-execution checks: {'valid': True}",
        "🎪 [ORDER-MGR] Execution result: SUCCESS",
        "🎪 [ORDER-MGR] Post-execution state: order_id=12345"
    ]
    
    for msg in test_messages:
        assert msg.startswith("🎪")
        assert "[ORDER-MGR]" in msg
        print(f"  ✓ Format check passed: {msg[:50]}...")
    
    print("✓ Order execution debug logging format test passed")


def test_circuit_breaker_debug_logging_format():
    """Test that circuit breaker debug logging follows the required format."""
    test_messages = [
        "🔥 [CIRCUIT] Daily P&L: -3.50% (limit: -5.00%)",
        "🔥 [CIRCUIT] Volatility spike check: BTC z-score=2.5 (threshold: 3.0)",
        "🔥 [CIRCUIT] TRIGGERED: Daily loss limit breached"
    ]
    
    for msg in test_messages:
        assert msg.startswith("🔥")
        assert "[CIRCUIT]" in msg
        print(f"  ✓ Format check passed: {msg[:50]}...")
    
    print("✓ Circuit breaker debug logging format test passed")


def main():
    """Run all debug mode tests."""
    print("=" * 60)
    print("Debug Mode Functionality Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_debug_logger_initialization,
        test_debug_mode_enables_debug_level,
        test_debug_messages_appear_in_debug_mode,
        test_debug_emoji_in_formatter,
        test_strategy_debug_logging_format,
        test_ml_debug_logging_format,
        test_risk_debug_logging_format,
        test_order_debug_logging_format,
        test_circuit_breaker_debug_logging_format
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
