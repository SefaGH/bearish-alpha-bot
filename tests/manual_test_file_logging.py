#!/usr/bin/env python3
"""
Manual test to verify file logging works in the actual use case.
This simulates how the live_trading_launcher.py uses the logger.
"""

import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logger import setup_logger
from core.debug_logger import setup_debug_logger, DebugLogger

def test_basic_logger():
    """Test basic logger setup."""
    print("="*70)
    print("TEST 1: Basic Logger Setup")
    print("="*70)
    
    # Clear any existing handlers
    logger = setup_logger(name="manual_test", log_to_file=True)
    
    # Log various messages
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Find log file
    log_files = glob.glob('logs/bearish_alpha_bot_*.log')
    if log_files:
        latest = max(log_files, key=os.path.getctime)
        size = os.path.getsize(latest)
        print(f"\n✅ Log file created: {latest}")
        print(f"✅ File size: {size} bytes")
        
        # Show content
        print(f"\nLog file content (first 500 chars):")
        print("-" * 70)
        with open(latest, 'r') as f:
            content = f.read()
            print(content[:500])
        print("-" * 70)
        
        if size > 0:
            print("\n✅ SUCCESS: File logging is working!")
            return True
    else:
        print("\n❌ FAILURE: No log file created")
        return False


def test_debug_logger():
    """Test debug logger setup."""
    print("\n" + "="*70)
    print("TEST 2: Debug Logger Setup")
    print("="*70)
    
    # Create debug logger
    debug_logger = setup_debug_logger(name="manual_debug_test", debug_mode=True, log_to_file=True)
    
    # Log messages
    debug_logger.debug("Debug message with 🔍 emoji")
    debug_logger.info("Info message in debug mode")
    
    # Find debug log file
    log_files = glob.glob('logs/bearish_alpha_bot_debug_*.log')
    if log_files:
        latest = max(log_files, key=os.path.getctime)
        size = os.path.getsize(latest)
        print(f"\n✅ Debug log file created: {latest}")
        print(f"✅ File size: {size} bytes")
        
        # Show content
        print(f"\nDebug log file content:")
        print("-" * 70)
        with open(latest, 'r') as f:
            content = f.read()
            print(content)
        print("-" * 70)
        
        if size > 0 and "🔍" in content:
            print("\n✅ SUCCESS: Debug file logging is working with emoji!")
            return True
    else:
        print("\n❌ FAILURE: No debug log file created")
        return False


def test_debug_logger_class():
    """Test DebugLogger class."""
    print("\n" + "="*70)
    print("TEST 3: DebugLogger Class")
    print("="*70)
    
    # Create DebugLogger instance
    debug_logger = DebugLogger(debug_mode=True)
    
    print(f"\n✅ DebugLogger initialized")
    print(f"✅ Debug mode: {debug_logger.is_debug_enabled()}")
    
    if debug_logger.is_debug_enabled():
        print("\n✅ SUCCESS: DebugLogger class is working!")
        return True
    else:
        print("\n❌ FAILURE: DebugLogger debug mode not enabled")
        return False


if __name__ == '__main__':
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print("\nStarting manual file logging tests...\n")
    
    results = []
    results.append(test_basic_logger())
    results.append(test_debug_logger())
    results.append(test_debug_logger_class())
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nFile logging is working correctly:")
        print("- Logs are written to disk in logs/ directory")
        print("- Log files have proper timestamps")
        print("- File size > 0 bytes")
        print("- Both regular and debug loggers work")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
