"""
Test file logging functionality for issue #127.
Verifies that logs are written to disk and not just console.
"""

import sys
import os
import tempfile
import shutil
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import logging
from pathlib import Path


def test_logger_creates_file():
    """Test that setup_logger creates a log file."""
    from core.logger import setup_logger
    
    # Clean up any existing handlers
    logger_name = "test_file_logger"
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()
    
    # Create logger with file logging enabled
    test_logger = setup_logger(name=logger_name, log_to_file=True)
    
    # Write a test message
    test_message = "Test log message for file logging verification"
    test_logger.info(test_message)
    
    # Find the created log file
    log_files = glob.glob('logs/bearish_alpha_bot_*.log')
    assert len(log_files) > 0, "No log file was created"
    
    # Get the most recent log file
    latest_log = max(log_files, key=os.path.getctime)
    
    # Verify file exists and is not empty
    assert os.path.exists(latest_log), f"Log file {latest_log} does not exist"
    assert os.path.getsize(latest_log) > 0, f"Log file {latest_log} is empty (0 bytes)"
    
    # Verify the log message is in the file
    with open(latest_log, 'r') as f:
        content = f.read()
        assert test_message in content, f"Test message not found in log file"
        assert "test_file_logger" in content, "Logger name not found in log file"
    
    print(f"✓ File logging test passed - Log file: {latest_log} ({os.path.getsize(latest_log)} bytes)")
    
    # Clean up
    test_logger.handlers.clear()


def test_logger_without_file():
    """Test that setup_logger works without file logging."""
    from core.logger import setup_logger
    
    # Clean up any existing handlers
    logger_name = "test_console_only"
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()
    
    # Count existing log files
    log_files_before = set(glob.glob('logs/bearish_alpha_bot_*.log'))
    
    # Create logger with file logging disabled
    test_logger = setup_logger(name=logger_name, log_to_file=False)
    
    # Write a test message
    test_logger.info("Console only message")
    
    # Verify no new log file was created
    log_files_after = set(glob.glob('logs/bearish_alpha_bot_*.log'))
    new_files = log_files_after - log_files_before
    
    assert len(new_files) == 0, "Log file was created when log_to_file=False"
    
    # Verify logger has only StreamHandler
    handler_types = [type(h).__name__ for h in test_logger.handlers]
    assert 'StreamHandler' in handler_types, "StreamHandler not found"
    assert 'FileHandler' not in handler_types, "FileHandler found when log_to_file=False"
    
    print("✓ Console-only logging test passed")
    
    # Clean up
    test_logger.handlers.clear()


def test_debug_logger_creates_file():
    """Test that setup_debug_logger creates a log file."""
    from core.debug_logger import setup_debug_logger
    
    # Clean up any existing handlers
    logger_name = "test_debug_file_logger"
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()
    
    # Create debug logger with file logging enabled
    test_logger = setup_debug_logger(name=logger_name, debug_mode=True, log_to_file=True)
    
    # Write a test message
    test_message = "Debug test log message"
    test_logger.debug(test_message)
    
    # Find the created log file
    log_files = glob.glob('logs/bearish_alpha_bot_debug_*.log')
    assert len(log_files) > 0, "No debug log file was created"
    
    # Get the most recent log file
    latest_log = max(log_files, key=os.path.getctime)
    
    # Verify file exists and is not empty
    assert os.path.exists(latest_log), f"Debug log file {latest_log} does not exist"
    assert os.path.getsize(latest_log) > 0, f"Debug log file {latest_log} is empty (0 bytes)"
    
    # Verify the log message is in the file
    with open(latest_log, 'r') as f:
        content = f.read()
        assert test_message in content, f"Test message not found in debug log file"
        # Check for debug emoji in debug mode
        assert "🔍" in content, "Debug emoji not found in log file (debug mode should be enabled)"
    
    print(f"✓ Debug file logging test passed - Log file: {latest_log} ({os.path.getsize(latest_log)} bytes)")
    
    # Clean up
    test_logger.handlers.clear()


def test_log_file_format():
    """Test that log file has proper formatting."""
    from core.logger import setup_logger
    
    # Clean up any existing handlers
    logger_name = "test_format_logger"
    test_logger = logging.getLogger(logger_name)
    test_logger.handlers.clear()
    
    # Create logger
    test_logger = setup_logger(name=logger_name, log_to_file=True)
    
    # Write different log levels
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    # Find the log file
    log_files = glob.glob('logs/bearish_alpha_bot_*.log')
    latest_log = max(log_files, key=os.path.getctime)
    
    # Read log file
    with open(latest_log, 'r') as f:
        content = f.read()
    
    # Verify format: timestamp - name - level - message
    assert "test_format_logger" in content, "Logger name not in log"
    assert "INFO" in content, "INFO level not in log"
    assert "WARNING" in content, "WARNING level not in log"
    assert "ERROR" in content, "ERROR level not in log"
    
    # Verify timestamp format (YYYY-MM-DD HH:MM:SS)
    import re
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    assert re.search(timestamp_pattern, content), "Timestamp format not found in log"
    
    print("✓ Log format test passed")
    
    # Clean up
    test_logger.handlers.clear()


def test_multiple_loggers_same_file():
    """Test that multiple logger instances can write to files."""
    from core.logger import setup_logger
    import time
    
    # Clean up
    logger1 = logging.getLogger("test_logger_1_unique")
    logger2 = logging.getLogger("test_logger_2_unique")
    logger1.handlers.clear()
    logger2.handlers.clear()
    
    # Create first logger
    logger1 = setup_logger(name="test_logger_1_unique", log_to_file=True)
    logger1.info("Message from logger 1")
    
    # Small delay to ensure different timestamp
    time.sleep(1.1)
    
    # Create second logger
    logger2 = setup_logger(name="test_logger_2_unique", log_to_file=True)
    logger2.info("Message from logger 2")
    
    # Find all log files
    log_files = glob.glob('logs/bearish_alpha_bot_*.log')
    
    # Should have at least some log files
    assert len(log_files) >= 1, f"Expected at least 1 log file, got {len(log_files)}"
    
    # Verify messages are logged
    all_log_content = ""
    for log_file in log_files:
        with open(log_file, 'r') as f:
            all_log_content += f.read()
    
    # At least one of the messages should be in the logs
    has_message = "Message from logger 1" in all_log_content or "Message from logger 2" in all_log_content
    assert has_message, "Neither logger message found in log files"
    
    print(f"✓ Multiple loggers test passed - Found {len(log_files)} log file(s)")
    
    # Clean up
    logger1.handlers.clear()
    logger2.handlers.clear()


if __name__ == '__main__':
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print("Running file logging tests...\n")
    test_logger_creates_file()
    test_logger_without_file()
    test_debug_logger_creates_file()
    test_log_file_format()
    test_multiple_loggers_same_file()
    print("\n✓ All file logging tests passed!")
