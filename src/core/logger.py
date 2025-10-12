"""Logging configuration for Bearish Alpha Bot."""
import logging
import sys
import os
from datetime import datetime, timezone

def setup_logger(name: str = "bearish_alpha_bot", level: str = None) -> logging.Logger:
    """
    Set up a configured logger for the bot.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, reads from LOG_LEVEL env var, defaults to INFO
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
