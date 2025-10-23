"""Logging configuration for Bearish Alpha Bot."""
import logging
import sys
import os
from datetime import datetime, timezone

def setup_logger(name: str = "bearish_alpha_bot", level: str = None, log_to_file: bool = True) -> logging.Logger:
    """
    Set up a configured logger for the bot.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, reads from LOG_LEVEL env var, defaults to INFO
        log_to_file: Whether to also log to a file (default: True)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Resolve desired numeric level once
    log_level = getattr(logging, level, logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if requested
    if log_to_file:
        # Ensure logs directory exists
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'bearish_alpha_bot_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"File logging enabled: {log_file}")

    # Prevent log duplication once root logger is configured
    logger.propagate = False

    # Ensure root logger forwards logs for modules that don't call setup_logger
    root_logger = logging.getLogger()

    # Always keep the root at least as verbose as the requested level
    root_logger.setLevel(min(root_logger.level or log_level, log_level))

    # Attach console handler to root if not already present
    if not any(getattr(h, "_bearish_handler", None) == "console" for h in root_logger.handlers):
        root_console = logging.StreamHandler(sys.stdout)
        root_console.setLevel(log_level)
        root_console.setFormatter(formatter)
        root_console._bearish_handler = "console"
        root_logger.addHandler(root_console)

    # Attach file handler to root when requested so other modules log to disk too
    if log_to_file and not any(getattr(h, "_bearish_handler", None) == "file" for h in root_logger.handlers):
        root_file = logging.FileHandler(log_file, mode='w')
        root_file.setLevel(log_level)
        root_file.setFormatter(formatter)
        root_file._bearish_handler = "file"
        root_logger.addHandler(root_file)

    return logger
