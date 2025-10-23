"""Logging configuration for Bearish Alpha Bot."""
import logging
import sys
import os
from datetime import datetime
from typing import Set


MANAGED_ROOT_LOG_FILES: Set[str] = set()

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
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ensure console handler exists and is configured
    console_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) == sys.stdout
    ]
    if console_handlers:
        for handler in console_handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent log duplication once root logger is configured
    logger.propagate = False

    # Create file handler if requested
    log_file = None
    existing_file_handler = next(
        (h for h in logger.handlers if isinstance(h, logging.FileHandler)),
        None
    )
    if log_to_file:
        if existing_file_handler:
            log_file = existing_file_handler.baseFilename
        else:
            # Ensure logs directory exists
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            log_file = os.path.join(log_dir, f'bearish_alpha_bot_{timestamp}.log')
    elif existing_file_handler:
        log_file = existing_file_handler.baseFilename

    if log_file:
        abs_log_file = os.path.abspath(log_file)
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
            and os.path.abspath(h.baseFilename) == abs_log_file
        ]
        if file_handlers:
            for handler in file_handlers:
                handler.setLevel(log_level)
                handler.setFormatter(formatter)
        else:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file}")

    # Ensure root logger forwards logs for modules that don't call setup_logger
    root_logger = logging.getLogger()

    # Always keep the root at least as verbose as the requested level
    root_logger.setLevel(max(root_logger.level or log_level, log_level))

    # Configure root console handler(s)
    root_console_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == sys.stdout
    ]
    if root_console_handlers:
        for handler in root_console_handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
    else:
        root_console = logging.StreamHandler(sys.stdout)
        root_console.setLevel(log_level)
        root_console.setFormatter(formatter)
        root_logger.addHandler(root_console)

    # Attach file handler to root when requested so other modules log to disk too
    if log_file:
        abs_log_file = os.path.abspath(log_file)
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == abs_log_file:
                handler.setLevel(log_level)
                handler.setFormatter(formatter)
                MANAGED_ROOT_LOG_FILES.add(abs_log_file)
                break
        else:
            root_file = logging.FileHandler(log_file, mode='w')
            root_file.setLevel(log_level)
            root_file.setFormatter(formatter)
            root_logger.addHandler(root_file)
            MANAGED_ROOT_LOG_FILES.add(abs_log_file)

    # Update levels for previously managed root file handlers even when not recreating them
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler_path = os.path.abspath(handler.baseFilename)
            if handler_path in MANAGED_ROOT_LOG_FILES:
                handler.setLevel(log_level)
                handler.setFormatter(formatter)

    return logger
