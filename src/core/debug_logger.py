"""
Enhanced Debug Logger for Live Trading Analysis.

Provides comprehensive debug logging with structured output for:
- Strategy signal generation
- AI decision reasoning
- Market data flow
- Risk management calculations
- Order execution analysis
- Circuit breaker monitoring
"""

import logging
import sys
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DebugLogger:
    """Enhanced logging for debug mode with structured output."""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize debug logger.
        
        Args:
            debug_mode: Enable debug mode logging
        """
        self.debug_mode = debug_mode
        self.setup_debug_logging()
    
    def setup_debug_logging(self):
        """Configure logging system for debug mode."""
        if self.debug_mode:
            # Set debug level for root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            
            # Update all existing handlers to DEBUG level
            for handler in root_logger.handlers:
                handler.setLevel(logging.DEBUG)
                
                # Update formatter to include debug emoji
                formatter = logging.Formatter(
                    '%(asctime)s - 🔍 %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
            
            # Also set debug level for common component loggers
            component_loggers = [
                'strategies',
                'ml',
                'core',
                'adaptive_ob',
                'adaptive_str',
                'regime_predictor',
                'price_predictor',
                'strategy_integration',
                'strategy_optimizer',
                'order_manager',
                'risk_manager',
                'circuit_breaker',
                'production_coordinator',
            ]
            
            for component_name in component_loggers:
                component_logger = logging.getLogger(component_name)
                component_logger.setLevel(logging.DEBUG)
            
            logger.info("🔍 DEBUG MODE: Enhanced logging enabled")
            logger.info("🔍 Monitoring: Strategy signals, AI decisions, Risk calculations, Order execution")
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode


def setup_debug_logger(name: str = "bearish_alpha_bot", debug_mode: bool = False, log_to_file: bool = True) -> logging.Logger:
    """
    Set up a configured logger with optional debug mode.
    
    Args:
        name: Logger name
        debug_mode: Enable debug mode with detailed logging
        log_to_file: Whether to also log to a file (default: True)
    
    Returns:
        Configured logger instance
    """
    # Determine log level
    if debug_mode:
        level = 'DEBUG'
    else:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid duplicate handlers
    if log.handlers:
        return log
    
    # Create formatter (with debug emoji if debug mode)
    if debug_mode:
        formatter = logging.Formatter(
            '%(asctime)s - 🔍 %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level, logging.INFO))
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        # Ensure logs directory exists
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'bearish_alpha_bot_debug_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        
        logger.info(f"Debug file logging enabled: {log_file}")
    
    return log
