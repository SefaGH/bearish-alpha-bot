"""Debug logging utilities built atop the standard logging module.

This module provides a convenient wrapper around the project's logging stack
for situations where verbose debug output (with emoji prefixes) is desirable.
The implementation intentionally keeps its own handlers so it can be used in
isolation during tests without mutating the queue-based production logger.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

__all__ = ["DebugLogger", "setup_debug_logger"]


def _build_formatter(debug_mode: bool) -> logging.Formatter:
    if debug_mode:
        fmt = "%(asctime)s - DEBUG [%(name)s] - %(levelname)s - 🔍 %(message)s"
    else:
        fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _ensure_log_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _create_file_handler(log_dir: str, filename: Optional[str], formatter: logging.Formatter) -> logging.Handler:
    _ensure_log_directory(log_dir)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"bearish_alpha_bot_debug_{timestamp}.log"
    file_path = os.path.join(log_dir, filename)
    handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    return handler


def setup_debug_logger(
    name: str = "bearish_alpha_bot_debug",
    *,
    debug_mode: bool = False,
    log_to_file: bool = True,
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """Return a logger that inserts a debug emoji when debug mode is active.

    Parameters
    ----------
    name: str
        Name of the logger instance to return.
    debug_mode: bool
        When True the logger operates at DEBUG level and prefixes messages
        with a magnifying glass emoji ("🔍").
    log_to_file: bool
        Controls whether a UTF-8 file handler should be created. Files are
        written under the provided log_dir using the
        "bearish_alpha_bot_debug_YYYYMMDD_HHMMSS_mmmmmm.log" pattern unless a
        log_filename override is supplied.
    log_dir: str
        Directory for log files when log_to_file is True.
    log_filename: Optional[str]
        Custom filename for the log file. Useful for tests that want a stable
        name.
    """

    level = logging.DEBUG if debug_mode else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = _build_formatter(debug_mode)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        file_handler = _create_file_handler(log_dir, log_filename, formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass(slots=True)
class DebugLogger:
    """Helper that exposes an is_debug_enabled predicate and delegates methods."""

    debug_mode: bool = False
    name: str = "bearish_alpha_bot_debug"
    log_to_file: bool = True
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = setup_debug_logger(
            name=self.name,
            debug_mode=self.debug_mode,
            log_to_file=self.log_to_file,
        )

    def __getattr__(self, item):
        return getattr(self._logger, item)

    def is_debug_enabled(self) -> bool:
        return self.debug_mode

    def get_logger(self) -> logging.Logger:
        return self._logger