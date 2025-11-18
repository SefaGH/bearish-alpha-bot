"""Tests for the queue-based logging helpers in src.core.logger."""

import io
import logging
import logging.handlers
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import logger as core_logger


@pytest.fixture(autouse=True)
def isolate_stdout(monkeypatch) -> Iterator[None]:
    """Provide a simple stdout target so pytest capture isn't disrupted."""
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    fake_sys = SimpleNamespace(stdout=fake_stdout, stderr=fake_stderr)
    monkeypatch.setattr(core_logger, "sys", fake_sys)
    yield


@pytest.fixture(autouse=True)
def reset_logger_state() -> Iterator[None]:
    """Ensure each test executes with a fresh logging configuration."""
    listener = getattr(core_logger, "_listener", None)
    if listener:
        listener.stop()
    core_logger._listener = None
    logging.getLogger().handlers.clear()
    yield
    listener = getattr(core_logger, "_listener", None)
    if listener:
        listener.stop()
    core_logger._listener = None
    logging.getLogger().handlers.clear()


def test_setup_logger_debug_mode_sets_root_level() -> None:
    logger = core_logger.setup_logger("test_debug", debug_mode=True, log_to_file=False)
    assert logger.getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logger_custom_level_overrides_debug_mode() -> None:
    logger = core_logger.setup_logger(
        "test_info",
        debug_mode=True,
        level=logging.INFO,
        log_to_file=False,
    )
    assert logger.getEffectiveLevel() == logging.INFO
    assert logging.getLogger().level == logging.INFO


def test_setup_logger_installs_single_queue_handler() -> None:
    core_logger.setup_logger("first", log_to_file=False)
    core_logger.setup_logger("second", log_to_file=False)

    root_handlers = logging.getLogger().handlers
    queue_handlers = [h for h in root_handlers if isinstance(h, logging.handlers.QueueHandler)]

    assert len(queue_handlers) == 1


def test_setup_logger_returns_named_logger() -> None:
    name = "custom.logger"
    logger = core_logger.setup_logger(name, log_to_file=False)
    assert logger.name == name


def test_setup_logger_respects_log_to_file_flag() -> None:
    core_logger.setup_logger("no_file", log_to_file=False)
    assert all(
        not isinstance(handler, logging.FileHandler)
        for handler in logging.getLogger().handlers
    )
