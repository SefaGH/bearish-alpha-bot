"""Shared helpers for instantiating the live trading launcher in tests."""

from __future__ import annotations

import os
import sys
from typing import Any
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"

for _path in (_SRC, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("SKIP_PYTHON_VERSION_CHECK", "1")

from live_trading_launcher import LiveTradingLauncher  # noqa: E402

_DEFAULT_LAUNCHER_KWARGS: dict[str, Any] = {
    "mode": "paper",
    "dry_run": True,
    "infinite": False,
    "auto_restart": False,
    "max_restarts": 0,
    "restart_delay": 0,
    "debug_mode": False,
}


def create_launcher(**overrides: Any) -> LiveTradingLauncher:
    """Factory for tests to build a `LiveTradingLauncher` with safe defaults."""
    params = dict(_DEFAULT_LAUNCHER_KWARGS)
    params.update(overrides)
    return LiveTradingLauncher(**params)
