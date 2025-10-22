"""Minimal ccxt stub used for offline integration tests."""
from __future__ import annotations

from typing import Any


class _DummyExchange:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple stub
        raise AttributeError(name)


def __getattr__(name: str) -> type[_DummyExchange]:
    return _DummyExchange


__all__ = ["_DummyExchange"]
