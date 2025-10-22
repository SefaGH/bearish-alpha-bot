"""Core package bootstrap for Bearish Alpha Bot.

This module exposes a ``__getattr__`` hook so that tests (and other
callers) can import submodules using attribute access, e.g.
``core.ccxt_client``.  The integration tests rely on
``unittest.mock.patch`` resolving dotted names such as
``core.ccxt_client.CcxtClient``; without the attribute hook Python raises
``AttributeError`` before the patch machinery can import the submodule.

The lazy loader keeps import time low while still providing the expected
namespace behaviour.  When optional production dependencies are missing
(such as ``ccxt`` or ``requests`` inside ``core.ccxt_client`` and
``core.notify``) we install lightweight stub modules so tests can patch
their classes during offline execution.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

__all__ = ["__getattr__"]


_loaded_submodules: Dict[str, ModuleType] = {}


def _install_stub(module_name: str, symbol: str) -> ModuleType:
    stub = ModuleType(module_name)

    if symbol == "ccxt_client":
        class _Unavailable:
            def __init__(self, *_, **__):  # pragma: no cover - defensive stub
                raise RuntimeError(
                    "core.ccxt_client dependencies are unavailable in this "
                    "environment. The class is replaced with a stub so tests "
                    "can patch it."
                )

        stub.CcxtClient = _Unavailable
    elif symbol == "notify":
        class _Notifier:
            def __init__(self, *_, **__):  # pragma: no cover - defensive stub
                raise RuntimeError(
                    "core.notify dependencies are unavailable in this "
                    "environment. The class is replaced with a stub so tests "
                    "can patch it."
                )

            def send(self, *_args, **_kwargs) -> None:  # pragma: no cover
                raise RuntimeError("Telegram stub cannot send messages")

        stub.Telegram = _Notifier
    else:  # pragma: no cover - should not occur in tests
        raise AttributeError(f"module '{module_name}' is unavailable")

    return stub


def __getattr__(name: str) -> ModuleType:
    if name in _loaded_submodules:
        return _loaded_submodules[name]
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - mirrors Python's behaviour
        if name in {"ccxt_client", "notify"}:
            module = _install_stub(f"{__name__}.{name}", name)
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    _loaded_submodules[name] = module
    return module
