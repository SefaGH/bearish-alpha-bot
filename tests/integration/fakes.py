"""Test doubles for integration tests.

These lightweight implementations provide the minimal behaviour used by
``LiveTradingLauncher`` so the integration suite can exercise the launcher
control-flow without bringing up the real production trading stack.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from types import ModuleType
from unittest.mock import MagicMock


class FakeRiskManager:
    """Minimal risk manager that returns a static portfolio summary."""

    def __init__(self) -> None:
        self._summary: Dict[str, Any] = {
            "portfolio_value": 1000.0,
            "positions": [],
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        return dict(self._summary)


class FakePortfolioManager:
    """Simple container for registered strategies."""

    def __init__(self) -> None:
        self.strategies: Dict[str, Dict[str, Any]] = {}


class FakeCircuitBreaker:
    """Placeholder circuit breaker used by the launcher checks."""

    active: bool = True


class FakeProductionCoordinator:
    """Async test double for the production coordinator."""

    def __init__(self) -> None:
        self.risk_manager = FakeRiskManager()
        self.portfolio_manager = FakePortfolioManager()
        self.circuit_breaker = FakeCircuitBreaker()
        self.websocket_manager = None
        self.active_symbols: List[str] = []
        self.is_initialized: bool = False
        self._running: bool = False

    async def initialize_production_system(
        self,
        exchange_clients: Dict[str, Any],
        portfolio_config: Dict[str, Any],
        mode: str,
        trading_symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.is_initialized = True
        self.active_symbols = list(trading_symbols or [])
        components = [
            "exchange_clients",
            "portfolio_manager",
            "risk_manager",
        ]
        if self.websocket_manager:
            components.append("websocket_manager")
        return {"success": True, "components": components}

    def register_strategy(
        self,
        strategy_name: str,
        strategy_instance: Any,
        initial_allocation: float,
    ) -> Dict[str, Any]:
        self.portfolio_manager.strategies[strategy_name] = {
            "instance": strategy_instance,
            "allocation": initial_allocation,
        }
        return {"status": "success"}

    def get_system_state(self) -> Dict[str, Any]:
        return {
            "is_initialized": self.is_initialized,
            "status": "running" if self._running else "stopped",
            "active_symbols": list(self.active_symbols),
        }

    async def run_production_loop(
        self,
        mode: str,
        duration: Optional[float] = None,
        continuous: bool = False,
    ) -> None:
        self._running = True
        try:
            sleep_for = 0.1
            if duration:
                sleep_for = min(duration, 1.0)
            await asyncio.sleep(sleep_for)
        finally:
            self._running = False

    async def stop_system(self) -> Dict[str, Any]:
        self._running = False
        await asyncio.sleep(0)
        return {"status": "stopped"}


class FakeWebSocketManager:
    """Simplified WebSocket manager used by the fake optimizer."""

    def __init__(self) -> None:
        self.clients: Dict[str, Any] = {}

    async def stream_ohlcv(self, *args: Any, **kwargs: Any) -> List[Any]:
        await asyncio.sleep(0)
        return []


class FakeOptimizedWebSocketManager:
    """Test double for ``OptimizedWebSocketManager``."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.ws_manager = FakeWebSocketManager()
        self.is_initialized = False
        self._connection_status: Dict[str, Any] = {
            "connected": False,
            "connecting": False,
            "error": None,
            "exchanges": {},
        }

    def setup_from_config(self, config: Optional[Dict[str, Any]]) -> None:
        self.config = config or {}
        universe = (self.config.get("universe") or {})
        fixed_symbols = list(universe.get("fixed_symbols", []))
        if fixed_symbols:
            self.ws_manager.clients = {symbol: object() for symbol in fixed_symbols}
        self.is_initialized = True

    async def initialize_websockets(self, exchange_clients: Dict[str, Any]) -> List[Any]:
        self.is_initialized = True
        self._connection_status.update(
            {"connecting": False, "connected": True, "error": None}
        )
        if not self.ws_manager.clients:
            self.ws_manager.clients = {"bingx": object()}
        return [object()]

    async def get_stream_status(self) -> Dict[str, Any]:
        active_streams = len(self.ws_manager.clients)
        return {
            "active_streams": active_streams,
            "status": "running" if active_streams else "stopped",
        }

    def get_connection_status(self) -> Dict[str, Any]:
        return dict(self._connection_status)

    async def stop_streaming(self) -> List[Any]:
        self._connection_status["connected"] = False
        return []

    async def shutdown(self) -> None:
        self._connection_status["connected"] = False


def _make_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _RiskConfiguration:
    def __init__(self, custom_limits: Optional[Dict[str, Any]] = None) -> None:
        self.custom_limits = custom_limits or {}


class _OptimizationConfiguration:
    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "strategies": {},
            "websocket": {},
        }


class _LiveTradingConfiguration:
    @staticmethod
    def load(log_summary: bool = False) -> Dict[str, Any]:
        return {
            "universe": {"fixed_symbols": ["BTC/USDT:USDT"]},
            "websocket": {},
            "strategies": {},
        }


class _SystemInfoCollector:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        return {}


def _format_startup_header(**_: Any) -> str:
    return "STARTUP SUMMARY"


class _WebSocketManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.clients: Dict[str, Any] = {}


def build_launcher_module_stubs() -> Dict[str, ModuleType]:
    """Return lightweight module stubs required by ``LiveTradingLauncher``."""

    return {
        "core.ccxt_client": _make_module("core.ccxt_client", CcxtClient=MagicMock()),
        "core.notify": _make_module("core.notify", Telegram=MagicMock()),
        "core.production_coordinator": _make_module(
            "core.production_coordinator",
            ProductionCoordinator=FakeProductionCoordinator,
        ),
        "core.state": _make_module(
            "core.state",
            load_state=lambda *args, **kwargs: {},
            save_state=lambda *args, **kwargs: None,
        ),
        "core.market_regime": _make_module("core.market_regime", MarketRegimeAnalyzer=MagicMock()),
        "core.debug_logger": _make_module("core.debug_logger", DebugLogger=MagicMock()),
        "core.system_info": _make_module(
            "core.system_info",
            SystemInfoCollector=_SystemInfoCollector,
            format_startup_header=_format_startup_header,
        ),
        "core.logger": _make_module("core.logger", setup_logger=lambda *args, **kwargs: MagicMock()),
        "core.websocket_manager": _make_module("core.websocket_manager", WebSocketManager=_WebSocketManager),
        "config.risk_config": _make_module("config.risk_config", RiskConfiguration=_RiskConfiguration),
        "config.optimization_config": _make_module(
            "config.optimization_config",
            OptimizationConfiguration=_OptimizationConfiguration,
        ),
        "config.live_trading_config": _make_module(
            "config.live_trading_config",
            LiveTradingConfiguration=_LiveTradingConfiguration,
        ),
        "ml.regime_predictor": _make_module("ml.regime_predictor", MLRegimePredictor=MagicMock()),
        "ml.price_predictor": _make_module(
            "ml.price_predictor",
            AdvancedPricePredictionEngine=MagicMock(),
            MultiTimeframePricePredictor=MagicMock(),
            EnsemblePricePredictor=MagicMock(),
        ),
        "ml.strategy_integration": _make_module("ml.strategy_integration", AIEnhancedStrategyAdapter=MagicMock()),
        "ml.strategy_optimizer": _make_module("ml.strategy_optimizer", StrategyOptimizer=MagicMock()),
        "strategies.adaptive_ob": _make_module("strategies.adaptive_ob", AdaptiveOversoldBounce=MagicMock()),
        "strategies.adaptive_str": _make_module("strategies.adaptive_str", AdaptiveShortTheRip=MagicMock()),
    }


__all__ = [
    "FakeProductionCoordinator",
    "FakeOptimizedWebSocketManager",
    "build_launcher_module_stubs",
]
