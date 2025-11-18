"""Test doubles for integration tests.

These lightweight implementations provide the minimal behaviour used by
``LiveTradingLauncher`` so the integration suite can exercise the launcher
control-flow without bringing up the real production trading stack.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from contextlib import contextmanager, ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
import sys


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
    """Async test double for the production coordinator with real task flow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.risk_manager = FakeRiskManager()
        self.portfolio_manager = FakePortfolioManager()
        self.circuit_breaker = FakeCircuitBreaker()
        self.websocket_manager = None
        self.active_symbols: List[str] = []
        self.is_initialized: bool = False
        self._running: bool = False
        self._background_tasks: set[asyncio.Task] = set()
        self._loop_start: float | None = None
        self.runtime_seconds: float = 0.0
        self.spawned_task_count: int = 0
        self.strategy_cycles: int = 0
        self.trading_engine = SimpleNamespace()
        self.config = config or {}

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

    async def initialize_core_systems(
        self,
        exchange_clients: Dict[str, Any],
        portfolio_value: float,
        risk_config: Any,
        mode: str,
        trading_symbols: Optional[List[str]] = None,
        websocket_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """New initializer path used by launcher phase 1."""

        self.is_initialized = True
        self.exchange_clients = dict(exchange_clients)
        self.active_symbols = list(trading_symbols or [])
        self.websocket_manager = websocket_manager
        self.portfolio_value = portfolio_value
        self.mode = mode
        self.risk_config = risk_config

        components = [
            "exchange_clients",
            "portfolio_manager",
            "risk_manager",
        ]
        if websocket_manager is not None:
            components.append("websocket_manager")

        return {"success": True, "components": components}

    async def is_data_layer_healthy(self) -> Dict[str, Any]:
        checks = {
            "websocket_connection": {
                "status": "healthy",
                "details": "Synthetic WebSocket connection active",
            },
            "subscriptions": {
                "status": "healthy",
                "details": f"{len(self.active_symbols)} symbols subscribed",
            },
            "data_flow": {
                "status": "healthy",
                "details": "Synthetic data stream producing payloads",
            },
        }
        return {"healthy": True, "checks": checks}

    async def initialize_ml_systems(self, price_engine: Any = None, regime_predictor: Any = None) -> Dict[str, Any]:
        self.ml_initialized = True
        return {
            "success": True,
            "components": [
                comp
                for comp in ("price_engine" if price_engine else None, "regime_predictor" if regime_predictor else None)
                if comp
            ],
        }

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
        loop = asyncio.get_running_loop()
        self._loop_start = loop.time()
        target = self._loop_start + (duration or 5.0)

        try:
            while loop.time() < target:
                task = loop.create_task(self._simulate_strategy_cycle())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                self.spawned_task_count += 1

                # Sleep until next cycle but do not overshoot the target
                remaining = target - loop.time()
                await asyncio.sleep(min(1.0, max(remaining, 0)))

            if self._background_tasks:
                await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        finally:
            self.runtime_seconds = loop.time() - (self._loop_start or loop.time())
            self._running = False

    async def stop_system(self) -> Dict[str, Any]:
        self._running = False
        for task in list(self._background_tasks):
            task.cancel()
        await asyncio.sleep(0)
        return {"status": "stopped"}

    async def _simulate_strategy_cycle(self) -> None:
        await asyncio.sleep(0.5)
        self.strategy_cycles += 1


class FakeStreamCollector:
    """Minimal collector mimicking StreamDataCollector behaviour."""

    def __init__(self) -> None:
        self._ohlcv: Dict[tuple[str, str, str], List[List[float]]] = {}

    def record_candle(self, exchange: str, symbol: str, timeframe: str, candle: List[float]) -> None:
        key = (exchange, symbol, timeframe)
        bucket = self._ohlcv.setdefault(key, [])
        bucket.append(list(candle))
        if len(bucket) > 500:
            del bucket[:-500]

    def get_latest_ohlcv(self, exchange: str, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        bucket = self._ohlcv.get((exchange, symbol, timeframe), [])
        if limit <= 0:
            return list(bucket)
        return list(bucket[-limit:])

    def get_latest_data(self, exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        return {"ohlcv": self.get_latest_ohlcv(exchange, symbol, timeframe, limit=250)}


class FakeWebSocketManager:
    """Simplified WebSocket manager used by the fake optimizer."""

    def __init__(self) -> None:
        self.clients: Dict[str, Any] = {}
        self._stream_tasks: set[asyncio.Task] = set()
        self.message_counts: Dict[str, int] = {}
        self.collector = FakeStreamCollector()

    def start_ohlcv_stream(self, exchange: str, symbol: str, timeframe: str) -> bool:
        stream_id = f"{exchange}:{symbol}:{timeframe}"
        if stream_id in self.message_counts:
            return True

        self.message_counts[stream_id] = 0
        return True

    async def stop_streams(self) -> None:
        for task in list(self._stream_tasks):
            task.cancel()
        await asyncio.sleep(0)

    def get_active_stream_count(self) -> int:
        return sum(1 for count in self.message_counts.values() if count > 0)

    def get_latest_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        return self.collector.get_latest_data("bingx", symbol, timeframe)

    def is_any_client_connected(self) -> bool:
        return bool(self.clients)


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
        self._stream_tasks: set[asyncio.Task] = set()
        self.message_log: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_calls: List[List[str]] = []

    def setup_from_config(self, config: Optional[Dict[str, Any]]) -> None:
        self.config = config or {}
        universe = (self.config.get("universe") or {})
        fixed_symbols = list(universe.get("fixed_symbols", []))
        self.is_initialized = True

    def _parse_stream_timeframes(self) -> List[str]:
        websocket_cfg = (self.config.get("websocket") or {})
        timeframes = websocket_cfg.get("stream_timeframes") or ["1m"]
        return list(timeframes)

    def _normalize_ccxt_futures_symbols(self, symbols: List[str]) -> List[str]:
        return list(symbols)

    async def initialize_websockets(self, exchange_clients: Dict[str, Any]) -> List[Any]:
        self.is_initialized = True
        self._connection_status.update(
            {"connecting": False, "connected": True, "error": None}
        )
        if exchange_clients:
            self.ws_manager.clients = dict(exchange_clients)
        elif not self.ws_manager.clients:
            self.ws_manager.clients = {"bingx": object()}

        universe = self.config.get("universe", {})
        symbols = universe.get("fixed_symbols", ["BTC/USDT:USDT"])
        timeframes = (self.config.get("websocket", {}) or {}).get("stream_timeframes", ["1m"])

        loop = asyncio.get_running_loop()

        async def _generate_stream(exchange: str, symbol: str, timeframe: str) -> None:
            stream_id = f"{exchange}:{symbol}:{timeframe}"
            self.message_log.setdefault(stream_id, [])
            for i in range(10):
                await asyncio.sleep(0.5)
                payload = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "sequence": i,
                }
                self.ws_manager.message_counts[stream_id] = self.ws_manager.message_counts.get(stream_id, 0) + 1
                base_price = 50000.0 + i
                candle = [float(i), base_price, base_price + 5.0, base_price - 5.0, base_price + 1.0, 100.0 + i]
                self.ws_manager.collector.record_candle(exchange, symbol, timeframe, candle)
                self.message_log[stream_id].append(payload)

        for exchange in exchange_clients or {"bingx": object()}:
            for symbol in symbols:
                for timeframe in timeframes:
                    self.ws_manager.start_ohlcv_stream(exchange, symbol, timeframe)
                    stream_id = f"{exchange}:{symbol}:{timeframe}"
                    self.message_log.setdefault(stream_id, [])
                    # Seed collector with immediate data so preflight checks see activity.
                    initial_candle = [0.0, 50000.0, 50005.0, 49995.0, 50001.0, 100.0]
                    self.ws_manager.collector.record_candle(exchange, symbol, timeframe, initial_candle)
                    self.ws_manager.message_counts[stream_id] = self.ws_manager.message_counts.get(stream_id, 0) + 1
                    self.message_log[stream_id].append(
                        {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "sequence": -1,
                        }
                    )
                    for seed_idx in range(1, 4):
                        base_price = 50000.0 + seed_idx
                        candle = [float(seed_idx), base_price, base_price + 5.0, base_price - 5.0, base_price + 1.0, 100.0 + seed_idx]
                        self.ws_manager.collector.record_candle(exchange, symbol, timeframe, candle)
                        self.ws_manager.message_counts[stream_id] += 1
                        self.message_log[stream_id].append(
                            {
                                "exchange": exchange,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "sequence": -1 - seed_idx,
                            }
                        )
                    task = loop.create_task(_generate_stream(exchange, symbol, timeframe))
                    self._stream_tasks.add(task)
                    task.add_done_callback(self._stream_tasks.discard)

        return list(self._stream_tasks)

    async def initialize_and_subscribe(
        self,
        exchange_clients: Dict[str, Any],
        symbols: List[str],
    ) -> bool:
        """Mimic production initializer: configure and launch synthetic streams."""

        # Align config with requested symbols so launcher diagnostics stay consistent.
        if symbols:
            normalized = list(symbols)
            self.config.setdefault("universe", {})["fixed_symbols"] = normalized
            self._last_symbols = list(normalized)

        self._initialize_calls.append(list(symbols))

        self.setup_from_config(self.config)

        tasks = await self.initialize_websockets(exchange_clients)
        success = bool(tasks)
        self._last_initialize_success = success
        if not success:
            return False

        self.is_initialized = True
        self._connection_status.update({"connected": True, "connecting": False, "error": None})
        return True

    async def get_stream_status(self) -> Dict[str, Any]:
        active_streams = len(self.ws_manager.clients)
        total_messages = sum(self.ws_manager.message_counts.values())
        return {
            "active_streams": active_streams,
            "status": "running" if active_streams else "stopped",
            "messages": total_messages,
        }

    def get_connection_status(self) -> Dict[str, Any]:
        return dict(self._connection_status)

    async def stop_streaming(self) -> List[Any]:
        self._connection_status["connected"] = False
        for task in list(self._stream_tasks):
            task.cancel()
        await self.ws_manager.stop_streams()
        return []

    async def shutdown(self) -> None:
        self._connection_status["connected"] = False
        await self.stop_streaming()


def _make_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _RiskConfiguration:
    def __init__(self, custom_limits: Optional[Dict[str, Any]] = None) -> None:
        limits = custom_limits or {}
        capital = float(limits.get('equity_usd', 100.0))

        self.custom_limits = limits
        self.initial_capital = capital
        self.risk_limits = SimpleNamespace(
            max_portfolio_risk=limits.get('max_portfolio_risk', 0.02),
            max_position_size=limits.get('max_position_size', 0.10),
            max_drawdown=limits.get('max_drawdown', 0.15),
        )
        self.circuit_breaker_limits = SimpleNamespace(
            daily_loss_limit=limits.get('daily_loss_limit', 0.05),
        )
        self.max_risk_per_trade_usd = capital * self.risk_limits.max_portfolio_risk
        self.daily_loss_limit_usd = capital * self.circuit_breaker_limits.daily_loss_limit
        self.circuit_breaker_limits_usd = {
            'daily_loss_limit': self.daily_loss_limit_usd,
        }

    def get_risk_limits(self) -> SimpleNamespace:
        return self.risk_limits

    def get_circuit_breaker_limits(self) -> SimpleNamespace:
        return self.circuit_breaker_limits


class _OptimizationConfiguration:
    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "strategies": {},
            "websocket": {},
        }


class _LiveTradingConfiguration:
    @staticmethod
    def load(log_summary: bool = False, **_: Any) -> Dict[str, Any]:
        return {
            "universe": {"fixed_symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"]},
            "websocket": {
                "stream_timeframes": ["1m", "5m"],
            },
            "risk": {
                "max_position_size": 0.2,
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.02,
                "min_stop_pct": 0.005,
                "max_drawdown": 0.05,
                "max_notional_pct": 0.1,
                "max_margin_pct": 0.1,
            },
            "signals": {
                "oversold_bounce": {"enable": True},
                "short_the_rip": {"enable": True},
            },
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


class _IndicatorValidator:
    def __init__(self, collector: Any, rest_client: Any = None) -> None:
        self.collector = collector
        self.rest_client = rest_client

    async def validate_all(self, symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, str]]:
        return {symbol: {"status": "OK", "reason": "synthetic data ready"} for symbol in symbols}


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
        "core.indicator_validator": _make_module("core.indicator_validator", IndicatorValidator=_IndicatorValidator),
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


@contextmanager
def ignore_test_task_cancellation(task: asyncio.Task) -> None:
    """Temporarily filter ``asyncio.all_tasks`` to exclude the test coroutine.

    ``LiveTradingLauncher.cleanup`` cancels every pending task on the event loop,
    which includes the coroutine executing the integration test.  When that test
    coroutine is cancelled ``asyncio.wait_for`` raises ``CancelledError`` even
    though the launcher exited cleanly.  This context manager hides the provided
    test task from ``asyncio.all_tasks`` during cleanup so the launcher can
    cancel its own internal tasks without touching the test harness.
    """

    original_all_tasks = asyncio.all_tasks

    def _filtered_all_tasks(loop: asyncio.AbstractEventLoop | None = None):
        tasks = original_all_tasks(loop)
        return [t for t in tasks if t is not task]

    with ExitStack() as stack:
        stack.enter_context(patch.object(asyncio, "all_tasks", _filtered_all_tasks))

        launcher_module = sys.modules.get("live_trading_launcher")
        if launcher_module is not None and hasattr(launcher_module, "asyncio"):
            stack.enter_context(
                patch.object(launcher_module.asyncio, "all_tasks", _filtered_all_tasks)
            )

        yield


__all__ = [
    "FakeProductionCoordinator",
    "FakeOptimizedWebSocketManager",
    "build_launcher_module_stubs",
    "ignore_test_task_cancellation",
]
