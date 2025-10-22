#!/usr/bin/env python3
"""
Live Trading Launcher for Bearish Alpha Bot

[... mevcut docstring ...]
"""

import sys
import os

# Check Python version at startup (before any other imports)
# Can be bypassed for testing by setting SKIP_PYTHON_VERSION_CHECK=1
REQUIRED_PYTHON = (3, 11)
if sys.version_info[:2] != REQUIRED_PYTHON and not os.environ.get('SKIP_PYTHON_VERSION_CHECK'):
    raise RuntimeError(
        f"❌ Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} is required!\n"
        f"   Current: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
        f"   Please install Python 3.11 and try again.\n"
        f"   Recommended: Use pyenv to manage Python versions.\n"
        f"   See README.md for installation instructions."
    )

import asyncio
import logging
import argparse
import time
import signal
import yaml  # ← Add this import
import inspect
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Note: Logging configuration is handled by setup_logger() and setup_debug_logger()
# Do not use logging.basicConfig() here as it interferes with file logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient
from core.notify import Telegram
from core.state import load_state, save_state
from core.market_regime import MarketRegimeAnalyzer
from core.debug_logger import DebugLogger
from core.system_info import SystemInfoCollector, format_startup_header
from config.risk_config import RiskConfiguration
from config.optimization_config import OptimizationConfiguration
from ml.regime_predictor import MLRegimePredictor
from ml.price_predictor import (
    AdvancedPricePredictionEngine, 
    MultiTimeframePricePredictor,
    EnsemblePricePredictor
)
from ml.strategy_integration import AIEnhancedStrategyAdapter
from ml.strategy_optimizer import StrategyOptimizer
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip

# Import logger setup from core
from core.logger import setup_logger

# Configure logging with file support
logger = setup_logger(name=__name__, log_to_file=True)


# ============= WebSocket Optimization Manager =============
class OptimizedWebSocketManager:
    """Production-optimized WebSocket Manager for fixed symbol list"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize optimized WebSocket manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.ws_manager = None
        self.fixed_symbols = []
        self.max_streams_config = {}
        self.is_initialized = False
        
        # Connection status tracking
        self._connection_status = {
            'connected': False,
            'connecting': False,
            'error': None,
            'last_check': None,
            'exchanges': {}
        }
        
        logger.info("[WS-OPT] Optimized WebSocket Manager initialized")

        def _coerce_config_types(self, obj):
            """Recursively coerce placeholder type-name strings to safe runtime values.
    
            Two improvements:
             - If the config contains placeholder type names (e.g. 'dict', 'list', 'int'),
               return the actual Python type object (dict, list, int, ...) so downstream
               isinstance(x, SOME_CONFIG_VALUE) works.
             - Still recurse into dict/list/tuple to sanitize nested placeholders.
            """
            if isinstance(obj, dict):
                return {k: self._coerce_config_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self._coerce_config_types(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(self._coerce_config_types(v) for v in obj)
            if isinstance(obj, str):
                lower = obj.strip().lower()
                # Map placeholder names to Python *types* (NOT instances)
                if lower == 'dict':
                    return dict
                if lower == 'list':
                    return list
                if lower == 'tuple':
                    return tuple
                if lower == 'set':
                    return set
                if lower == 'int':
                    return int
                if lower == 'float':
                    return float
                if lower == 'bool':
                    return bool
                if lower == 'str':
                    return str
                # keep other strings as-is
                return obj
            return obj
    
    def setup_from_config(self, config: Dict[str, Any]) -> None:
        """
        Setup WebSocket configuration from config.
        Coerces malformed values into safe defaults and extracts fixed symbols.
        """
        try:
            safe_config = self._coerce_config_types(config or {})
        except Exception:
            # Fallback to a shallow copy if something unexpected occurs
            safe_config = dict(config or {})
    
        universe_cfg = safe_config.get('universe', {}) or {}
        fixed_symbols = universe_cfg.get('fixed_symbols', [])
        if isinstance(fixed_symbols, str):
            fixed_symbols = [fixed_symbols]
        if not isinstance(fixed_symbols, (list, tuple)):
            logger.warning("[WS-OPT] fixed_symbols not list/tuple; coercing to empty list")
            fixed_symbols = []
    
        ws_cfg = safe_config.get('websocket', {}) or {}
        if not isinstance(ws_cfg, dict):
            ws_cfg = {
                'enabled': True,
                'max_streams_per_exchange': {'default': 10}
            }
    
        # Coerce max_streams_per_exchange entries to ints where possible
        max_streams = ws_cfg.get('max_streams_per_exchange', {}) or {}
        if not isinstance(max_streams, dict):
            max_streams = {'default': 10}
    
        coerced_max_streams = {}
        for k, v in list(max_streams.items()):
            try:
                coerced_max_streams[k] = int(v)
            except Exception:
                logger.warning(f"[WS-OPT] Invalid max_streams value for {k}: {v} -> using default 10")
                coerced_max_streams[k] = 10
    
        ws_cfg['max_streams_per_exchange'] = coerced_max_streams
    
        # Assign sanitized values
        safe_config['websocket'] = ws_cfg
        safe_config.setdefault('universe', {})['fixed_symbols'] = list(fixed_symbols)
    
        self.config = safe_config
        self.fixed_symbols = list(fixed_symbols)
        self.max_streams_config = ws_cfg.get('max_streams_per_exchange', {})
    
        logger.info(f"[WS-OPT] Configured with {len(self.fixed_symbols)} fixed symbols")
        if not self.fixed_symbols:
            logger.warning("[WS-OPT] No fixed symbols configured!")
    
        async def initialize_websockets(self, exchange_clients: Dict[str, Any]) -> List[asyncio.Task]:
            """
            Initialize WebSocket connections with optimization.
            Returns empty list on failure without raising TypeError.
            """
            try:
                # Use instance method if available, otherwise use fallback sanitizer
                coerce_fn = getattr(self, '_coerce_config_types', None)
    
                if callable(coerce_fn):
                    safe_config = coerce_fn(self.config or {})
                else:
                    # Fallback coerce function (keeps behavior consistent with expected mapping to types)
                    def _fallback_coerce(obj):
                        if isinstance(obj, dict):
                            return {k: _fallback_coerce(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_fallback_coerce(v) for v in obj]
                        if isinstance(obj, tuple):
                            return tuple(_fallback_coerce(v) for v in obj)
                        if isinstance(obj, str):
                            lower = obj.strip().lower()
                            if lower == 'dict':
                                return dict
                            if lower == 'list':
                                return list
                            if lower == 'tuple':
                                return tuple
                            if lower == 'set':
                                return set
                            if lower == 'int':
                                return int
                            if lower == 'float':
                                return float
                            if lower == 'bool':
                                return bool
                            if lower == 'str':
                                return str
                            return obj
                        return obj
    
                    safe_config = _fallback_coerce(self.config or {})
    
                # Ensure websocket config is dict
                ws_cfg = safe_config.get('websocket', {}) or {}
                if not isinstance(ws_cfg, dict):
                    logger.warning("[WS-OPT] websocket config not a dict, coercing to defaults")
                    ws_cfg = {'enabled': True, 'max_streams_per_exchange': {'default': 10}}
    
                # sanitize max_streams_per_exchange
                max_streams = ws_cfg.get('max_streams_per_exchange', {}) or {}
                if not isinstance(max_streams, dict):
                    logger.warning("[WS-OPT] max_streams_per_exchange invalid; replacing with defaults")
                    max_streams = {'default': 10}
    
                for k, v in list(max_streams.items()):
                    try:
                        max_streams[k] = int(v)
                    except Exception:
                        logger.warning(f"[WS-OPT] Invalid max_streams value for {k}: {v} -> using default 10")
                        max_streams[k] = 10
    
                ws_cfg['max_streams_per_exchange'] = max_streams
                safe_config['websocket'] = ws_cfg
    
                # Ensure universe.fixed_symbols is list
                universe = safe_config.get('universe', {}) or {}
                fixed_syms = universe.get('fixed_symbols', [])
                if isinstance(fixed_syms, str):
                    fixed_syms = [fixed_syms]
                if not isinstance(fixed_syms, (list, tuple)):
                    logger.warning("[WS-OPT] fixed_symbols not list/tuple; coercing to empty list")
                    fixed_syms = []
                safe_config.setdefault('universe', {})['fixed_symbols'] = list(fixed_syms)
    
                # assign sanitized config
                self.config = safe_config
                self.fixed_symbols = list(safe_config['universe']['fixed_symbols'])
                self.max_streams_config = ws_cfg.get('max_streams_per_exchange', {})
    
                if not self.fixed_symbols:
                    logger.warning("[WS-OPT] No fixed symbols, WebSocket disabled")
                    return []
    
                # Import WebSocketManager lazily and protect against TypeError
                try:
                    from core.websocket_manager import WebSocketManager
                except Exception:
                    logger.debug("[WS-OPT] core.websocket_manager not available in test env; skipping WebSocket setup")
                    return []
    
                try:
                    self.ws_manager = WebSocketManager(
                        exchanges=exchange_clients,
                        config=self.config
                    )
                except TypeError as e:
                    try:
                        type_map = {k: type(v).__name__ for k, v in (self.config or {}).items()}
                    except Exception:
                        type_map = str(self.config)
                    logger.error(f"[WS-OPT] WebSocketManager init TypeError: {e}; config types: {type_map}")
                    return []
                except Exception as e:
                    logger.error(f"[WS-OPT] WebSocketManager init failed: {e}")
                    return []
    
            # Setup stream limits per exchange
            for exchange_name in exchange_clients.keys():
                max_streams = self.max_streams_config.get(
                    exchange_name,
                    self.max_streams_config.get('default', 10)
                )
                logger.info(f"[WS-OPT] {exchange_name}: Max streams set to {max_streams}")
    
            tasks = await self._subscribe_optimized()
    
            if tasks:
                logger.info(f"[WS-OPT] ✅ WebSocket initialized with {len(tasks)} streams")
                self.is_initialized = True
                return tasks
            else:
                logger.warning("[WS-OPT] No WebSocket streams started")
                return []
    
        except TypeError as e:
            try:
                type_map = {k: type(v).__name__ for k, v in (self.config or {}).items()}
            except Exception:
                type_map = str(self.config)
            logger.error(f"[WS-OPT] TypeError during WebSocket init: {e} - config types: {type_map}")
            return []
        except Exception as e:
            logger.error(f"[WS-OPT] Failed to initialize WebSocket: {e}")
            return []
    
    async def _subscribe_optimized(self) -> List[asyncio.Task]:
        """
        Subscribe to WebSocket streams with optimization.
        
        Returns:
            List of stream tasks
        """
        if not self.ws_manager:
            return []
        
        tasks = []
        stream_count = {}
        
        for exchange_name, client in self.ws_manager.clients.items():
            max_streams = self.max_streams_config.get(
                exchange_name,
                self.max_streams_config.get('default', 10)
            )
            
            exchange_symbols = []
            for symbol in self.fixed_symbols:
                if len(exchange_symbols) >= max_streams:
                    logger.info(f"[WS-OPT] {exchange_name}: Max streams ({max_streams}) reached")
                    break
                exchange_symbols.append(symbol)
            
            if exchange_symbols:
                # Subscribe to symbols
                symbols_per_exchange = {exchange_name: exchange_symbols}
                
                # OHLCV streams for main timeframe
                ohlcv_tasks = await self.ws_manager.stream_ohlcv(
                    symbols_per_exchange=symbols_per_exchange,
                    timeframe='1m',
                    callback=None,
                    max_iterations=None
                )
                
                tasks.extend(ohlcv_tasks)
                stream_count[exchange_name] = len(exchange_symbols)
                
                logger.info(f"[WS-OPT] {exchange_name}: Subscribed to {len(exchange_symbols)} symbols")
        
        logger.info(f"[WS-OPT] Total streams: {sum(stream_count.values())}")
        return tasks
    
    def _ensure_awaitable(self, maybe_awaitable, coro_callable=None):
        """Return an awaitable for maybe_awaitable. If it's awaitable, return it.
        If it's a synchronous callable (coro_callable provided or a function), run it in a thread.
        """
        if inspect.isawaitable(maybe_awaitable):
            return maybe_awaitable
        if coro_callable:
            # assume coro_callable is a synchronous function to run in a thread
            return asyncio.to_thread(coro_callable)
        # If it's not awaitable and no callable provided, wrap a no-op
        return asyncio.sleep(0)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current WebSocket connection status.
        
        Returns:
            dict: Connection status including:
                - connected: bool (True if any exchange connected)
                - connecting: bool (True if connection in progress)
                - error: str or None (last error message)
                - last_check: float or None (timestamp of last check)
                - exchanges: dict (per-exchange status)
        """
        # Update status
        self._connection_status['last_check'] = time.time()
        
        # Check each exchange
        all_connected = True
        any_connected = False
        
        if self.ws_manager and hasattr(self.ws_manager, 'clients'):
            for exchange_name, client in self.ws_manager.clients.items():
                try:
                    # Check if client has connection status
                    is_connected = getattr(client, '_is_connected', False)
                    
                    self._connection_status['exchanges'][exchange_name] = {
                        'connected': is_connected,
                        'last_message': getattr(client, '_last_message_time', None)
                    }
                    
                    if is_connected:
                        any_connected = True
                    else:
                        all_connected = False
                
                except Exception as e:
                    logger.debug(f"[WS-OPT] Status check failed for {exchange_name}: {e}")
                    self._connection_status['exchanges'][exchange_name] = {
                        'connected': False,
                        'error': str(e)
                    }
                    all_connected = False
        
        # Update overall status
        self._connection_status['connected'] = any_connected
        self._connection_status['all_connected'] = all_connected
        
        return self._connection_status.copy()
    
    async def get_stream_status(self) -> Dict[str, Any]:
        """
        Get WebSocket stream status.
        
        Returns:
            Status dictionary
        """
        if not self.ws_manager:
            return {
                'initialized': False,
                'running': False,
                'streams': 0
            }
        
        status = self.ws_manager.get_stream_status()
        status['optimized'] = True
        status['fixed_symbols'] = len(self.fixed_symbols)
        
        return status
    
    async def stop_streaming(self) -> None:
        """
        Stop all WebSocket streams properly.
        
        CRITICAL: Must be called on shutdown to close connections!
        This method ensures all WebSocket streams are properly terminated and
        prevents resource leaks that can cause subsequent runs to hang.
        """
        if not self.ws_manager:
            logger.info("[WS-OPT] No WebSocket manager to stop")
            return
        
        logger.info("[WS-OPT] Stopping WebSocket streams...")
        
        try:
            close_ret = self.ws_manager.close()
            if inspect.isawaitable(close_ret):
                await asyncio.wait_for(close_ret, timeout=10.0)
            else:
                # run blocking close in a thread
                await asyncio.to_thread(self.ws_manager.close)
            logger.info("[WS-OPT] ✅ WebSocket streams stopped")
            self.is_initialized = False
            
        except asyncio.TimeoutError:
            logger.warning("[WS-OPT] ⚠️ WebSocket stop timeout (10s)")
        except Exception as e:
            logger.error(f"[WS-OPT] ⚠️ Error stopping WebSocket: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket connections."""
        await self.stop_streaming()

# ============= End of WebSocket Optimization Manager =============


class HealthMonitor:
    """
    HEALTH MONITORING SYSTEM (Layer 3 Guardian)
    
    Non-blocking health monitoring system that runs in the background
    and provides periodic health checks and alerts.
    """
    
    def __init__(self, telegram: Optional[Telegram] = None):
        """
        Initialize health monitor.
        
        Args:
            telegram: Telegram notifier for health alerts
        """
        self.telegram = telegram
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '300'))  # 5 minutes default
        
        # Performance metrics
        self.metrics = {
            'loops_completed': 0,
            'errors_caught': 0,
            'signals_processed': 0,
            'last_error': None,
            'last_error_time': None
        }
        
        # Health status
        self.health_status = 'healthy'
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        
        logger.info("="*70)
        logger.info("HEALTH MONITORING SYSTEM INITIALIZED (Layer 3 Guardian)")
        logger.info("="*70)
        logger.info(f"Heartbeat Interval: {self.heartbeat_interval}s")
        logger.info("="*70)
    
    async def start_monitoring(self) -> asyncio.Task:
        """
        Start monitoring in background (idempotent, non-blocking).
        
        Returns:
            The asyncio task running the monitoring loop
        """
        if self._task and not self._task.done():
            logger.warning("Health monitor already running")
            return self._task
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitor loop started in background")
        return self._task
    
    async def stop_monitoring(self):
        """Stop monitoring gracefully."""
        if not self._task:
            return
        
        logger.info("Stopping health monitor...")
        self._stop_event.set()
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _monitoring_loop(self):
        """Internal loop - runs in background."""
        logger.info("Health monitor loop started")
        
        try:
            while not self._stop_event.is_set():
                # Wait for heartbeat interval or stop event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.heartbeat_interval
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    # Normal timeout - perform health check
                    pass
                
                # Perform health checks
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                
                logger.info(f"💓 Heartbeat - Uptime: {uptime/3600:.1f}h, Status: {self.health_status}")
                
                # Update heartbeat
                self.last_heartbeat = datetime.now(timezone.utc)
                self.metrics['loops_completed'] += 1
                
                # Send periodic Telegram update
                if self.telegram and self.metrics['loops_completed'] % 12 == 0:  # Every hour
                    self.telegram.send(
                        f"💓 <b>Health Check</b>\n"
                        f"Status: {self.health_status.upper()}\n"
                        f"Uptime: {uptime/3600:.1f}h\n"
                        f"Loops: {self.metrics['loops_completed']}\n"
                        f"Errors: {self.metrics['errors_caught']}"
                    )
        
        except asyncio.CancelledError:
            logger.info("Health monitor loop cancelled")
            raise
        finally:
            logger.info("Health monitor loop exited")
    
    def record_error(self, error: str):
        """Record an error in the metrics."""
        self.metrics['errors_caught'] += 1
        self.metrics['last_error'] = error
        self.metrics['last_error_time'] = datetime.now(timezone.utc)
        
        # Update health status based on error frequency
        if self.metrics['errors_caught'] > 10:
            self.health_status = 'degraded'
        if self.metrics['errors_caught'] > 50:
            self.health_status = 'critical'
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            'status': self.health_status,
            'uptime_hours': uptime / 3600,
            'metrics': self.metrics,
            'last_heartbeat': self.last_heartbeat.isoformat()
        }


class AutoRestartManager:
    """
    AUTO-RESTART FAILSAFE (Layer 2 Defense)
    
    [... mevcut kod ...]
    """
    # Mevcut AutoRestartManager sınıfı aynen kalıyor
    def __init__(self, max_restarts: int = 1000, restart_delay: int = 30, 
                 telegram: Optional[Telegram] = None):
        self.max_restarts = max_restarts
        self.base_restart_delay = restart_delay
        self.telegram = telegram
        
        # Tracking
        self.restart_count = 0
        self.last_restart_time = None
        self.consecutive_failures = 0
        self.start_time = datetime.now(timezone.utc)
        
        # Health monitoring
        self.last_heartbeat = datetime.now(timezone.utc)
        self.health_check_interval = 60  # seconds
        
        logger.info("="*70)
        logger.info("AUTO-RESTART FAILSAFE INITIALIZED (Layer 2 Defense)")
        logger.info("="*70)
        logger.info(f"Max Restarts: {max_restarts}")
        logger.info(f"Base Restart Delay: {restart_delay}s")
        logger.info(f"Exponential Backoff: ENABLED")
        logger.info("="*70)
    
    def calculate_restart_delay(self) -> int:
        delay = min(
            self.base_restart_delay * (2 ** self.consecutive_failures),
            3600  # Max 1 hour
        )
        return int(delay)
    
    def should_restart(self) -> tuple[bool, str]:
        if self.restart_count >= self.max_restarts:
            return False, f"Maximum restart limit reached ({self.max_restarts})"
        
        if self.consecutive_failures > 10:
            return False, "Too many consecutive failures (10+), manual intervention required"
        
        return True, "Restart approved"
    
    def record_success(self):
        self.consecutive_failures = 0
        logger.info("✓ Bot operating normally, failure counter reset")
    
    def record_failure(self, reason: str):
        self.restart_count += 1
        self.consecutive_failures += 1
        self.last_restart_time = datetime.now(timezone.utc)
        
        logger.error("="*70)
        logger.error(f"FAILURE RECORDED (Attempt {self.restart_count}/{self.max_restarts})")
        logger.error(f"Reason: {reason}")
        logger.error(f"Consecutive Failures: {self.consecutive_failures}")
        logger.error("="*70)
        
        if self.telegram:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.telegram.send(
                f"🔄 <b>AUTO-RESTART TRIGGERED</b>\n"
                f"Attempt: {self.restart_count}/{self.max_restarts}\n"
                f"Reason: {reason}\n"
                f"Consecutive Failures: {self.consecutive_failures}\n"
                f"Uptime: {uptime/3600:.1f}h\n"
                f"Next restart in: {self.calculate_restart_delay()}s"
            )
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'restart_count': self.restart_count,
            'max_restarts': self.max_restarts,
            'consecutive_failures': self.consecutive_failures,
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            'last_restart': self.last_restart_time.isoformat() if self.last_restart_time else None
        }


class LiveTradingLauncher:
    """
    Comprehensive live trading launcher integrating all system components.
    """
    
    def __init__(self, mode: str = 'live', dry_run: bool = False, 
                 infinite: bool = False, auto_restart: bool = False,
                 max_restarts: int = 1000, restart_delay: int = 30,
                 debug_mode: bool = False):
        """
        Initialize live trading launcher.
        
        [... mevcut init docstring ...]
        """
                     
        # Define capital and risk parameters FIRST
        self.CAPITAL_USDT = 100  # Default capital in USDT
        self.RISK_PARAMS = {
            'max_position_size': 0.20,  # 20% max position
            'stop_loss_pct': 0.02,      # 2% stop loss
            'take_profit_pct': 0.015,   # 1.5% take profit
            'max_portfolio_risk': 0.05, # 5% max portfolio risk
            'max_drawdown': 0.10        # 10% max drawdown
        }
                     
        # Config ve trading pairs için instance variables
        self.config = None
        self.trading_pairs = []  # ← Config'den gelecek
        self.mode = mode
        self.dry_run = dry_run
        self.infinite = infinite
        self.auto_restart = auto_restart
        self.debug_mode = debug_mode
        self.coordinator = None
        self.telegram = None
        self.exchange_clients = {}
        self.strategies = {}
        self.restart_manager = None
        self.health_monitor = None
        
        # Cleanup tracking
        self._cleanup_done = False
        self._shutdown_event = asyncio.Event()
        
        # WebSocket optimization manager
        self.ws_optimizer = None
        
        # Phase 4 AI components
        self.regime_predictor = None
        self.price_engine = None
        self.strategy_adapter = None
        self.strategy_optimizer = None
        
        # Ultimate mode settings
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        
        # Debug logger
        self.debug_logger = None
        
        # Get trading pairs FIRST, before logging
        self.TRADING_PAIRS = self._get_trading_pairs()
        
        logger.info("="*70)
        logger.info("BEARISH ALPHA BOT - LIVE TRADING LAUNCHER")
        logger.info("="*70)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Capital: {self.CAPITAL_USDT} USDT")
        logger.info(f"Exchange: BingX")
        logger.info(f"Trading Pairs: {len(self.TRADING_PAIRS)}")
        if self.TRADING_PAIRS:
            logger.info(f"Symbols: {', '.join(self.TRADING_PAIRS[:3])}...")
        logger.info(f"Dry Run: {dry_run}")
        
        # Debug mode indicator
        if debug_mode:
            logger.info("")
            logger.info("🔍 DEBUG MODE ACTIVATED - Enhanced logging enabled")
            logger.info("🔍 Monitoring: Strategy signals, AI decisions, Risk calculations")
            logger.info("")
        
        # Live trading warning
        if mode == 'live':
            logger.warning("")
            logger.warning("⚠️  LIVE TRADING MODE: Using real USDT capital")
            logger.warning("⚠️  Ensure you understand the risks before proceeding")
            logger.warning("")
        
        # Ultimate mode indicators
        if infinite or auto_restart:
            logger.info("")
            logger.info("🚀 ULTIMATE CONTINUOUS TRADING MODE 🚀")
            logger.info(f"Infinite Mode: {'ENABLED' if infinite else 'DISABLED'}")
            logger.info(f"Auto-Restart: {'ENABLED' if auto_restart else 'DISABLED'}")
            if auto_restart:
                logger.info(f"Max Restarts: {max_restarts}")
                logger.info(f"Restart Delay: {restart_delay}s")
        
        logger.info("="*70)

    def _load_config(self) -> Dict[str, Any]:
        """Load and cache configuration using unified loader."""
        if self.config is None:
            from config.live_trading_config import LiveTradingConfiguration
            self.config = LiveTradingConfiguration.load(log_summary=False)
            logger.info("✓ Config loaded (ENV > YAML > Defaults)")
        return self.config
    
    def _get_trading_pairs(self) -> List[str]:
        """Get trading pairs from config, not hardcoded!"""
        if self.trading_pairs:
            return self.trading_pairs
            
        config = self._load_config()
        universe_cfg = config.get('universe', {})
        
        # 1. Önce fixed_symbols bak
        fixed_symbols = universe_cfg.get('fixed_symbols', [])
        
        # 2. Auto-select KAPALI mı kontrol et
        auto_select = universe_cfg.get('auto_select', False)
        
        if fixed_symbols and not auto_select:
            self.trading_pairs = fixed_symbols
            logger.info(f"✓ Using {len(fixed_symbols)} symbols from config (fixed mode)")
            logger.info(f"✓ Symbols: {', '.join(fixed_symbols)}")
        else:
            # Fallback: Default 3 symbols
            logger.warning("⚠️ No fixed symbols in config or auto_select=true")
            self.trading_pairs = [
                'BTC/USDT:USDT',
                'ETH/USDT:USDT', 
                'SOL/USDT:USDT'
            ]
            logger.info(f"✓ Using default {len(self.trading_pairs)} symbols")
        
        return self.trading_pairs
    
    async def cleanup(self):
        """
        Properly cleanup all resources.
        
        CRITICAL: Must be called before exit to prevent resource leaks!
        This method is idempotent - safe to call multiple times.
        
        Cleans up:
        - WebSocket streams
        - Exchange connections (ccxt clients)
        - Production system components
        - Async tasks
        - Log handlers
        """
        if self._cleanup_done:
            logger.info("Cleanup already completed, skipping")
            return
        
        logger.info("=" * 70)
        logger.info("🧹 STARTING CLEANUP")
        logger.info("=" * 70)
        
        cleanup_errors = []
        
        try:
            # 1. Stop WebSocket streams
            if self.ws_optimizer:
                logger.info("Stopping WebSocket streams...")
                try:
                    await asyncio.wait_for(
                        self.ws_optimizer.stop_streaming(),
                        timeout=10.0
                    )
                    logger.info("✅ WebSocket streams stopped")
                except asyncio.TimeoutError:
                    logger.error("⚠️ WebSocket stop timeout (10s)")
                    cleanup_errors.append("WebSocket stop timeout")
                except Exception as e:
                    logger.error(f"⚠️ WebSocket stop failed: {e}")
                    cleanup_errors.append(f"WebSocket: {e}")
            
            # 2. Close production system components
            if self.coordinator:
                logger.info("Closing production system...")
                try:
                    await asyncio.wait_for(
                        self.coordinator.stop_system(),
                        timeout=10.0
                    )
                    logger.info("✅ Production system stopped")
                except asyncio.TimeoutError:
                    logger.error("⚠️ Production system stop timeout (10s)")
                    cleanup_errors.append("Production system timeout")
                except Exception as e:
                    logger.error(f"⚠️ Production system stop failed: {e}")
                    cleanup_errors.append(f"Production system: {e}")
            
            # 3. Close exchange clients (CRITICAL!)
            if self.exchange_clients:
                logger.info("Closing exchange connections...")
                for exchange_name, client in self.exchange_clients.items():
                    try:
                        close_ret = client.close()
                        if inspect.isawaitable(close_ret):
                            await asyncio.wait_for(close_ret, timeout=5.0)
                        else:
                            await asyncio.to_thread(client.close)
                        logger.info(f"✅ {exchange_name} connection closed")
                    except asyncio.TimeoutError:
                        logger.error(f"⚠️ {exchange_name} close timeout (5s)")
                        cleanup_errors.append(f"{exchange_name} close timeout")
                    except Exception as e:
                        logger.error(f"⚠️ {exchange_name} close failed: {e}")
                        cleanup_errors.append(f"{exchange_name}: {e}")
            
            # 4. Cancel pending async tasks
            logger.info("Cancelling pending tasks...")
            pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
            
            if pending:
                logger.info(f"⚠️ Found {len(pending)} pending tasks")
                
                for task in pending:
                    task.cancel()
                    logger.debug(f"Cancelled task: {task.get_name()}")
                
                # Wait for cancellation with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=5.0
                    )
                    logger.info("✅ All pending tasks cancelled")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Some tasks did not cancel in time")
                    cleanup_errors.append("Task cancellation timeout")
            else:
                logger.info("✅ No pending tasks")
            
            # 5. Flush logs
            logger.info("Flushing logs...")
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception as e:
                    logger.error(f"⚠️ Log flush failed: {e}")
            
            self._cleanup_done = True
            
            logger.info("=" * 70)
            if cleanup_errors:
                logger.warning(f"⚠️ CLEANUP COMPLETED WITH {len(cleanup_errors)} ERRORS:")
                for error in cleanup_errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("✅ CLEANUP COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Cleanup fatal error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_environment(self) -> bool:
        """
        Load and validate environment variables.
        
        Returns:
            True if all required variables are present
        """
        logger.info("\n[1/8] Loading Environment Configuration...")
        
        required_vars = ['BINGX_KEY', 'BINGX_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        logger.info("✓ BingX credentials found")
        
        # Optional Telegram setup
        tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        tg_chat = os.getenv('TELEGRAM_CHAT_ID')
        
        if tg_token and tg_chat:
            self.telegram = Telegram(tg_token, tg_chat)
            logger.info("✓ Telegram notifications enabled")
        else:
            logger.info("ℹ Telegram notifications disabled (optional)")
        
        # Initialize debug logger if debug mode is enabled
        if self.debug_mode:
            self.debug_logger = DebugLogger(debug_mode=True)
            logger.info("✓ Debug Logger initialized")
        
        # Initialize health monitor (Layer 3)
        if self.infinite or self.auto_restart:
            self.health_monitor = HealthMonitor(telegram=self.telegram)
            logger.info("✓ Health Monitor initialized (Layer 3 Guardian)")
        
        # Initialize auto-restart manager if enabled
        if self.auto_restart:
            self.restart_manager = AutoRestartManager(
                max_restarts=self.max_restarts,
                restart_delay=self.restart_delay,
                telegram=self.telegram
            )
        
        # Initialize WebSocket optimizer
        self.ws_optimizer = OptimizedWebSocketManager()
        logger.info("✓ WebSocket Optimizer initialized")
        
        return True
    
    def _initialize_exchange_connection(self) -> bool:
        """OPTIMIZED BingX initialization."""
        logger.info("\n[2/8] Initializing BingX Exchange Connection...")
        
        # Get trading pairs from config
        trading_pairs = self._get_trading_pairs()
        
        try:
            # Create BingX client with credentials
            bingx_creds = {
                'apiKey': os.getenv('BINGX_KEY'),
                'secret': os.getenv('BINGX_SECRET'),
            }
            
            bingx_client = CcxtClient('bingx', bingx_creds)
    
            # WebSocket optimization with CONFIG symbols
            bingx_client.set_required_symbols(trading_pairs)
            logger.info(f"✓ BingX client optimized for {len(trading_pairs)} symbols only")
            
            self.exchange_clients['bingx'] = bingx_client
            
            # Test connection with single API call instead of loading 2528 markets
            logger.info("Testing BingX connection...")
            test_ticker = bingx_client.fetch_ticker('BTC/USDT:USDT')
            logger.info(f"✓ Connected to BingX - Test price: BTC=${test_ticker['last']:.2f}")
            
            # Test authentication with balance check
            try:
                balance = bingx_client.get_bingx_balance()
                logger.info("✓ BingX authentication successful")
            except Exception as e:
                logger.warning(f"⚠️  BingX authentication test failed: {e}")
            
            # Verify ONLY configured pairs
            logger.info(f"Verifying {len(trading_pairs)} trading pairs...")
            verified_pairs = []
        
            for pair in trading_pairs:  # ← CONFIG'DEN GELEN LİSTE
                try:
                    ticker = bingx_client.fetch_ticker(pair)
                    verified_pairs.append(pair)
                    logger.info(f"  ✓ {pair}: ${ticker['last']:.2f}")
                except Exception as e:
                    logger.warning(f"  ❌ {pair}: {e}")
            
            # Check if enough pairs were verified (allow some failures)
            min_required = max(1, len(trading_pairs) // 2)  # At least half should work
            
            if len(verified_pairs) >= min_required:
                logger.info(f"✓ {len(verified_pairs)}/{len(trading_pairs)} trading pairs verified")
                return True
            else:
                logger.error(f"Only {len(verified_pairs)}/{len(trading_pairs)} pairs verified (minimum {min_required} required)")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to BingX: {e}")
            return False
    
    def _initialize_risk_management(self) -> bool:
        """
        Initialize risk management system with custom parameters.
        
        Returns:
            True if initialization successful
        """
        logger.info("\n[3/8] Initializing Risk Management System...")
        
        try:
            # Create risk configuration with custom limits
            risk_config = RiskConfiguration(custom_limits=self.RISK_PARAMS)
            logger.info("✓ Risk configuration loaded")
            logger.info(f"  - Max position size: {self.RISK_PARAMS['max_position_size']:.1%}")
            logger.info(f"  - Stop loss: {self.RISK_PARAMS['stop_loss_pct']:.1%}")
            logger.info(f"  - Take profit: {self.RISK_PARAMS['take_profit_pct']:.1%}")
            logger.info(f"  - Max drawdown: {self.RISK_PARAMS['max_drawdown']:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize risk management: {e}")
            return False
    
    def _is_ws_initialized(self) -> bool:
        """
        Helper method to safely check if WebSocket optimizer is initialized.
        
        Returns:
            True if ws_optimizer exists and is initialized, False otherwise
        """
        return self.ws_optimizer is not None and getattr(self.ws_optimizer, 'is_initialized', False)
    
    async def _initialize_ai_components(self) -> bool:
        """
        Initialize Phase 4 AI enhancement components.
        
        [... mevcut kod ...]
        """
        logger.info("\n[4/8] Initializing Phase 4 AI Components...")
        
        try:
            # Phase 4.1: ML Regime Prediction
            logger.info("Initializing regime predictor...")
            self.regime_predictor = MLRegimePredictor()
            logger.info("✓ ML Regime Predictor initialized")
            
            # Phase 4.2: Adaptive Learning - integrated with strategies
            logger.info("✓ Adaptive Learning ready (integrated with strategies)")
            
            # Phase 4.3: Strategy Optimization
            logger.info("Initializing strategy optimizer...")
            config = OptimizationConfiguration.get_default_config()
            self.strategy_optimizer = StrategyOptimizer(config)
            logger.info("✓ Strategy Optimizer initialized")
            
            # Phase 4.4: Price Prediction
            logger.info("Initializing price prediction engine...")
            
            # Initialize multi-timeframe predictor
            models = {
                '5m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
                '15m': EnsemblePricePredictor({'lstm': None, 'transformer': None}),
                '1h': EnsemblePricePredictor({'lstm': None, 'transformer': None})
            }
            multi_timeframe_predictor = MultiTimeframePricePredictor(models)
            
            # Correct initialization with required parameter
            self.price_engine = AdvancedPricePredictionEngine(multi_timeframe_predictor)
            logger.info("✓ Price Prediction Engine initialized")
            
            # Strategy integration adapter
            if self.regime_predictor and self.price_engine:
                self.strategy_adapter = AIEnhancedStrategyAdapter(
                    self.price_engine,
                    self.regime_predictor
                )
                logger.info("✓ AI-Enhanced Strategy Adapter initialized")
            
            logger.info("\n✓ Phase 4 AI Components fully integrated:")
            logger.info("  - ML Regime Prediction: ACTIVE")
            logger.info("  - Adaptive Learning: ACTIVE")
            logger.info("  - Strategy Optimization: ACTIVE")
            logger.info("  - Price Prediction: ACTIVE")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI components: {e}")
            logger.warning("⚠ Continuing with limited AI features")
            return False  # Non-critical, can continue
    
    async def _initialize_strategies(self) -> bool:
        """Initialize adaptive trading strategies."""
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # ÖNCE CONFIG'İ YÜKLE
            import yaml
            config_path = os.getenv('CONFIG_PATH', 'config/config.example.yaml')
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Config loaded from {config_path}")
            
            # Initialize regime analyzer for adaptive strategies
            from core.market_regime import MarketRegimeAnalyzer
            regime_analyzer = MarketRegimeAnalyzer()
            
            # Strategy configurations FROM CONFIG FILE
            signals_config = self.config.get('signals', {})
    
            # Adaptive OB config - config dosyasından oku!
            ob_cfg = signals_config.get('oversold_bounce', {})
            if not ob_cfg.get('enable', True):
                logger.info("⚠️ OversoldBounce strategy disabled in config")
                
            adaptive_ob_config = {
                'adaptive_rsi_base': ob_cfg.get('adaptive_rsi_base', 40),
                'adaptive_rsi_range': ob_cfg.get('adaptive_rsi_range', 15),
                'tp_pct': ob_cfg.get('tp_pct', 0.015),
                'sl_atr_mult': ob_cfg.get('sl_atr_mult', 1.0),
                'ignore_regime': ob_cfg.get('ignore_regime', True),
                'enable': ob_cfg.get('enable', True),
                # Backwards compatibility
                'rsi_max': ob_cfg.get('rsi_max', ob_cfg.get('adaptive_rsi_base', 40))
            }
    
            # Adaptive STR config - config dosyasından oku!
            str_cfg = signals_config.get('short_the_rip', {})
            if not str_cfg.get('enable', True):
                logger.info("⚠️ ShortTheRip strategy disabled in config")
                
            adaptive_str_config = {
                'adaptive_rsi_base': str_cfg.get('adaptive_rsi_base', 40),
                'adaptive_rsi_range': str_cfg.get('adaptive_rsi_range', 15),
                'tp_pct': str_cfg.get('tp_pct', 0.012),
                'sl_atr_mult': str_cfg.get('sl_atr_mult', 1.2),
                'ignore_regime': str_cfg.get('ignore_regime', True),
                'enable': str_cfg.get('enable', True),
                # Backwards compatibility
                'rsi_min': str_cfg.get('rsi_min', str_cfg.get('adaptive_rsi_base', 40))
            }
    
            logger.info(f"✓ OB Config: base={adaptive_ob_config['adaptive_rsi_base']}, "
                       f"range=±{adaptive_ob_config['adaptive_rsi_range']}, "
                       f"enabled={adaptive_ob_config['enable']}")
            logger.info(f"✓ STR Config: base={adaptive_str_config['adaptive_rsi_base']}, "
                       f"range=±{adaptive_str_config['adaptive_rsi_range']}, "
                       f"enabled={adaptive_str_config['enable']}")
            
            # Adaptive Oversold Bounce strategy
            if adaptive_ob_config['enable']:
                self.strategies['adaptive_ob'] = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
                logger.info("✓ Adaptive Oversold Bounce strategy initialized")
            
            # Adaptive Short The Rip strategy
            if adaptive_str_config['enable']:
                self.strategies['adaptive_str'] = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
                logger.info("✓ Adaptive Short The Rip strategy initialized")
            
            if not self.strategies:
                logger.warning("⚠️ No strategies enabled!")
                return False
                
            logger.info(f"\n✓ {len(self.strategies)} strategies ready for trading")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategies: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_production_system(self) -> bool:
        """Initialize Phase 3 production coordinator with all components."""
        logger.info("\n[6/8] Initializing Production Trading System...")
        
        try:
            # Config'i yükle
            if not self.config:
                self._load_config()
            
            # ProductionCoordinator'ı SADECE BİR KEZ yarat!
            from core.production_coordinator import ProductionCoordinator
            self.coordinator = ProductionCoordinator()
            
            # WebSocket optimizer'ı ayarla ve başlat
            self.ws_optimizer.setup_from_config(self.config)
            ws_initialized = await self.ws_optimizer.initialize_websockets(self.exchange_clients)
            
            # Eğer WebSocket başarılıysa, coordinator'a ver
            if ws_initialized:
                logger.info("✓ WebSocket connections initialized")
                # Coordinator'ın websocket_manager attribute'una dışarıdan set et
                self.coordinator.websocket_manager = self.ws_optimizer.ws_manager
            else:
                logger.warning("⚠️ WebSocket failed, using REST API mode")
                # WebSocket yoksa bile devam et, REST API kullanılacak
            
            # Portfolio config hazırla
            portfolio_config = {
                'equity_usd': self.CAPITAL_USDT,
                'max_portfolio_risk': self.RISK_PARAMS['max_portfolio_risk'],
                'max_position_size': self.RISK_PARAMS['max_position_size'],
                'max_drawdown': self.RISK_PARAMS['max_drawdown']
            }
            
            # PUBLIC initialize_production_system metodunu çağır (artık var!)
            init_result = await self.coordinator.initialize_production_system(
                exchange_clients=self.exchange_clients,
                portfolio_config=portfolio_config,
                mode=self.mode,
                trading_symbols=self.TRADING_PAIRS  # ← FIX: Pass symbols
            )
            
            if not init_result['success']:
                logger.error(f"❌ Failed: {init_result.get('reason')}")
                return False
            
            # Active symbols'ı ayarla (fallback for edge cases)
            if hasattr(self.coordinator, 'active_symbols'):
                if not self.coordinator.active_symbols:  # Only if still empty
                    self.coordinator.active_symbols = self.trading_pairs
                    logger.info(f"✓ Fallback: Configured with {len(self.trading_pairs)} symbols")
            
            logger.info("✓ Production system initialized")
            logger.info(f"  Components: {init_result['components']}")
            
            # WebSocket durumunu göster
            if self.ws_optimizer:
                ws_status = await self.ws_optimizer.get_stream_status()
                logger.info(f"✓ WebSocket Status: {ws_status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _register_strategies(self) -> bool:
        """Initialize adaptive trading strategies."""
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # Config ZATEN yüklü olmalı
            if not self.config:
                self._load_config()
            
            logger.info(f"✓ Using config with {len(self.trading_pairs)} symbols")
            # Equal allocation across strategies
            allocation_per_strategy = 1.0 / len(self.strategies)
            
            for strategy_name, strategy_instance in self.strategies.items():
                result = self.coordinator.register_strategy(
                    strategy_name=strategy_name,
                    strategy_instance=strategy_instance,
                    initial_allocation=allocation_per_strategy
                )
                
                if result.get('status') == 'success':
                    logger.info(f"✓ {strategy_name}: {allocation_per_strategy:.1%} allocation")
                else:
                    logger.warning(f"⚠ Failed to register {strategy_name}: {result.get('reason')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register strategies: {e}")
            return False
    
    async def _perform_preflight_checks(self) -> bool:
        """
        Perform comprehensive pre-flight system checks.
        
        Returns:
            True if all checks pass
        """
        logger.info("\n[8/8] Performing Pre-Flight System Checks...")
        
        checks_passed = True
        
        try:
            # Check 1: Exchange connectivity
            logger.info("Check 1/6: Exchange connectivity...")
            try:
                ticker = self.exchange_clients['bingx'].ticker('BTC/USDT:USDT')
                logger.info(f"✓ BTC/USDT:USDT price: ${ticker.get('last', 0):.2f}")
            except Exception as e:
                logger.error(f"❌ Exchange connectivity failed: {e}")
                checks_passed = False
            
            # Check 2: System state
            logger.info("Check 2/6: System state...")
            state = self.coordinator.get_system_state()
            if state['is_initialized']:
                logger.info("✓ Production system initialized")
            else:
                logger.error("❌ Production system not initialized")
                checks_passed = False
            
            # Check 3: Risk limits
            logger.info("Check 3/6: Risk limits...")
            if self.coordinator.risk_manager:
                risk_summary = self.coordinator.risk_manager.get_portfolio_summary()
                logger.info(f"✓ Portfolio value: ${risk_summary['portfolio_value']:.2f}")
                logger.info(f"✓ Risk limits configured")
            else:
                logger.error("❌ Risk manager not available")
                checks_passed = False
            
            # Check 4: Strategies
            logger.info("Check 4/6: Strategy registration...")
            if self.coordinator.portfolio_manager:
                strategies = self.coordinator.portfolio_manager.strategies
                logger.info(f"✓ {len(strategies)} strategies registered")
            else:
                logger.error("❌ Portfolio manager not available")
                checks_passed = False
            
            # Check 5: Emergency protocols
            logger.info("Check 5/6: Emergency shutdown protocols...")
            if self.coordinator.circuit_breaker:
                logger.info("✓ Circuit breaker active")
            else:
                logger.warning("⚠ Circuit breaker not available")
            
            # Check 6: WebSocket optimization
            logger.info("Check 6/6: WebSocket optimization...")
            # Use helper method to safely check WebSocket initialization
            if self._is_ws_initialized():
                ws_status = await self.ws_optimizer.get_stream_status()
                logger.info(f"✓ WebSocket optimized: {ws_status['active_streams']} streams active")
            else:
                logger.warning("⚠ WebSocket not initialized (will use REST API)")
            
            logger.info("\n" + "="*70)
            if checks_passed:
                logger.info("✓ ALL PRE-FLIGHT CHECKS PASSED")
            else:
                logger.error("❌ SOME PRE-FLIGHT CHECKS FAILED")
            logger.info("="*70)
            
            return checks_passed
            
        except Exception as e:
            logger.error(f"❌ Pre-flight checks failed: {e}")
            return False
    
    async def _wait_for_websocket_connection(self, timeout: int = 30, check_interval: int = 1) -> bool:
        """
        Wait for WebSocket connection to establish with timeout.
        
        Args:
            timeout: Max seconds to wait (default: 30s)
            check_interval: Seconds between checks (default: 1s)
        
        Returns:
            bool: True if connected, False if timeout
        """
        logger.info(f"[CONNECTION] Waiting for WebSocket connection (timeout: {timeout}s)...")
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Check timeout
            if elapsed >= timeout:
                logger.error(f"❌ WebSocket connection TIMEOUT after {elapsed:.1f}s")
                return False
            
            # Check connection status
            try:
                status = self.ws_optimizer.get_connection_status()
                
                # Log status periodically
                if int(elapsed) % 5 == 0 and int(elapsed) > 0:  # Every 5 seconds (after first second)
                    logger.info(f"[CONNECTION] Status check ({elapsed:.0f}s): connected={status.get('connected')}, exchanges={len(status.get('exchanges', {}))}")
                
                # Check if connected
                if status.get('connected'):
                    logger.info(f"✅ WebSocket CONNECTED after {elapsed:.1f}s")
                    return True
                
                # Check for errors
                if status.get('error'):
                    logger.error(f"❌ WebSocket error: {status['error']}")
                    return False
            
            except Exception as e:
                logger.error(f"⚠️ Status check failed: {e}")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    async def _establish_websocket_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """
        Establish WebSocket connection with retry logic.
        
        Args:
            max_retries: Maximum retry attempts (default: 3)
            timeout: Timeout per attempt in seconds (default: 30s)
        
        Returns:
            bool: True if connection established, False otherwise
        """
        logger.info("=" * 70)
        logger.info("ESTABLISHING WEBSOCKET CONNECTION")
        logger.info("=" * 70)
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"[ATTEMPT {attempt}/{max_retries}] Starting WebSocket streams...")
            
            try:
                # Mark as connecting
                if self.ws_optimizer:
                    self.ws_optimizer._connection_status['connecting'] = True
                    self.ws_optimizer._connection_status['error'] = None
                
                # Start streaming (initialize_websockets already starts tasks)
                streaming_tasks = await self.ws_optimizer.initialize_websockets(
                    self.exchange_clients
                )
                
                if not streaming_tasks:
                    logger.warning(f"⚠️ No streaming tasks created on attempt {attempt}/{max_retries}")
                    if attempt < max_retries:
                        retry_delay = 5 * attempt
                        logger.info(f"Waiting {retry_delay}s before retry...")
                        await asyncio.sleep(retry_delay)
                    continue
                
                # Wait for connection with timeout
                connected = await self._wait_for_websocket_connection(timeout=timeout)
                
                if connected:
                    logger.info(f"✅ Connection established on attempt {attempt}/{max_retries}")
                    if self.ws_optimizer:
                        self.ws_optimizer._connection_status['connecting'] = False
                    return True
                else:
                    logger.warning(f"⚠️ Connection timeout on attempt {attempt}/{max_retries}")
                    
                    # Stop current attempt before retry
                    if attempt < max_retries:
                        logger.info("Stopping current streams before retry...")
                        try:
                            await asyncio.wait_for(
                                self.ws_optimizer.stop_streaming(),
                                timeout=10.0
                            )
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ Timeout stopping streams")
                        
                        # Wait before retry (exponential backoff)
                        retry_delay = 5 * attempt  # 5s, 10s, 15s
                        logger.info(f"Waiting {retry_delay}s before retry...")
                        await asyncio.sleep(retry_delay)
            
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout on attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)
            
            except Exception as e:
                logger.error(f"❌ Error on attempt {attempt}/{max_retries}: {e}")
                if self.ws_optimizer:
                    self.ws_optimizer._connection_status['error'] = str(e)
                if attempt < max_retries:
                    await asyncio.sleep(5 * attempt)
        
        # All attempts failed
        logger.error("=" * 70)
        logger.error(f"❌ WEBSOCKET CONNECTION FAILED AFTER {max_retries} ATTEMPTS")
        logger.error("=" * 70)
        
        if self.ws_optimizer:
            self.ws_optimizer._connection_status['connecting'] = False
        
        return False
    
    async def _start_trading_loop(self, duration: Optional[float] = None) -> None:
        """
        Start the main trading loop with WebSocket optimization and connection retry.
        
        Args:
            duration: Optional duration in seconds (None for indefinite)
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING LIVE TRADING")
        logger.info("="*70)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Duration: {'Indefinite' if duration is None else f'{duration}s'}")
        logger.info(f"Trading Pairs: {len(self.TRADING_PAIRS)}")
        
        # Store health task for cleanup
        _health_task = None
        
        try:
            # Establish WebSocket connection with retry logic
            ws_connected = False
            if self.ws_optimizer and self._is_ws_initialized():
                ws_connected = await self._establish_websocket_connection(
                    max_retries=3,
                    timeout=30
                )
                
                if not ws_connected:
                    logger.warning("=" * 70)
                    logger.warning("⚠️ WebSocket connection failed after multiple attempts")
                    logger.warning("⚠️ Continuing with REST API mode (reduced real-time data)")
                    logger.warning("=" * 70)
                    
                    # Send Telegram notification
                    if self.telegram:
                        self.telegram.send(
                            "⚠️ <b>WebSocket Connection Failed</b>\n"
                            "Trading will continue using REST API\n"
                            "Real-time data may be limited"
                        )
                else:
                    logger.info("=" * 70)
                    logger.info("✅ WEBSOCKET CONNECTED - REAL-TIME DATA STREAMING")
                    logger.info("=" * 70)
            else:
                logger.info("WebSocket: ❌ DISABLED or NOT INITIALIZED")
            
            logger.info("="*70)
            
            # Send Telegram notification
            if self.telegram:
                if ws_connected:
                    ws_info = "WebSocket CONNECTED ✅"
                else:
                    ws_info = "REST API mode (WebSocket unavailable)"
                
                self.telegram.send(
                    f"🚀 <b>LIVE TRADING STARTED</b>\n"
                    f"Mode: {self.mode.upper()}\n"
                    f"Capital: {self.CAPITAL_USDT} USDT\n"
                    f"Exchange: BingX\n"
                    f"Pairs: {len(self.TRADING_PAIRS)}\n"
                    f"Data: {ws_info}\n"
                    f"Max Position: {self.RISK_PARAMS['max_position_size']:.1%}\n"
                    f"Stop Loss: {self.RISK_PARAMS['stop_loss_pct']:.1%}\n"
                    f"Take Profit: {self.RISK_PARAMS['take_profit_pct']:.1%}"
                )
            
            # Start health monitoring in background (non-blocking)
            if self.health_monitor:
                _health_task = asyncio.create_task(
                    self.health_monitor.start_monitoring()
                )
                logger.info("✓ Health monitor started in background")
            
            # Now main flow can proceed!
            await self.coordinator.run_production_loop(
                mode=self.mode,
                duration=duration,
                continuous=self.infinite
            )
            
        except KeyboardInterrupt:
            logger.info("\n⚠ Keyboard interrupt received - initiating shutdown...")
            raise  # Re-raise to be caught by main()
            
        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
            if self.health_monitor:
                self.health_monitor.record_error(str(e))
            raise  # Re-raise to be caught by main()
        
        finally:
            # Stop health monitor first
            if self.health_monitor:
                try:
                    await self.health_monitor.stop_monitoring()
                except Exception as e:
                    logger.error(f"Error stopping health monitor: {e}")
            
            # Cancel background task if still running
            if _health_task and not _health_task.done():
                _health_task.cancel()
                try:
                    await _health_task
                except asyncio.CancelledError:
                    pass
    
    async def _monitor_websocket_health(self):
        """
        Enhanced WebSocket health monitor with error recovery.
        
        This method monitors WebSocket stream health and attempts automatic recovery
        when issues are detected. Includes connection status checking, consecutive 
        error tracking, parse frame error detection, and exponential backoff.
        """
        logger.info("Starting WebSocket health monitor...")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        # Use helper method to safely check WebSocket initialization
        while self._is_ws_initialized():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get connection status using new method
                status = self.ws_optimizer.get_connection_status()
                
                # Check connection status
                if not status.get('connected'):
                    consecutive_errors += 1
                    logger.error(f"❌ No active WebSocket connection! (attempt {consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical("❌ WebSocket completely failed after multiple checks!")
                        if self.telegram:
                            self.telegram.send(
                                "🛑 <b>CRITICAL</b>\n"
                                "WebSocket system failure!\n"
                                "Trading continues with REST API.\n"
                                "Manual intervention may be required."
                            )
                        # Don't shutdown - continue with REST API
                        logger.warning("⚠️ Continuing with REST API mode")
                        break
                    else:
                        # Attempt restart with exponential backoff
                        logger.warning(f"Attempting WebSocket recovery ({consecutive_errors}/{max_consecutive_errors})...")
                        await self._restart_websockets_with_backoff()
                
                else:
                    # Connection is healthy
                    consecutive_errors = 0
                    
                    # Check for errors in status
                    error_msg = status.get('error')
                    if error_msg:
                        logger.warning(f"⚠️ WebSocket error detected: {error_msg}")
                    
                    # Log healthy status periodically
                    connected_exchanges = [
                        ex for ex, st in status.get('exchanges', {}).items() 
                        if st.get('connected')
                    ]
                    logger.info(f"✅ WebSocket healthy: {len(connected_exchanges)} exchange(s) connected")
                
            except Exception as e:
                logger.error(f"WebSocket monitor error: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Monitor failed {max_consecutive_errors} times!")
                    break
                
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _restart_websockets_with_backoff(self):
        """
        Restart WebSockets with exponential backoff strategy.
        
        Attempts to restart WebSocket connections up to max_attempts times with
        increasing delays between attempts to allow system stabilization.
        """
        max_attempts = 3
        base_delay = 5  # seconds
        
        for attempt in range(max_attempts):
            try:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                
                logger.info(f"Restarting WebSockets (attempt {attempt + 1}/{max_attempts})...")
                
                if attempt > 0:
                    logger.info(f"Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
                
                # First, close existing connections
                await self.ws_optimizer.shutdown()
                await asyncio.sleep(2)
                
                # Restart WebSocket connections
                await self.ws_optimizer.initialize_websockets(self.exchange_clients)
                
                # Check if restart was successful
                await asyncio.sleep(5)  # Wait for stabilization
                status = await self.ws_optimizer.get_stream_status()
                
                if status.get('active_streams', 0) > 0:
                    logger.info(f"✅ WebSocket restart successful! {status['active_streams']} streams active")
                    if self.telegram:
                        self.telegram.send(
                            f"✅ <b>WebSocket Recovered</b>\n"
                            f"Active streams: {status['active_streams']}\n"
                            f"System operational"
                        )
                    return True
                else:
                    logger.warning(f"WebSocket restart attempt {attempt + 1} failed")
                    
            except Exception as e:
                logger.error(f"WebSocket restart error (attempt {attempt + 1}): {e}")
        
        logger.error(f"❌ Failed to restart WebSockets after {max_attempts} attempts")
        return False
    
    async def _shutdown(self) -> None:
        """Graceful shutdown of trading system."""
        logger.info("\n" + "="*70)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("="*70)
        
        try:
            # Stop health monitoring first
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
                health_report = self.health_monitor.get_health_report()
                logger.info(f"Final health report: {health_report}")
            
            # Shutdown WebSocket connections
            if self.ws_optimizer:
                await self.ws_optimizer.shutdown()
                logger.info("✓ WebSocket connections closed")
            
            if self.coordinator:
                await self.coordinator.stop_system()
                logger.info("✓ Production system stopped")
            
            # Send Telegram notification with health summary
            if self.telegram:
                msg = "🛑 <b>Trading stopped - Graceful shutdown completed</b>"
                if self.health_monitor:
                    hr = self.health_monitor.get_health_report()
                    msg += f"\n\nUptime: {hr['uptime_hours']:.1f}h\n"
                    msg += f"Status: {hr['status']}\n"
                    msg += f"Errors: {hr['metrics']['errors_caught']}"
                # Use helper method to safely check WebSocket initialization
                if self._is_ws_initialized():
                    ws_status = await self.ws_optimizer.get_stream_status()
                    msg += f"\nWebSocket streams: {ws_status['active_streams']}"
                self.telegram.send(msg)
            
            # Generate post-session analysis
            self._generate_post_session_analysis()
            
            logger.info("="*70)
            logger.info("SHUTDOWN COMPLETE")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _print_configuration_summary(self):
        """
        Print comprehensive configuration summary at startup.
        Issue #119: Enhanced log header with complete system information.
        """
        # Add explicit header for test compatibility (Issue #106)
        logger.info("="*70)
        logger.info("📊 CONFIGURATION SUMMARY")
        logger.info("="*70)
        
        # Collect system information
        system_info = SystemInfoCollector.get_system_info()
        
        # Get risk manager from coordinator if available
        risk_manager = None
        if self.coordinator and hasattr(self.coordinator, 'risk_manager'):
            risk_manager = self.coordinator.risk_manager
        
        # Format startup header with all parameters
        header = format_startup_header(
            system_info=system_info,
            mode=self.mode,
            dry_run=self.dry_run,
            debug_mode=self.debug_mode,
            exchange_clients=self.exchange_clients,
            ws_manager=self.ws_optimizer if hasattr(self, 'ws_optimizer') else None,
            capital=self.CAPITAL_USDT,
            trading_pairs=self.TRADING_PAIRS,
            strategies=self.strategies if hasattr(self, 'strategies') else {},
            risk_params=self.RISK_PARAMS,
            risk_manager=risk_manager
        )
        
        # Log formatted header
        logger.info("\n" + header + "\n")
    
    def _generate_post_session_analysis(self, log_filename: str = None):
        """
        Generate post-session analysis from log files.
        Issue #106: Parse logs for errors, warnings, and trade statistics.
        
        Args:
            log_filename: Log file to analyze (optional)
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("POST-SESSION ANALYSIS")
            logger.info("="*70)
            
            # Find log file
            if not log_filename:
                import glob
                log_files = glob.glob('live_trading_*.log')
                if log_files:
                    log_filename = sorted(log_files)[-1]  # Most recent
            
            if not log_filename or not os.path.exists(log_filename):
                logger.warning("No log file found for analysis")
                return
            
            # Parse log file
            error_count = 0
            warning_count = 0
            signal_count = 0
            trade_count = 0
            
            with open(log_filename, 'r') as f:
                for line in f:
                    if 'ERROR' in line:
                        error_count += 1
                    elif 'WARNING' in line:
                        warning_count += 1
                    elif 'Signal submitted' in line or 'signal from' in line.lower():
                        signal_count += 1
                    elif 'Position opened' in line or 'Trade executed' in line:
                        trade_count += 1
            
            # Summary
            logger.info(f"Log File: {log_filename}")
            logger.info(f"\nSession Statistics:")
            logger.info(f"  Signals Generated: {signal_count}")
            logger.info(f"  Trades Executed: {trade_count}")
            logger.info(f"  Warnings: {warning_count}")
            logger.info(f"  Errors: {error_count}")
            
            # Health assessment
            if error_count > 50:
                logger.warning("⚠️  High error count - system may need attention")
            elif error_count > 10:
                logger.info("ℹ️  Moderate error count - review logs")
            else:
                logger.info("✅ Low error count - system healthy")
            
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Error generating post-session analysis: {e}")
    
    async def _emergency_shutdown(self, reason: str) -> None:
        """
        Emergency shutdown protocol.
        
        Args:
            reason: Reason for emergency shutdown
        """
        logger.critical("\n" + "="*70)
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        logger.critical(f"Reason: {reason}")
        logger.critical("="*70)
        
        try:
            # Force close WebSocket connections
            if self.ws_optimizer:
                await self.ws_optimizer.shutdown()
                logger.critical("✓ WebSocket connections force closed")
            
            # Stop coordinator
            if self.coordinator:
                await self.coordinator.stop_system()
                logger.critical("✓ Production system emergency stopped")
            
            # Send Telegram alert
            if self.telegram:
                self.telegram.send(
                    f"🚨 <b>EMERGENCY SHUTDOWN</b>\n"
                    f"Reason: {reason}\n"
                    f"Time: {datetime.now(timezone.utc).isoformat()}"
                )
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}")
    
    async def run(self, duration: Optional[float] = None) -> int:
        """
        Main entry point - run complete live trading system.
        
        Args:
            duration: Optional trading duration in seconds
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # If auto-restart is enabled, wrap in restart loop
        if self.auto_restart:
            return await self._run_with_auto_restart(duration)
        else:
            return await self._run_once(duration)
    
    async def _run_once(self, duration: Optional[float] = None) -> int:
        """
        Run trading system once without auto-restart.
        
        Args:
            duration: Optional trading duration in seconds
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Step 1: Load environment
            if not self._load_environment():
                return 1
            
            # Step 2: Initialize exchange
            if not self._initialize_exchange_connection():
                return 1
            
            # Step 3: Initialize risk management
            if not self._initialize_risk_management():
                return 1
            
            # Step 4: Initialize AI components (non-critical)
            await self._initialize_ai_components()
            
            # Step 5: Initialize strategies
            if not await self._initialize_strategies():
                return 1
            
            # Step 6: Initialize production system (includes WebSocket)
            if not await self._initialize_production_system():
                return 1
            
            # Step 7: Register strategies
            if not await self._register_strategies():
                return 1
            
            # Step 8: Pre-flight checks
            if not await self._perform_preflight_checks():
                logger.error("\n❌ Pre-flight checks failed - aborting launch")
                return 1
            
            # Print configuration summary after initialization
            self._print_configuration_summary()
            
            # If dry-run, stop here
            if self.dry_run:
                logger.info("\n✓ Dry run completed successfully - no trading started")
                return 0
            
            # Start trading loop
            await self._start_trading_loop(duration)
            
            return 0
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Interrupted by user (Ctrl+C)")
            return 130  # Standard exit code for Ctrl+C
            
        except Exception as e:
            logger.critical(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            # ✅ ALWAYS cleanup, even on error!
            logger.info("Performing cleanup...")
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"❌ Cleanup failed: {e}")
    
    async def _run_with_auto_restart(self, duration: Optional[float] = None) -> int:
        """
        Run trading system with auto-restart failsafe (Layer 2 Defense).
        
        [... mevcut kod ...]
        """
        logger.info("\n" + "="*70)
        logger.info("ULTIMATE CONTINUOUS MODE: AUTO-RESTART WRAPPER ACTIVE")
        logger.info("="*70)
        
        # Guard clause: Check if restart_manager is initialized
        if self.restart_manager is None:
            logger.critical("❌ Auto-restart manager is not initialized. Check --auto-restart flag.")
            logger.info("Falling back to normal execution mode...")
            return await self._run_once(duration)
        
        while True:
            # Check if we should attempt restart
            should_restart, reason = self.restart_manager.should_restart()
            
            if not should_restart:
                logger.critical(f"❌ Auto-restart disabled: {reason}")
                if self.telegram:
                    self.telegram.send(
                        f"🛑 <b>AUTO-RESTART STOPPED</b>\n"
                        f"Reason: {reason}\n"
                        f"Total Restarts: {self.restart_manager.restart_count}\n"
                        f"Manual intervention required"
                    )
                return 1
            
            try:
                logger.info(f"\n🚀 Starting bot (Attempt {self.restart_manager.restart_count + 1}/{self.restart_manager.max_restarts})")
                
                # Run the bot
                exit_code = await self._run_once(duration)
                
                # If exit was clean (0), record success
                if exit_code == 0:
                    logger.info("✓ Bot exited cleanly")
                    self.restart_manager.record_success()
                    
                    # In infinite mode, restart even on clean exit
                    if self.infinite:
                        logger.info("INFINITE MODE: Restarting after clean exit...")
                        self.restart_manager.record_failure("Clean exit in infinite mode")
                    else:
                        # Non-infinite mode with clean exit - stop here
                        return 0
                else:
                    # Exit code indicates failure
                    self.restart_manager.record_failure(f"Bot exited with code {exit_code}")
                
            except KeyboardInterrupt:
                logger.info("\n⚠ Keyboard interrupt - Manual stop requested")
                if self.telegram:
                    self.telegram.send("⛔ <b>Manual Stop</b> - Keyboard interrupt received")
                return 0
                
            except Exception as e:
                logger.error(f"❌ Bot crashed: {e}")
                self.restart_manager.record_failure(f"Exception: {str(e)[:100]}")
            
            # Calculate restart delay with exponential backoff
            delay = self.restart_manager.calculate_restart_delay()
            
            logger.warning("="*70)
            logger.warning(f"RESTARTING IN {delay} SECONDS...")
            logger.warning(f"Restart {self.restart_manager.restart_count}/{self.restart_manager.max_restarts}")
            logger.warning(f"Consecutive Failures: {self.restart_manager.consecutive_failures}")
            logger.warning("="*70)
            
            # Wait before restarting
            try:
                await asyncio.sleep(delay)
            except KeyboardInterrupt:
                logger.info("\n⚠ Keyboard interrupt during restart delay")
                return 0


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Bearish Alpha Bot - Live Trading Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start live trading with BingX
  python scripts/live_trading_launcher.py
  
  # Run in paper trading mode
  python scripts/live_trading_launcher.py --paper
  
  # Run for 1 hour (3600 seconds)
  python scripts/live_trading_launcher.py --duration 3600
  
  # Dry run (pre-flight checks only)
  python scripts/live_trading_launcher.py --dry-run
  
  # ULTIMATE MODE: True continuous trading (Layer 1 - never stops)
  python scripts/live_trading_launcher.py --infinite
  
  # ULTIMATE MODE: Auto-restart failsafe (Layer 2 - external monitoring)
  python scripts/live_trading_launcher.py --auto-restart
  
  # ULTIMATE MODE: Both layers enabled (maximum resilience)
  python scripts/live_trading_launcher.py --infinite --auto-restart
  
  # ULTIMATE MODE: Custom restart parameters
  python scripts/live_trading_launcher.py --infinite --auto-restart --max-restarts 500 --restart-delay 60
        """
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode (simulated trades)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Trading duration in seconds (default: indefinite)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform pre-flight checks only without starting trading'
    )
    
    parser.add_argument(
        '--infinite',
        action='store_true',
        help='Enable TRUE CONTINUOUS mode (Layer 1: never stops, auto-recovers from errors)'
    )
    
    parser.add_argument(
        '--auto-restart',
        action='store_true',
        help='Enable auto-restart failsafe (Layer 2: external monitoring and restart)'
    )
    
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=1000,
        help='Maximum restart attempts when auto-restart is enabled (default: 1000)'
    )
    
    parser.add_argument(
        '--restart-delay',
        type=int,
        default=30,
        help='Base delay between restarts in seconds (default: 30, uses exponential backoff)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with comprehensive logging for analysis'
    )
        
    args = parser.parse_args()
    
    # Determine mode
    mode = 'paper' if args.paper else 'live'
    
    launcher = None
    exit_code = 0
    
    try:
        logger.info("=" * 70)
        logger.info("BEARISH ALPHA BOT - STARTING")
        logger.info("=" * 70)
        
        # Create launcher
        launcher = LiveTradingLauncher(
            mode=mode, 
            dry_run=args.dry_run,
            infinite=args.infinite,
            auto_restart=args.auto_restart,
            max_restarts=args.max_restarts,
            restart_delay=args.restart_delay,
            debug_mode=args.debug
        )
        
        # Run launcher
        exit_code = await launcher.run(duration=args.duration)
        
        logger.info("✅ Trading completed successfully")
    
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupted by user (Ctrl+C)")
        exit_code = 130  # Standard exit code for Ctrl+C
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    finally:
        # ✅ ALWAYS cleanup, even on error!
        if launcher:
            logger.info("Performing final cleanup...")
            try:
                await launcher.cleanup()
            except Exception as e:
                logger.error(f"❌ Final cleanup failed: {e}")
                if exit_code == 0:
                    exit_code = 1
        
        logger.info("=" * 70)
        logger.info(f"👋 Bot shutdown complete (exit code: {exit_code})")
        logger.info("=" * 70)
    
    return exit_code


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
