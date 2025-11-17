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

# Add src to path BEFORE importing from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import logging
import argparse
import time
import inspect
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Ensure ML stack stays enabled when launch scripts forget to export the flag.
os.environ.setdefault('ML_ENABLED', 'true')

from core.logger import setup_logger

# ====================================================================
# ===             YENİ VE MERKEZİ YAPIYI KULLANMA                  ===
# ====================================================================
# Artık yapılandırmayı almak için tek bir standart yolumuz var.
from config.live_trading_config import LiveTradingConfiguration
# ====================================================================

from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient
from core.notify import Telegram
from core.system_info import SystemInfoCollector, format_startup_header
from config.risk_config import RiskConfiguration
from config.optimization_config import OptimizationConfiguration
from ml.regime_predictor import MLRegimePredictor
# --- GÜNCELLENDİ: Gerçek model sınıflarını da import et ---
from ml.price_predictor import AdvancedPricePredictionEngine
from ml.strategy_integration import AIEnhancedStrategyAdapter
from ml.strategy_optimizer import StrategyOptimizer
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip
from core.indicator_validator import IndicatorValidator

# Debug modu ortam değişkenine göre log seviyesini ayarla
is_debug = os.getenv('DEBUG_STRATEGY_LOGGING', 'false').lower() == 'true'
# Merkezi logger'ı çağır. Bu, tüm uygulama için loglamayı başlatacak.
logger = setup_logger("bearish-alpha-bot", debug_mode=is_debug, log_to_file=True)

import sentry_sdk

sentry_sdk.init(
    dsn="https://dec4fa87a85bf839cdd318be02111404@o4510318189281280.ingest.de.sentry.io/4510318291583056",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
    # Enable sending logs to Sentry
    enable_logs=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profile_session_sample_rate to 1.0 to profile 100%
    # of profile sessions.
    profile_session_sample_rate=1.0,
)

def slow_function():
    import time
    time.sleep(0.1)
    return "done"

def fast_function():
    import time
    time.sleep(0.05)
    return "done"

# Manually call start_profiler and stop_profiler
# to profile the code in between
sentry_sdk.profiler.start_profiler()

for i in range(0, 10):
    slow_function()
    fast_function()

# Calls to stop_profiler are optional - if you don't stop the profiler, it will keep profiling
# your application until the process exits or stop_profiler is called.
sentry_sdk.profiler.stop_profiler()

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
        """Recursively coerce placeholder type-name strings to safe Python types.

        If the configuration contains placeholder names like 'dict'/'list' etc,
        return the Python type object (dict, list, int ...) so downstream
        isinstance(x, config_value) uses a valid type as the second arg.
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

    def _parse_stream_timeframes(self) -> List[str]:
        """
        Normalize timeframe list from config (supports:
        - '1m,5m,15m,30m,1h,4h'
        - ['1m,5m,15m,30m,1h,4h']
        - ['1m','5m','15m','30m','1h','4h'])
        """
        try:
            raw = (self.config.get('websocket') or {}).get('stream_timeframes', None) \
                if isinstance(self.config, dict) else None
            
            # ✅ ÇÖZÜM: '15m' eksikti, eklendi.
            default_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']

            if raw is None:
                return default_timeframes
            if isinstance(raw, str):
                return [x.strip() for x in raw.split(',') if x.strip()]
            if isinstance(raw, list):
                if len(raw) == 1 and isinstance(raw[0], str) and ',' in raw[0]:
                    return [x.strip() for x in raw[0].split(',') if x.strip()]
                return [str(x).strip() for x in raw if str(x).strip()]
            return default_timeframes
        except Exception:
            return ['1m', '5m', '15m', '30m', '1h', '4h'] # ✅ Hata durumunda da '15m' içerdiğinden emin ol.

    def _normalize_ccxt_futures_symbols(self, symbols: List[str]) -> List[str]:
        """
        Ensure CCXT futures format ':USDT' for USDT perpetuals.
        'BTC/USDT' -> 'BTC/USDT:USDT', preserve if already contains ':'.
        """
        norm: List[str] = []
        for s in symbols or []:
            s = (s or '').strip().upper()
            if not s:
                continue
            if ':' in s:
                norm.append(s)
            elif s.endswith('/USDT'):
                norm.append(f"{s}:USDT")
            else:
                norm.append(s)
        return norm

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
                # Fallback coerce function mapping placeholders to types
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

            try:
                from core.websocket_manager import WebSocketManager
            except Exception:
                logger.debug("[WS-OPT] core.websocket_manager not available; skipping WebSocket setup")
                return []

            try:
                self.ws_manager = WebSocketManager(
                    exchanges=exchange_clients,
                    config=self.config
                )
            except Exception as e:
                logger.error(f"[WS-OPT] WebSocketManager init failed: {e}")
                return []         

            # Start streams (single path)
            tasks = await self._subscribe_optimized()
            if tasks:
                logger.info(f"[WS-OPT] ✅ WebSocket initialized with {len(tasks)} streams")
                self.is_initialized = True
            else:
                logger.warning("[WS-OPT] No WebSocket streams started")
            return tasks

        except Exception as e:
            logger.error(f"[WS-OPT] Failed to initialize WebSocket: {e}", exc_info=True)
            return []
    
    async def _subscribe_optimized(self) -> List[asyncio.Task]:
        if not self.ws_manager:
            return []

        tasks: List[asyncio.Task] = []
        stream_count: Dict[str, int] = {}

        # Normalize TFs and symbols
        timeframes = self._parse_stream_timeframes()
        self.fixed_symbols = self._normalize_ccxt_futures_symbols(self.fixed_symbols)
        logger.info(f"[WS-OPT] Parsed timeframes: {timeframes}")

        for exchange_name, _client in self.ws_manager.clients.items():
            max_streams = self.max_streams_config.get(exchange_name, self.max_streams_config.get('default', 10))
            streams_per_symbol = max(1, len(timeframes))
            max_symbols = max(1, max_streams // streams_per_symbol)

            exchange_symbols = self.fixed_symbols[:max_symbols]
            if not exchange_symbols:
                logger.warning(f"[WS-OPT] {exchange_name}: No symbols under limit={max_streams} with TFs={len(timeframes)}")
                continue

            symbols_per_exchange = {exchange_name: exchange_symbols}
            for tf in timeframes:
                ohlcv_tasks = await self.ws_manager.stream_ohlcv(
                    symbols_per_exchange=symbols_per_exchange,
                    timeframe=tf,
                    callback=None,
                    max_iterations=None
                )
                tasks.extend(ohlcv_tasks)

            stream_count[exchange_name] = len(exchange_symbols) * len(timeframes)
            logger.info(f"[WS-OPT] {exchange_name}: Subscribed to {len(exchange_symbols)} symbols across {len(timeframes)} TFs (limit={max_streams})")

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
    
    def _convert_symbol_for_exchange(self, symbol: str, exchange: str = 'bingx') -> str:
        """
        Convert CCXT symbol format to exchange-specific format.
        ✅ FIX 3: Symbol format conversion for BingX
        
        Args:
            symbol: CCXT format symbol (e.g., 'BTC/USDT:USDT')
            exchange: Target exchange name
            
        Returns:
            Exchange-specific symbol format
        """
        if exchange.lower() == 'bingx':
            # BTC/USDT:USDT -> BTC-USDT
            base_symbol = symbol.split(':')[0] if ':' in symbol else symbol
            return base_symbol.replace('/', '-')
        
        # Default: return as-is
        return symbol
    
    async def initialize_and_subscribe(self, exchange_clients: Dict[str, Any], symbols: List[str]) -> bool:
        try:
            logger.info("[WS-INIT] Starting WebSocket initialization and subscription...")

            # 1) Setup configuration
            self.setup_from_config(self.config)

            # Normalize incoming symbols for verification consistency
            symbols = self._normalize_ccxt_futures_symbols(symbols)

            # 2) Initialize + start streams
            logger.info("[WS-INIT] Initializing WebSocket connections...")
            tasks = await self.initialize_websockets(exchange_clients)
            if not tasks:
                logger.error("[WS-INIT] ❌ Failed to initialize WebSocket connections")
                return False

            logger.info(f"[WS-INIT] ✅ Initialized {len(tasks)} WebSocket tasks")

            # 3) Client health verification (≤20s)
            if not self.ws_manager or not getattr(self.ws_manager, "clients", None):
                logger.error("[WS-VERIFY] ❌ No WebSocket clients available")
                return False

            if "bingx" in self.ws_manager.clients:
                client = self.ws_manager.clients["bingx"]
                client_name = "bingx"
            else:
                client_name, client = next(iter(self.ws_manager.clients.items()))

            max_health_wait_s = 30  # Increased from 20 to 30 seconds
            health_ok = False
            for sec in range(max_health_wait_s):
                try:
                    health = client.get_health_status()
                except Exception as e:
                    logger.debug(f"[WS-VERIFY] Health status error: {e}")
                    health = {}

                connected = bool(health.get("connected"))
                listen = health.get("listen_task_status", "unknown")
                subs = int(health.get("subscriptions", 0)) if isinstance(health.get("subscriptions"), int) else 0
                msg_count = int(health.get("message_count", 0)) if isinstance(health.get("message_count"), int) else 0

                logger.info(f"[WS-VERIFY][{client_name}] t+{sec:02d}s connected={connected} listen={listen} subs={subs} messages={msg_count}")

                # More lenient health check: Accept if connected OR has messages
                if connected or msg_count > 0 or (listen == "running" and subs > 0):
                    health_ok = True
                    break
                await asyncio.sleep(1.0)

            if not health_ok:
                logger.warning("[WS-VERIFY] ⚠️ Client health not fully established, but proceeding with fallback support")
                # Don't fail - allow system to continue with REST API fallback
                self.is_initialized = True
                return True

            # 4) Collector verification (multi-TF, retry) - with increased tolerance
            tfs = self._parse_stream_timeframes()

            def tf_sort_key(tf: str) -> int:
                order = {"1m": 0, "3m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5, "4h": 6, "1d": 7}
                return order.get(tf, 99)

            tfs_sorted = sorted(tfs, key=tf_sort_key)
            logger.info(f"[WS-VERIFY] Verifying collector data for TFs={tfs_sorted}")

            max_checks = 5  # Increased from 3 to 5 attempts
            sleep_between = 5.0  # Increased from 4.0 to 5.0 seconds
            verified_symbols = set()

            for attempt in range(1, max_checks + 1):
                if attempt > 1:
                    logger.info(f"[WS-VERIFY] Attempt {attempt}/{max_checks} - waiting {sleep_between:.1f}s...")
                    await asyncio.sleep(sleep_between)

                for sym in symbols[:3]:
                    if sym in verified_symbols:
                        continue
                    for tf in tfs_sorted:
                        try:
                            data = self.ws_manager.get_latest_data(sym, tf)
                            if data and data.get('ohlcv'):
                                logger.info(f"[WS-VERIFY] ✅ Collector data confirmed for {sym} [{tf}] (candles={len(data['ohlcv'])})")
                                verified_symbols.add(sym)
                                break
                        except Exception as e:
                            logger.debug(f"[WS-VERIFY] Error checking {sym} {tf}: {e}")

                if verified_symbols:
                    break

            if verified_symbols:
                logger.info(f"[WS-VERIFY] ✅ WebSocket data flow verified for {len(verified_symbols)}/{min(3, len(symbols))} symbols")
                self.is_initialized = True
                return True

            # CRITICAL FIX: Always proceed even if collector verification fails
            # The system will fall back to REST API for data
            logger.warning("[WS-VERIFY] ⚠️ WebSocket collector data not verified yet")
            logger.warning("[WS-VERIFY] ⚠️ System will use REST API fallback for market data")
            logger.info("[WS-VERIFY] ✅ Proceeding with initialization despite WebSocket data verification failure")
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"[WS-ERROR] Failed to initialize and subscribe: {e}", exc_info=True)
            return False
        
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
        
        # Health report file path
        ts = self.start_time.strftime('%Y%m%d_%H%M%S')
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.health_log_path = os.path.join(log_dir, f'health_{ts}.json')
        
        logger.info("="*70)
        logger.info("HEALTH MONITORING SYSTEM INITIALIZED (Layer 3 Guardian)")
        logger.info("="*70)
        logger.info(f"Heartbeat Interval: {self.heartbeat_interval}s")
        logger.info(f"Health Report Path: {self.health_log_path}")
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
        
        # Write final health report
        final_snapshot = self.get_health_report()
        self._write_health_report(snapshot=final_snapshot, final=True)
        
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
                
                # Write periodic health report
                snapshot = self.get_health_report()
                self._write_health_report(snapshot=snapshot, final=False)
                
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
    
    def _write_health_report(self, snapshot: Optional[Dict[str, Any]] = None, final: bool = False):
        """
        Write health report to JSON file.
        
        Args:
            snapshot: Optional health snapshot to write. If None, generates one from get_health_report()
            final: Whether this is a final report (on shutdown)
        """
        try:
            if snapshot is None:
                snapshot = self.get_health_report()
            
            # Add metadata
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'report_type': 'final' if final else 'periodic',
                'health': snapshot
            }
            
            # Write to file
            with open(self.health_log_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            if final:
                logger.info(f"Final health report written to {self.health_log_path}")
            else:
                logger.debug(f"Health report updated: {self.health_log_path}")
                
        except Exception as e:
            logger.warning(f"Failed to write health report: {e}")


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

    def _load_config(
        self,
        *,
        force_reload: bool = False,
        log_summary: bool = True
    ) -> Dict[str, Any]:
        """Load the centralized configuration via the unified loader."""
        needs_refresh = force_reload or getattr(self, 'config', None) is None
        if needs_refresh:
            self.config = LiveTradingConfiguration.load(
                log_summary=log_summary,
                force_reload=force_reload
            )
        return self.config
    
    # Default risk parameters - used across normalization and fallbacks
    DEFAULT_RISK_PARAMS = {
        'max_position_size': 0.2,
        'stop_loss_pct': None,      # Sentinel value - will be calculated dynamically
        'take_profit_pct': None,    # Sentinel value - will be calculated dynamically
        'risk_per_trade': 0.05,
        'max_drawdown': 0.05
    }
    
    def __init__(self, mode: str, dry_run: bool, infinite: bool, auto_restart: bool,
                 max_restarts: int, restart_delay: int, debug_mode: bool):

        # 1. Gelen argümanları doğrudan sınıf değişkenlerine ata
        self.mode = mode
        self.dry_run = dry_run
        self.infinite = infinite
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.debug_mode = debug_mode

        # 2. YAPILANDIRMAYI MERKEZDEN VE TEK SEFERDE AL (TEK DOĞRU KAYNAK)
        self.config = self._load_config(log_summary=True)
        
        # 3. Gerekli tüm parametreleri DOĞRUDAN bu tek, güvenilir kaynaktan al.
        #    _load_config() metodu, sembollerin her zaman bir LİSTE olmasını garanti eder.
        self.CAPITAL_USDT = self.config.get('risk', {}).get('equity_usd', 100.0)
        self.TRADING_PAIRS = self.config.get('universe', {}).get('fixed_symbols', [])

        # ✅ ADD THIS LINE FOR COMPATIBILITY
        self.trading_pairs = self.TRADING_PAIRS  # Lowercase alias for compatibility
        
        # DEFENSIVE: Ensure TRADING_PAIRS is always a list (last-resort safety check)
        # The config module should already return a list, but this handles edge cases
        if not isinstance(self.TRADING_PAIRS, list):
            logger.warning(
                f"⚠️ TRADING_PAIRS has unexpected type {type(self.TRADING_PAIRS).__name__}. "
                f"Converting to list. Value: {self.TRADING_PAIRS}"
            )
            if isinstance(self.TRADING_PAIRS, str):
                # Use config module's parsing logic
                if LiveTradingConfiguration._is_trading_symbol(self.TRADING_PAIRS):
                    self.TRADING_PAIRS = LiveTradingConfiguration._parse_trading_symbols(self.TRADING_PAIRS)
                else:
                    self.TRADING_PAIRS = ['BTC/USDT']  # Safe fallback
            else:
                self.TRADING_PAIRS = ['BTC/USDT']  # Safe fallback
        
        self.RISK_PARAMS = self.config.get('risk', {})

        # 4. Diğer tüm başlangıç değişkenlerini boş olarak başlat
        self.coordinator = None
        self.telegram = None
        self.exchange_clients = {}
        self.strategies = {}
        self.restart_manager = None
        self.health_monitor = None
        self.ws_optimizer = None
        self._cleanup_completed = False
        self._has_bingx_credentials = False
        self._cached_exchange_status = None
        self._cached_ws_status = None
        # ML bileşenleri daha sonra yüklenecek
        self.regime_predictor = None
        self.price_engine = None
        # CRITICAL: Initialize all task attributes to prevent AttributeError during cleanup
        self._main_trading_task = None
        self._prediction_loop_task = None
        self._websocket_task = None
        self._heartbeat_task = None
        self._monitoring_task = None

        # 5. Başlangıç loglarını, YENİ ve DOĞRU verilerle yazdır
        logger.info("="*70)
        logger.info("BEARISH ALPHA BOT - LAUNCHER (v3.2 - Final Config)")
        logger.info("="*70)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info(f"Capital: {self.CAPITAL_USDT} USDT (from config)")
        logger.info(f"Exchange: BingX")
        
        # Bu loglama artık her zaman doğru çalışacak
        if self.TRADING_PAIRS:
            logger.info(f"Trading Pairs ({len(self.TRADING_PAIRS)}): {', '.join(self.TRADING_PAIRS)}")
        else:
            logger.warning("⚠️ No trading pairs configured! This will likely cause an error.")

    def _normalize_risk_params(self):
        """
        Normalize risk parameters with support for dynamic (sentinel) values.
        Maps all variations to standard keys while preserving None for dynamic params.
        """
        if not hasattr(self, 'RISK_PARAMS'):
            self.RISK_PARAMS = {}
        
        logger.debug(f"Current RISK_PARAMS keys: {list(self.RISK_PARAMS.keys())}")
        
        # Map all possible variations to standard keys
        # CRITICAL FIX: Remove max_notional_per_trade from max_position_size mapping
        key_mappings = {
            'max_position_size': ['max_position_size_pct'],  # FIXED: Removed 'max_notional_per_trade'
            'stop_loss_pct': ['stop_loss', 'stop_loss_multiplier'],  # Separated from min_stop_pct 
            'take_profit_pct': ['take_profit', 'take_profit_ratio'],  # Separated from min_tp_pct
            'min_stop_pct': ['min_stop_pct'],  # Now a separate parameter (not merged with stop_loss_pct)
            'risk_per_trade': ['per_trade_risk_pct'],
            'max_drawdown': ['daily_loss_limit_pct', 'max_daily_loss'],
            # NEW mappings for dynamic sizing
            'max_notional_pct': ['max_notional_pct_per_trade'],
            'max_margin_pct': ['max_margin_pct_per_trade'],
        }
        
        # Check config if available
        config_trading = self.config.get('trading', {}) if hasattr(self, 'config') else {}
        config_risk = self.config.get('risk', {}) if hasattr(self, 'config') else {}
        
        for standard_key, variations in key_mappings.items():
            found = False
            
            # Preserve existing standard key if already set
            if standard_key in self.RISK_PARAMS:
                continue
            
            # ===== SENTINEL VALUE SUPPORT =====
            # For take_profit_pct and stop_loss_pct, allow None as a valid value
            is_dynamic_param = standard_key in ['take_profit_pct', 'stop_loss_pct']
            
            # Check RISK_PARAMS first
            for variant in variations:
                if variant in self.RISK_PARAMS:
                    self.RISK_PARAMS[standard_key] = self.RISK_PARAMS[variant]
                    found = True
                    break
            
            # Check config if not found
            if not found:
                for variant in variations:
                    if variant in config_trading:
                        self.RISK_PARAMS[standard_key] = config_trading[variant]
                        found = True
                        break
                    elif variant in config_risk:
                        self.RISK_PARAMS[standard_key] = config_risk[variant]
                        found = True
                        break
            
            # Check environment variables if still not found
            if not found:
                env_names = [v.upper() for v in variations]
                for env_name in env_names:
                    env_val = os.getenv(env_name)
                    if env_val:
                        try:
                            self.RISK_PARAMS[standard_key] = float(env_val)
                            found = True
                            break
                        except ValueError:
                            logger.warning(f"Invalid value for environment variable '{env_name}': '{env_val}' (expected float)")
            
            # ===== KEY CHANGE: Handle missing params differently =====
            if not found:
                if is_dynamic_param:
                    # For dynamic params, use None (sentinel value)
                    self.RISK_PARAMS[standard_key] = None
                    logger.info(f"✓ Risk param '{standard_key}' will be calculated dynamically by strategies")
                else:
                    # For other params, use defaults
                    self.RISK_PARAMS[standard_key] = self.DEFAULT_RISK_PARAMS[standard_key]
                    logger.warning(f"Risk param '{standard_key}' not found, using default: {self.DEFAULT_RISK_PARAMS[standard_key]}")
        
        # CRITICAL: Add safety check for max_position_size
        if 'max_position_size' in self.RISK_PARAMS and self.RISK_PARAMS['max_position_size'] is not None:
            value = self.RISK_PARAMS['max_position_size']
            if value > 1.0:
                logger.warning(f"⚠️ max_position_size={value} > 1.0, converting to ratio")
                self.RISK_PARAMS['max_position_size'] = min(value / 100.0, 0.20)  # Cap at 20%
                logger.info(f"✅ Corrected max_position_size to {self.RISK_PARAMS['max_position_size']:.2%}")
        
        # Validate risk parameter values (skip None values)
        for key, value in self.RISK_PARAMS.items():
            if value is None:
                # Skip validation for sentinel values
                continue
                
            if key in self.DEFAULT_RISK_PARAMS:  # Only validate known risk params
                if value < 0:
                    logger.error(f"Invalid risk param '{key}': {value} (negative value not allowed). Using default.")
                    self.RISK_PARAMS[key] = self.DEFAULT_RISK_PARAMS[key]
                elif value > 1.0 and key != 'max_position_size':  # Skip max_position_size as we handle it above
                    logger.warning(f"Risk param '{key}': {value:.1%} (> 100%). This may be intentional for leverage.")
                elif key == 'max_position_size' and value > 0.5:
                    logger.warning(f"Risk param 'max_position_size': {value:.1%} (> 50%). This is quite high.")
        
        logger.info("Risk parameters normalized successfully")
        
        # Log final state with special handling for None values
        for key, value in self.RISK_PARAMS.items():
            if value is None:
                logger.debug(f"  {key}: <dynamic>")
            else:
                logger.debug(f"  {key}: {value}")

    @property
    def capital_source(self) -> str:
        """Return the source used for resolving capital."""

        return self._capital_source
    
    async def cleanup(self, signum=None, frame=None):
        """
        Graceful shutdown procedure in the CRITICAL CORRECT ORDER.
        
        SHUTDOWN ORDER (DO NOT CHANGE):
        1. Stop trading loop - prevents new signals from being processed
        2. Close all positions - MUST happen while exchange connections are ALIVE
        3. Stop WebSocket streams - can now safely disconnect
        4. Close exchange connections - final cleanup
        
        This order ensures positions are closed successfully before connections die.
        """
        if self._cleanup_completed:
            logger.info("Cleanup already completed, skipping.")
            return

        logger.info("\n" + "="*70)
        logger.info("🧹 STARTING GRACEFUL SHUTDOWN")
        logger.info("="*70)
        logger.info("CRITICAL: Following correct shutdown order to prevent orphaned positions")
        logger.info("="*70)

        errors = []
        self._cleanup_completed = True

        # ========================================================================
        # STEP 1: Stop Trading Loop (Prevent New Signals)
        # ========================================================================
        logger.info("\nStep 1: Stopping main trading loop (no new signals)...")
        if self.coordinator:
            try:
                await self.coordinator.stop()
                logger.info("✅ Main trading loop stopped - no new signals will be processed")
            except Exception as e:
                errors.append(f"Error stopping coordinator: {e}")
                logger.error(f"Error stopping coordinator: {e}", exc_info=True)
        
        # Stop price prediction loop
        logger.info("\nStopping price prediction loop...")
        if hasattr(self, '_prediction_loop_task') and self._prediction_loop_task:
            try:
                if self.price_engine:
                    await self.price_engine.stop_prediction_loop()
                self._prediction_loop_task.cancel()
                try:
                    await self._prediction_loop_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Price prediction loop stopped")
            except Exception as e:
                errors.append(f"Error stopping prediction loop: {e}")
                logger.error(f"Error stopping prediction loop: {e}", exc_info=True)

        # ========================================================================
        # STEP 2: Close All Open Positions (CRITICAL - Must happen BEFORE closing connections)
        # ========================================================================
        logger.info("\nStep 2: Closing all open positions (exchange connections ALIVE)...")
        if self.coordinator and hasattr(self.coordinator, 'position_manager') and self.coordinator.position_manager:
            try:
                # Get position count before closing
                positions = getattr(self.coordinator.position_manager, 'positions', {})
                position_count = len(positions)
                
                if position_count > 0:
                    logger.info(f"🔄 Attempting to close {position_count} open position(s)...")
                else:
                    logger.info("ℹ️ No open positions to close")
                
                # CRITICAL FIX: Pass LIVE exchange_clients to ensure positions can be closed
                logger.info(f"🔑 Injecting {len(self.exchange_clients)} live exchange client(s) for position closure")
                result = await self.coordinator.position_manager.close_all_positions(
                    exchange_clients=self.exchange_clients,  # *** CRITICAL: Pass live clients ***
                    reason="shutdown"
                )
                logger.info(f"✅ Position closure completed. Result: {result}")
                
                # Verify positions were closed
                remaining = len(getattr(self.coordinator.position_manager, 'positions', {}))
                if remaining > 0:
                    logger.warning(f"⚠️ Warning: {remaining} position(s) may still be open")
                else:
                    logger.info("✅ All positions successfully closed")
                    
            except Exception as e:
                error_msg = f"Critical error during position closure: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                logger.error("❌ CRITICAL: Positions may still be open on the exchange!")
        else:
            logger.info("ℹ️ Position manager not available, skipping position closure")

        # ========================================================================
        # STEP 3: Stop WebSocket Streams (Safe to disconnect now)
        # ========================================================================
        logger.info("\nStep 3: Stopping WebSocket streams...")
        if self.ws_optimizer:
            try:
                await self.ws_optimizer.stop_streaming()
                logger.info("✅ WebSocket streams stopped")
            except Exception as e:
                errors.append(f"Error stopping WebSocket: {e}")
                logger.error(f"Error stopping WebSocket: {e}", exc_info=True)
        
        # ========================================================================
        # STEP 4: Stop Health Monitor
        # ========================================================================
        if self.health_monitor:
            logger.info("\nStep 4: Stopping health monitor...")
            try:
                await self.health_monitor.stop_monitoring()
                logger.info("✅ Health monitor stopped")
            except Exception as e:
                errors.append(f"Error stopping health monitor: {e}")
                logger.error(f"Error stopping health monitor: {e}", exc_info=True)

        # ========================================================================
        # STEP 5: Close Exchange Connections (Final cleanup)
        # ========================================================================
        logger.info("\nStep 5: Closing exchange connections...")
        if self.exchange_clients:
            for name, client in self.exchange_clients.items():
                try:
                    if hasattr(client, 'close'):
                        await client.close()
                    logger.info(f"✅ {name} exchange connection closed")
                except Exception as e:
                    errors.append(f"Failed to close {name}: {e}")
                    logger.error(f"Failed to close {name}: {e}", exc_info=True)
        
        # ========================================================================
        # Shutdown Summary
        # ========================================================================
        logger.info("\n" + "="*70)
        if not errors:
            logger.info("✅ GRACEFUL SHUTDOWN COMPLETED SUCCESSFULLY")
            logger.info("✅ All positions closed, all connections terminated")
        else:
            logger.warning(f"⚠️ GRACEFUL SHUTDOWN COMPLETED WITH {len(errors)} ERROR(S)")
            for i, err in enumerate(errors, 1):
                logger.warning(f"  - Error {i}: {err}")
        logger.info("="*70)
    
    def _load_environment(self) -> bool:
        """
        Load and validate environment variables.
        
        Returns:
            True if all required variables are present
        """
        logger.info("\n[1/8] Loading Environment Configuration...")
        
        required_vars = ['BINGX_KEY', 'BINGX_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        self._has_bingx_credentials = not missing_vars

        if self.mode == 'live' and not self.dry_run and not self._has_bingx_credentials:
            logger.error(f"❌ Missing required environment variables for LIVE mode: {missing_vars}")
            return False
        
        if self._has_bingx_credentials:
            logger.info("✓ BingX credentials found")
        else:
            logger.info("ℹ️ BingX credentials not provided (OK for dry-run or paper mode)")

        # Optional Telegram setup
        tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        tg_chat = os.getenv('TELEGRAM_CHAT_ID')
        
        if tg_token and tg_chat:
            self.telegram = Telegram(tg_token, tg_chat)
            logger.info("✓ Telegram notifications enabled")
        else:
            logger.info("ℹ️  Telegram notifications disabled (optional)")
        
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
            
            # Safe extraction with fallbacks
            max_pos = self.RISK_PARAMS.get('max_position_size', self.DEFAULT_RISK_PARAMS['max_position_size'])
            stop_loss = self.RISK_PARAMS.get('stop_loss_pct', self.DEFAULT_RISK_PARAMS['stop_loss_pct'])
            take_profit = self.RISK_PARAMS.get('take_profit_pct', self.DEFAULT_RISK_PARAMS['take_profit_pct'])
            max_dd = self.RISK_PARAMS.get('max_drawdown', self.DEFAULT_RISK_PARAMS['max_drawdown'])
            
            logger.info(f"  - Max position size: {max_pos:.1%}")
            logger.info(f"  - Stop loss: {stop_loss:.1%}")
            logger.info(f"  - Take profit: {take_profit:.1%}")
            logger.info(f"  - Max drawdown: {max_dd:.1%}")
            
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
    
    async def _initialize_strategies(self) -> bool:
        """Initialize adaptive trading strategies from the pre-loaded config."""
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # Yapılandırma zaten __init__ içinde yüklendi. Sadece kullan.
            if not self.config:
                logger.error("❌ Cannot initialize strategies, config is not loaded.")
                return False

            logger.info("✓ Using pre-loaded centralized configuration.")
            
            # Log active configuration
            symbols = self.config.get('universe', {}).get('fixed_symbols', [])
            logger.info(f"✓ Trading symbols from config: {symbols}")
            
            # Initialize regime analyzer for adaptive strategies
            from core.market_regime import MarketRegimeAnalyzer
            regime_analyzer = MarketRegimeAnalyzer()
            
            # Strategy configurations FROM CONFIG FILE
            signals_config = self.config.get('signals', {})
    
            # 🔥 KALICI DÜZELTME: Artık manuel kopyalama yok. Stratejinin tüm config bloğunu alıyoruz.
            # Bu, `min_rr_ratio` dahil tüm ayarların stratejiye ulaşmasını garanti eder.
    
            # Adaptive OB config
            adaptive_ob_config = signals_config.get('oversold_bounce', {})
            if not adaptive_ob_config.get('enable', True):
                logger.info("⚠️ OversoldBounce strategy disabled in config")
            else:
                self.strategies['adaptive_ob'] = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
                logger.info(f"✓ Adaptive Oversold Bounce strategy initialized")
                logger.info(f"  - OB Config: { {k: v for k, v in adaptive_ob_config.items() if k != 'enable'} }")
            
            # Adaptive STR config
            adaptive_str_config = signals_config.get('short_the_rip', {})
            if not adaptive_str_config.get('enable', True):
                logger.info("⚠️ ShortTheRip strategy disabled in config")
            else:
                self.strategies['adaptive_str'] = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
                logger.info(f"✓ Adaptive Short The Rip strategy initialized")
                logger.info(f"  - STR Config: { {k: v for k, v in adaptive_str_config.items() if k != 'enable'} }")
    
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
    
    async def _initialize_production_system_core(self) -> bool:
        """
        Initialize CORE production systems only (Phase 1).
        This initializes exchange, WebSocket, data pipeline, risk, portfolio, trading engine.
        ML systems are initialized separately in Phase 2.
        """
        logger.info("\n[PHASE 1] Initializing Core Production Systems...")
        
        try:
            # 1. Prepare config
            if not self.config:
                self._load_config()
    
            # 2. Create RiskConfiguration from config file
            from config.risk_config import RiskConfiguration
            risk_params_from_config = self.config.get('risk', {})
            risk_config_object = RiskConfiguration(custom_limits=risk_params_from_config)
            logger.info("✓ RiskConfiguration created from config file")
            logger.info(f"  - Max Portfolio Risk: {risk_config_object.risk_limits.max_portfolio_risk}")
            logger.info(f"  - Max Position Size: {risk_config_object.risk_limits.max_position_size}")
            logger.info(f"  - Max Drawdown: {risk_config_object.risk_limits.max_drawdown}")
    
            # 3. Configure WebSocket Optimizer
            self.ws_optimizer.setup_from_config(self.config)
            
            # 4. Start WebSocket connections and streams
            logger.info("Starting WebSocket connections and data streams...")
            ws_success = await self.ws_optimizer.initialize_and_subscribe(
                self.exchange_clients,
                self.TRADING_PAIRS
            )
    
            if not ws_success:
                logger.warning("⚠️ [WS] WebSocket initialization was not successful, but we will proceed. The system may rely on REST API.")
    
            # 5. Create ProductionCoordinator
            from core.production_coordinator import ProductionCoordinator
            # ProductionCoordinator'a konfigürasyon nesnesini aktar
            self.coordinator = ProductionCoordinator(config=self.config)
            
            # 6. Initialize CORE systems with standardized risk configuration
            logger.info("Initializing core production systems...")
            core_result = await self.coordinator.initialize_core_systems(
                exchange_clients=self.exchange_clients,
                portfolio_value=self.CAPITAL_USDT,
                risk_config=risk_config_object,
                mode=self.mode,
                trading_symbols=self.TRADING_PAIRS,
                websocket_manager=self.ws_optimizer.ws_manager
            )
    
            if not core_result.get('success'):
                logger.error(f"❌ Core systems initialization failed: {core_result.get('reason')}")
                return False
    
            logger.info("✅ Core production systems initialized")
            logger.info(f"  Components: {', '.join(core_result.get('components', []))}")
            return True
    
        except Exception as e:
            logger.error(f"❌ Failed to initialize core production systems: {e}", exc_info=True)
            return False
    
    async def _wait_for_subscription_confirmations(self, timeout: int = 30) -> bool:
        """
        Wait for WebSocket subscriptions to be confirmed before health check.
        
        This fixes the race condition where health check runs before subscription
        confirmations arrive, causing false negatives.
        
        Args:
            timeout: Maximum seconds to wait (default: 30)
            
        Returns:
            True if subscriptions confirmed, False if timeout
        """
        logger.info(f"[SUBSCRIPTION-WAIT] Waiting up to {timeout}s for WebSocket subscription confirmations...")
        
        start_time = asyncio.get_event_loop().time()
        check_interval = 1.0  # Check every second
        last_log_time = 0  # Track when we last logged progress
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Check timeout
            if elapsed >= timeout:
                logger.warning(f"[SUBSCRIPTION-WAIT] ⏱️ Timeout after {elapsed:.1f}s")
                logger.warning("[SUBSCRIPTION-WAIT] Proceeding with health check despite timeout")
                return False
            
            # Check if subscriptions are active
            try:
                if self.ws_optimizer and self.ws_optimizer.ws_manager:
                    stream_count = self.ws_optimizer.ws_manager.get_active_stream_count()
                    
                    # Log progress every 5 seconds (only once per interval)
                    if int(elapsed) >= last_log_time + 5 and int(elapsed) > 0:
                        logger.info(f"[SUBSCRIPTION-WAIT] t+{int(elapsed)}s: {stream_count} active streams")
                        last_log_time = int(elapsed)
                    
                    # Success condition: at least one active stream
                    if stream_count > 0:
                        logger.info(f"[SUBSCRIPTION-WAIT] ✅ Subscriptions confirmed after {elapsed:.1f}s ({stream_count} streams)")
                        return True
                else:
                    logger.warning("[SUBSCRIPTION-WAIT] WebSocket manager not available")
                    return False
            except Exception as e:
                logger.warning(f"[SUBSCRIPTION-WAIT] Error checking subscriptions: {e}")
            
            # Wait before next check
            await asyncio.sleep(check_interval)
    
    async def _perform_data_health_check(self) -> bool:
        """
        Perform data layer health check (Phase 1.5).
        Validates that WebSocket connections are active and data is flowing.
        
        CRITICAL FIX (Issue #259 followup):
        - Health check gate now actually BLOCKS on failure
        - Waits for WebSocket subscription confirmations before checking
        - Prevents race condition where check runs before subscriptions confirm
        """
        logger.info("\n[PHASE 1.5] Performing Data Layer Health Check...")
        
        try:
            if not self.coordinator:
                logger.error("❌ Coordinator not initialized")
                return False
            
            # STEP 1: Wait for WebSocket subscriptions to confirm (fixes race condition)
            if self.ws_optimizer and self.ws_optimizer.ws_manager:
                logger.info("⏳ Waiting for WebSocket subscription confirmations...")
                await self._wait_for_subscription_confirmations(timeout=30)
            else:
                logger.info("ℹ️ No WebSocket manager - skipping subscription wait")
            
            # STEP 2: Perform health check
            health_result = await self.coordinator.is_data_layer_healthy()
            
            # Log detailed results
            logger.info("\n📊 Health Check Results:")
            for check_name, check_result in health_result.get('checks', {}).items():
                status = check_result.get('status', 'unknown')
                details = check_result.get('details', 'No details')
                
                if status == 'healthy':
                    logger.info(f"  ✅ {check_name}: {details}")
                elif status == 'degraded':
                    logger.warning(f"  ⚠️ {check_name}: {details}")
                elif status == 'not_available':
                    logger.info(f"  ℹ️ {check_name}: {details}")
                else:
                    logger.error(f"  ❌ {check_name}: {details}")
            
            # STEP 3: Enforce health check gate (CRITICAL FIX)
            # Previously always returned True - now properly enforces the gate
            is_healthy = health_result.get('healthy', False)
            
            if is_healthy:
                logger.info("\n✅ [HEALTH-CHECK] Data layer is HEALTHY")
                logger.info("   Proceeding to ML initialization")
                return True
            else:
                # CRITICAL CHANGE: Actually fail when unhealthy
                # This prevents ML phase from starting with broken data layer
                logger.error("\n❌ [HEALTH-CHECK] Data layer is UNHEALTHY")
                logger.error("   Cannot proceed to ML initialization")
                logger.error("   System requires working data layer for ML features")
                
                # Check individual components to provide actionable feedback
                checks = health_result.get('checks', {})
                if checks.get('websocket_connection', {}).get('status') == 'unhealthy':
                    logger.error("   - WebSocket connection failed")
                if checks.get('subscriptions', {}).get('status') == 'unhealthy':
                    logger.error("   - WebSocket subscriptions not active")
                if checks.get('data_flow', {}).get('status') in ['unhealthy', 'degraded']:
                    logger.error("   - Data flow not confirmed")
                
                return False  # GATE BLOCKS: Do not proceed to ML phase
                
        except Exception as e:
            logger.error(f"❌ Data health check failed with exception: {e}", exc_info=True)
            logger.error("   Cannot proceed to ML initialization due to health check failure")
            return False  # GATE BLOCKS on exception
    
    async def _initialize_production_system_ml(self) -> bool:
        """
        Initialize ML systems in production coordinator (Phase 2).
        This should only be called after core systems are initialized and data layer is healthy.
        """
        logger.info("\n[PHASE 2] Initializing ML Systems in Production Coordinator...")
        
        try:
            if not self.coordinator:
                logger.error("❌ Coordinator not initialized")
                return False
            
            # Initialize ML systems in coordinator
            ml_result = await self.coordinator.initialize_ml_systems(
                price_engine=self.price_engine,
                regime_predictor=self.regime_predictor
            )
            
            if ml_result.get('success'):
                logger.info("✅ ML systems initialized in coordinator")
                logger.info(f"  Components: {', '.join(ml_result.get('components', []))}")
                return True
            else:
                logger.warning(f"⚠️ ML initialization partial or failed: {ml_result.get('reason')}")
                logger.warning("   Continuing with limited ML features")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML systems: {e}", exc_info=True)
            return False
    
    async def _initialize_production_system(self) -> bool:
        """
        DEPRECATED: Use phased initialization methods instead.
        This method is kept for backward compatibility.
        
        Use instead:
        - _initialize_production_system_core() for Phase 1
        - _perform_data_health_check() for Phase 1.5
        - _initialize_production_system_ml() for Phase 2
        """
        logger.warning("⚠️ Using deprecated _initialize_production_system() method")
        logger.warning("   Consider using phased initialization methods instead")
        
        # Call phased methods in sequence
        if not await self._initialize_production_system_core():
            return False
        
        if not await self._perform_data_health_check():
            return False
        
        if not await self._initialize_production_system_ml():
            logger.warning("⚠️ ML initialization failed - continuing with limited features")
        
        return True
        
    async def _register_strategies(self) -> bool:
        """Initialize adaptive trading strategies."""
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # Config ZATEN yüklü olmalı
            if not self.config:
                self._load_config()
    
            # FIX: Use TRADING_PAIRS (uppercase) which exists
            trading_pairs = getattr(self, 'trading_pairs', self.TRADING_PAIRS)
            logger.info(f"✓ Using config with {len(trading_pairs)} symbols")
            
            # ❌ REMOVE THIS LINE - IT'S DUPLICATE AND WRONG
            # logger.info(f"✓ Using config with {len(self.trading_pairs)} symbols")
            
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
        Perform comprehensive pre-flight system checks with defensive programming.
        (GÜNCELLENDİ: API bağlantısı olmadan da çökmemesi sağlandı)
        
        Returns:
            True if all checks pass, False otherwise.
        """
        logger.info("\n[8/8] Performing Pre-Flight System Checks...")
        
        failed_checks = []
        
        # --- YENİ SAVUNMACI YAKLAŞIM: Mevcut borsa istemcisini en başta belirle ---
        # exchange_clients sözlüğündeki ilk (ve tek) istemcinin adını al.
        # Eğer sözlük boşsa, bu değişken None olacak ve sonraki adımlar bunu bilecek.
        available_exchange_name = next(iter(self.exchange_clients), None)
        # --- YENİ YAKLAŞIM SONU ---

        try:
            # Check 1: Exchange connectivity
            logger.info("Check 1/7: Exchange connectivity...")
            if available_exchange_name:
                try:
                    client = self.exchange_clients[available_exchange_name]
                    ticker = client.ticker('BTC/USDT')
                    logger.info(f"✓ BTC/USDT price via '{available_exchange_name}': ${ticker.get('last', 0):.2f}")
                    
                    # *** Cache successful exchange status ***
                    self._cached_exchange_status = {
                        'connected': True,
                        'status_emoji': '✅',
                        'status_text': 'CONNECTED',
                        'latency_ms': None,  # We don't measure latency here
                        'error': None
                    }
                except Exception as e:
                    logger.error(f"❌ Exchange connectivity failed: {e}")
                    failed_checks.append("Exchange connectivity")
                    
                    # *** Cache failed exchange status ***
                    self._cached_exchange_status = {
                        'connected': False,
                        'status_emoji': '❌',
                        'status_text': 'FAILED',
                        'latency_ms': None,
                        'error': str(e)
                    }
            else:
                logger.warning("⚠️ Exchange client not available, skipping connectivity check.")
                
                # *** Cache no exchange status ***
                self._cached_exchange_status = {
                    'connected': False,
                    'status_emoji': '❌',
                    'status_text': 'NO EXCHANGE CLIENT',
                    'latency_ms': None,
                    'error': 'No exchange clients configured'
                }
            
            # Check 2: System state
            logger.info("Check 2/7: System state...")
            state = self.coordinator.get_system_state()
            if state.get('is_initialized'):
                logger.info("✓ Production system initialized")
            else:
                logger.error("❌ Production system not initialized")
                failed_checks.append("System initialization")
            
            # Check 3: Risk limits
            logger.info("Check 3/7: Risk limits...")
            if self.coordinator.risk_manager:
                risk_summary = self.coordinator.risk_manager.get_portfolio_summary()
                logger.info(f"✓ Portfolio value: ${risk_summary.get('portfolio_value', 0):.2f}")
                logger.info("✓ Risk limits configured")
            else:
                logger.error("❌ Risk manager not available")
                failed_checks.append("Risk manager")
            
            # Check 4: Strategies
            logger.info("Check 4/7: Strategy registration...")
            if self.coordinator.portfolio_manager:
                strategies = self.coordinator.portfolio_manager.strategies
                logger.info(f"✓ {len(strategies)} strategies registered")
            else:
                logger.error("❌ Portfolio manager not available")
                failed_checks.append("Strategy registration")
            
            # Check 5: Emergency protocols
            logger.info("Check 5/7: Emergency shutdown protocols...")
            if self.coordinator.circuit_breaker:
                logger.info("✓ Circuit breaker active")
            else:
                logger.warning("⚠ Circuit breaker not available")
            
            # Check 6/7: WebSocket data flow (Bu blok büyük ölçüde aynı kalabilir)
            logger.info("Check 6/8: WebSocket data flow...")
            if self._is_ws_initialized() and self.ws_optimizer.ws_manager:
                timeframes = self.ws_optimizer._parse_stream_timeframes()
                symbols = self.ws_optimizer._normalize_ccxt_futures_symbols(self.TRADING_PAIRS)
                logger.info(f"  • Parsed timeframes: {timeframes}")

                working_symbols = 0
                for symbol in symbols[:3]: # Sadece ilk 3 sembolü kontrol et
                    if any(self.ws_optimizer.ws_manager.get_latest_data(symbol, tf) for tf in timeframes):
                        working_symbols += 1
                        logger.info(f"  ✅ {symbol}: Receiving data")
                    else:
                        logger.warning(f"  ⚠️ {symbol}: No data available across checked TFs")

                if working_symbols > 0:
                    logger.info(f"✅ WebSocket data flow confirmed for {working_symbols}/{min(3, len(symbols))} symbols")
                    
                    # *** Cache successful WebSocket status ***
                    stream_count = working_symbols  # Simplified - at least this many streams are working
                    self._cached_ws_status = {
                        'enabled': True,
                        'status_emoji': '✅',
                        'status_text': 'CONNECTED',
                        'stream_count': stream_count,
                        'mode': 'websocket'
                    }
                else:
                    logger.error("❌ WebSocket connected but no data is flowing for initial symbols")
                    failed_checks.append("WebSocket data flow")
                    
                    # *** Cache failed WebSocket status ***
                    self._cached_ws_status = {
                        'enabled': True,
                        'status_emoji': '⚠️',
                        'status_text': 'INITIALIZED (no data)',
                        'stream_count': 0,
                        'mode': 'rest_fallback'
                    }
            else:
                logger.error("❌ WebSocket not initialized or manager is missing")
                failed_checks.append("WebSocket initialization")
                
                # *** Cache disconnected WebSocket status ***
                self._cached_ws_status = {
                    'enabled': False,
                    'status_emoji': '⚠️',
                    'status_text': 'DISCONNECTED',
                    'stream_count': 0,
                    'mode': 'rest_fallback'
                }

            # Indicator Warmup Validation
            # CRITICAL FIX (Issue #259 followup): Removed duplicate prefetch
            # Historical data is already fetched during initialize_core_systems()
            # via market_data_pipeline.prime_data_buffers_async()
            # Calling prefetch_data() here was redundant and caused double data fetching
            logger.info("Check 7/8: Indicator Warmup Validation...")
            if not available_exchange_name:
                logger.warning("⚠️ Skipping Indicator Warmup Validation: No exchange client available.")
            elif not self.coordinator or not self.coordinator.trading_engine:
                reason = "Coordinator or Trading Engine not available for validation."
                logger.error(f"❌ {reason}")
                failed_checks.append(f"IndicatorValidator: {reason}")
            elif not self.ws_optimizer or not self.ws_optimizer.ws_manager or not self.ws_optimizer.ws_manager.collector:
                reason = "WebSocket Manager or Collector not available for validation."
                logger.error(f"❌ {reason}")
                failed_checks.append(f"IndicatorValidator: {reason}")
            else:
                try:
                    # REMOVED: Duplicate prefetch call
                    # OLD CODE: await self.coordinator.trading_engine.prefetch_data()
                    # Historical data was already fetched in initialize_core_systems()
                    
                    logger.info("  -> Validating indicator data (prefetch already completed in Phase 1)...")
                    # --- DÜZELTME 1: Validator'a artık ws_manager yerine doğrudan collector veriliyor ---
                    validator = IndicatorValidator(self.ws_optimizer.ws_manager.collector)
                    
                    # --- DÜZELTME 2: Yeni `validate_all` metodu çağrılıyor ---
                    timeframes = self.ws_optimizer._parse_stream_timeframes()
                    validation_results = await validator.validate_all(
                        symbols=self.TRADING_PAIRS,
                        timeframes=timeframes
                    )
                    
                    # Sonuçları kontrol et
                    all_valid = all(res.get('status') == 'OK' for res in validation_results.values())
                    
                    if not all_valid:
                        failed_reasons = [f"{s}: {r.get('reason', 'Unknown')}" for s, r in validation_results.items() if r.get('status') != 'OK']
                        reason = f"Indicator validation failed. Details: {'; '.join(failed_reasons)}"
                        logger.critical(f"❌ {reason}")
                        failed_checks.append(f"IndicatorValidator: {reason}")
                    else:
                        logger.info("✅ ALL INDICATORS VALIDATED AND READY FOR TRADING.")
    
                except Exception as e:
                    reason = f"Indicator validation crashed: {e}"
                    logger.critical(f"❌ {reason}", exc_info=True)
                    failed_checks.append(f"IndicatorValidatorCrash: {reason}")
            
            # Check 7: WebSocket optimization
            logger.info("Check 8/8: WebSocket optimization...")
            if self.ws_optimizer and self.ws_optimizer.ws_manager:
                is_connected = self.ws_optimizer.ws_manager.is_any_client_connected()
                if is_connected:
                    # Bağlantı varsa, gerçek aktif stream sayısını alıp loglayalım.
                    stream_count = self.ws_optimizer.ws_manager.get_active_stream_count()
                    logger.info(f"✓ WebSocket connected. Active streams: {stream_count}")
                else:
                    # Bağlantı yoksa, bunu bir uyarı olarak belirtelim.
                    logger.warning("⚠️ WebSocket initialized but not connected. System may rely on REST API.")
                    # Bu durumu bir hata olarak saymak istemeyebiliriz, çünkü sistem REST ile devam edebilir.
            else:
                logger.error("❌ WebSocket manager not initialized.")
                failed_checks.append("WebSocket manager initialization")
            
            # --- FINAL SUMMARY ---
            logger.info("\n" + "="*70)
            if not failed_checks:
                logger.info("✓ ALL PRE-FLIGHT CHECKS PASSED")
                return True
            else:
                logger.error("❌ SOME PRE-FLIGHT CHECKS FAILED")
                logger.error("Failing checks:")
                for i, check_reason in enumerate(failed_checks, 1):
                    logger.error(f"  {i}. {check_reason}")
                logger.info("="*70)
                return False

        except Exception as e:
            logger.error(f"❌ Pre-flight checks crashed unexpectedly: {e}", exc_info=True)
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
        logger.info("="*70)
        
        _health_task = None
        
        try:
            # STEP 1: USE EXISTING WS IF ALREADY RUNNING; OTHERWISE TRY TO ESTABLISH
            ws_connected = False
            if self._is_ws_initialized():
                try:
                    ws_status = await self.ws_optimizer.get_stream_status()
                    # Accept as connected if either explicit 'running' flag is True or active stream count > 0
                    if ws_status.get('running') or ws_status.get('active_streams', 0) > 0:
                        logger.info("WebSocket already initialized and running; skipping re-initialization")
                        ws_connected = True
                    else:
                        logger.info("WebSocket initialized but no active streams; attempting to (re)establish...")
                        ws_connected = await self._establish_websocket_connection(max_retries=3, timeout=30)
                except Exception as e:
                    logger.warning(f"WS status check failed: {e}; attempting to (re)establish...")
                    ws_connected = await self._establish_websocket_connection(max_retries=3, timeout=30)
            else:
                logger.info("WebSocket not initialized; continuing with REST mode")
                ws_connected = False

            # Report final WS status
            if not ws_connected:
                logger.warning("=" * 70)
                logger.warning("⚠️ WebSocket connection unavailable")
                logger.warning("⚠️ Continuing with REST API mode (reduced real-time data)")
                logger.warning("=" * 70)
                if self.telegram:
                    self.telegram.send(
                        "⚠️ <b>WebSocket Unavailable</b>\n"
                        "Trading will continue using REST API\n"
                        "Real-time data may be limited"
                    )
            else:
                logger.info("=" * 70)
                logger.info("✅ WEBSOCKET CONNECTED - REAL-TIME DATA STREAMING")
                logger.info("=" * 70)

            # STEP 2: SEND STARTUP NOTIFICATION
            if self.telegram:
                ws_info = "WebSocket CONNECTED ✅" if ws_connected else "REST API mode (WebSocket unavailable)"
                
                # Safe extraction with sentinel value support
                max_pos = self.RISK_PARAMS.get('max_position_size', self.DEFAULT_RISK_PARAMS['max_position_size'])
                stop_loss = self.RISK_PARAMS.get('stop_loss_pct')
                take_profit = self.RISK_PARAMS.get('take_profit_pct')
                
                # Format values for display
                max_pos_str = f"{max_pos:.1%}" if max_pos is not None else "Dynamic"
                stop_loss_str = f"{stop_loss:.1%}" if stop_loss is not None else "Dynamic (ATR-based)"
                take_profit_str = f"{take_profit:.1%}" if take_profit is not None else "Dynamic (ATR-based)"
                
                self.telegram.send(
                    f"🚀 <b>LIVE TRADING STARTED</b>\n"
                    f"Mode: {self.mode.upper()}\n"
                    f"Capital: {self.CAPITAL_USDT} USDT\n"
                    f"Exchange: BingX\n"
                    f"Pairs: {len(self.TRADING_PAIRS)}\n"
                    f"Data: {ws_info}\n"
                    f"Max Position: {max_pos_str}\n"
                    f"Stop Loss: {stop_loss_str}\n"
                    f"Take Profit: {take_profit_str}"
                )
                        
            # STEP 3: ACTIVATE TRADING SYSTEMS
            logger.info("\n" + "="*70)
            logger.info("🚀 ACTIVATING TRADING SYSTEMS")
            logger.info("="*70)
            
            self.coordinator.is_running = True
            logger.info("✅ Production coordinator activated (is_running = True)")
            
            if self.coordinator.trading_engine:
                start_result = await self.coordinator.trading_engine.start_live_trading(mode=self.mode)
                if not start_result.get('success'):
                    logger.error(f"❌ Failed to start trading engine: {start_result.get('reason')}")
                    return
                logger.info("✅ Trading engine started (state = RUNNING)")
                logger.info(f"   - Active tasks: {start_result.get('active_tasks', 0)}")
                logger.info(f"   - Mode: {start_result.get('mode', 'unknown')}")
            else:
                logger.error("❌ Trading engine not available")
                return
            
            # 3.3: Start health monitoring (if enabled)
            if self.health_monitor:
                # Use the task returned by start_monitoring (avoid wrapping with create_task again)
                _health_task = await self.health_monitor.start_monitoring()
                logger.info("✅ Health monitor started in background")
            
            # STEP 4: RUN PRODUCTION LOOP
            logger.info("\n" + "="*70)
            logger.info("🚀 STARTING PRODUCTION LOOP")
            logger.info("="*70)
            logger.info("🔍 [LAUNCHER-DEBUG] About to call coordinator.run_production_loop()")
            
            if self.coordinator is None:
                logger.critical("❌ [LAUNCHER-DEBUG] coordinator is None! Cannot proceed!")
                raise RuntimeError("Coordinator is None - initialization failed")
            
            logger.info(f"🔍 [LAUNCHER-DEBUG] coordinator type: {type(self.coordinator)}")
            logger.info(f"🔍 [LAUNCHER-DEBUG] coordinator.is_running: {self.coordinator.is_running}")
            logger.info(f"🔍 [LAUNCHER-DEBUG] coordinator.is_initialized: {self.coordinator.is_initialized}")
            logger.info(f"🔍 [LAUNCHER-DEBUG] Parameters: mode={self.mode}, duration={duration}, continuous={self.infinite}")
            
            if not hasattr(self.coordinator, 'run_production_loop'):
                logger.critical("❌ [LAUNCHER-DEBUG] coordinator has no run_production_loop method!")
                raise RuntimeError("Coordinator missing run_production_loop method")
            
            await self.coordinator.run_production_loop(
                mode=self.mode,
                duration=duration,
                continuous=self.infinite
            )
            logger.info("🔍 [LAUNCHER-DEBUG] coordinator.run_production_loop() RETURNED")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Keyboard interrupt received - initiating shutdown...")
            raise
        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}", exc_info=True)
            if self.health_monitor:
                self.health_monitor.record_error(str(e))
            raise
        finally:
            if self.health_monitor:
                try:
                    await self.health_monitor.stop_monitoring()
                except Exception as e:
                    logger.error(f"Error stopping health monitor: {e}")
            
            if _health_task and not _health_task.done():
                _health_task.cancel()
                try:
                    await _health_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("\n" + "="*70)
            logger.info("INITIATING GRACEFUL SHUTDOWN")
            logger.info("="*70)
            await self.cleanup()
    
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
    
    # *** DÜZELTME 1: Bu metod, `TypeError` hatasını önlemek için güncellendi. ***
    def _print_configuration_summary(self):
        """
        Print comprehensive configuration summary at startup.
        Issue #119: Enhanced log header with complete system information.
        """
        logger.info("="*70)
        logger.info("📊 CONFIGURATION SUMMARY")
        logger.info("="*70)
        
        system_info = SystemInfoCollector.get_system_info()
        
        risk_manager = None
        if self.coordinator and hasattr(self.coordinator, 'risk_manager'):
            risk_manager = self.coordinator.risk_manager
        
        # *** UPDATED: Pass cached health status from pre-flight checks ***
        header = format_startup_header(
            system_info=system_info,
            mode=self.mode,
            dry_run=self.dry_run,
            debug_mode=self.debug_mode,
            exchange_clients=self.exchange_clients,
            ws_manager=(self.ws_optimizer.ws_manager if self._is_ws_initialized() else None),
            capital=self.CAPITAL_USDT,
            trading_pairs=self.TRADING_PAIRS,
            strategies=self.strategies if hasattr(self, 'strategies') else {},
            risk_params=self.RISK_PARAMS,
            risk_manager=risk_manager,
            cached_exchange_status=self._cached_exchange_status,  # *** NEW ***
            cached_ws_status=self._cached_ws_status  # *** NEW ***
        )
        
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
            
            with open(log_filename, 'r', encoding='utf-8') as f:
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
        Run trading system once without auto-restart. This is the main
        execution flow of the application.
        """
        exit_code = 0
        try:
            # ===================================================================
            # ADIM 1: ORTAM VE BORSA BAĞLANTISINI HAZIRLA
            # ===================================================================
            if not self._load_environment(): return 1
            
            # Normalize risk parameters after loading environment
            self._normalize_risk_params()

            logger.info("\n[2/8] Initializing BingX Exchange Connection...")
            try:
                creds = {'apiKey': os.getenv('BINGX_KEY'), 'secret': os.getenv('BINGX_SECRET')} if self._has_bingx_credentials and not self.dry_run else None
                bingx_client = CcxtClient('bingx', creds=creds)
                bingx_client.set_required_symbols(self.TRADING_PAIRS)
                bingx_client.load_markets() # Not async
                self.exchange_clients['bingx'] = bingx_client
                logger.info(f"✓ BingX client created and configured for {len(self.TRADING_PAIRS)} symbols.")
                
                # Hızlı bir test yap
                test_ticker = bingx_client.ticker('BTC/USDT') # Not async
                logger.info(f"✓ Connection test OK. BTC price: ${test_ticker['last']:.2f}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize exchange connection: {e}", exc_info=True)
                return 1

            # ===================================================================
            # ADIM 2: STRATEJİLERİ YÜKLE
            # ===================================================================
            # Bu, Phase 1'den ÖNCE yapılmalı ki, coordinator stratejileri bilsin.
            if not await self._initialize_strategies():
                logger.error("\n❌ Strategy initialization failed - aborting launch.")
                return 1

            # ===================================================================
            # ADIM 3: TEMEL SİSTEMLERİ (CORE) BAŞLAT (PHASE 1)
            # ===================================================================
            logger.info("\n" + "="*70 + "\n[PHASE 1] INITIALIZING CORE SYSTEMS\n" + "="*70)
            if not await self._initialize_production_system_core():
                return 1

            # ===================================================================
            # ADIM 4: VERİ KATI SAĞLIĞINI KONTROL ET (PHASE 1.5)
            # ===================================================================
            logger.info("\n" + "="*70 + "\n[PHASE 1.5] DATA LAYER HEALTH CHECK\n" + "="*70)
            if not await self._perform_data_health_check():
                logger.error("\n❌ Data layer health check failed - aborting launch.")
                return 1
                
            # ===================================================================
            # ADIM 5: ML SİSTEMLERİNİ BAŞLAT (PHASE 2)
            # ===================================================================
            logger.info("\n" + "="*70 + "\n[PHASE 2] INITIALIZING ML SYSTEMS\n" + "="*70)
            if not await self._initialize_production_system_ml():
                logger.warning("⚠️ ML initialization failed - continuing with limited AI features.")
            
            self.coordinator.is_initialized = True
            logger.info("✅ Production coordinator marked as initialized.")
            
            # ===================================================================
            # ADIM 6: STRATEJİLERİ KOORDİNATÖRE KAYDET
            # ===================================================================
            # Bu, TÜM sistemler başlatıldıktan SONRA yapılmalı.
            logger.info("\n[FINAL STEP] Registering initialized strategies with the coordinator...")
            if not await self._register_strategies():
                logger.error("\n❌ Strategy registration failed - aborting launch.")
                return 1

            # ===================================================================
            # ADIM 7: SON UÇUŞ ÖNCESİ KONTROLLER
            # ===================================================================
            if not await self._perform_preflight_checks():
                logger.error("\n❌ Pre-flight checks failed - aborting launch.")
                return 1
            
            self._print_configuration_summary()
            
            if self.dry_run:
                logger.info("\n✓ Dry run completed successfully. No trading was started.")
                return 0
            
            # ===================================================================
            # ADIM 8: TİCARET DÖNGÜSÜNÜ BAŞLAT
            # ===================================================================
            await self._start_trading_loop(duration)
            return 0
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Interrupted by user (Ctrl+C).")
            return 130
            
        except Exception as e:
            logger.critical(f"❌ A fatal error occurred in the main execution flow: {e}", exc_info=True)
            return 1
        
        finally:
            if not self._cleanup_completed:
                logger.info("Performing final cleanup in _run_once...")
                await self.cleanup()
    
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
        logger.info("✅ Trading completed successfully" if exit_code == 0 else f"⚠️ Trading exited with code {exit_code}")
    
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
        if launcher and not launcher._cleanup_completed:
            logger.info("Performing final cleanup...")
            try:
                await launcher.cleanup()
            except Exception as e:
                logger.error(f"❌ Cleanup failed: {e}")
                if exit_code == 0:
                    exit_code = 1
        
        logger.info("=" * 70)
        logger.info(f"👋 Bot shutdown complete (exit code: {exit_code})")
        logger.info("=" * 70)
    
    return exit_code

if __name__ == '__main__':
    # Hata yakalama ve loglama için bu bloğu kullanın
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        # Hata anında loglamanın çalıştığından emin ol
        import traceback
        
        # Hata detaylarını bir string olarak al
        error_details = traceback.format_exc()
        
        # Hem konsola hem de dosyaya yaz
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        error_log_path = os.path.join(log_dir, "CRITICAL_FAILURE.log")
        
        # Konsola hatayı bas
        print("="*80, file=sys.stderr)
        print("❌ A CRITICAL, UNHANDLED ERROR OCCURRED IN THE LAUNCHER!", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(error_details, file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(f"Full error details have been saved to: {error_log_path}", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        # Dosyaya hatayı yaz
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("="*80 + "\n")
            f.write("A critical, unhandled error occurred in the launcher:\n")
            f.write("="*80 + "\n")
            f.write(error_details)
            
        # Hata koduyla çık
        sys.exit(1)
