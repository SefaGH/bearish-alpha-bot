"""
Live Trading Engine.
Production-ready live trading EXECUTION engine (execution-only mode).

✅ PURE EXECUTION MODE:
- Processes signals from ProductionCoordinator's queue
- Executes orders via OrderManager
- Manages positions via PositionManager
- NO active market scanning (delegated to ProductionCoordinator)

Architecture:
    ProductionCoordinator → StrategyCoordinator → LiveTradingEngine
    (Scanning + Signals)    (Validation + Queue)   (Execution Only)
"""
import os
import asyncio
import logging
import time
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
from src.core.signal_intents import INTENT_ENTRY, INTENT_FORCE_SWAP, INTENT_REVERSE
from .dca_watcher import DCAWatcher
from .execution_env import is_prod_canary_0_enabled, is_real_execution_enabled, is_vst_fullbot_canary_enabled

from core.logger import get_current_run_id

# ✅ DÜZELTME 1: Import hatası düzeltildi
# Strategy imports - both adaptive and base strategies
try:
    from strategies.adaptive_ob import AdaptiveOversoldBounce
    from strategies.adaptive_str import AdaptiveShortTheRip
except ImportError:
    # Fallback if adaptive strategies don't exist
    from strategies.oversold_bounce import OversoldBounce
    from strategies.short_the_rip import ShortTheRip
    
    # Create adaptive wrappers
    class AdaptiveOversoldBounce(OversoldBounce):
        """Fallback adaptive wrapper for OversoldBounce"""
        def __init__(self, cfg, regime_analyzer=None):
            super().__init__(cfg)
            self.regime_analyzer = regime_analyzer  # ✅ 'p' kaldırıldı!
            
    class AdaptiveShortTheRip(ShortTheRip):
        """Fallback adaptive wrapper for ShortTheRip"""
        def __init__(self, cfg, regime_analyzer=None):
            super().__init__(cfg)
            self.regime_analyzer = regime_analyzer  # ✅ 'p' kaldırıldı!

# Import config validator (needed for signal processing loop)
try:
    from .config_validator import ConfigValidator
except ImportError:
    from .config_validator import ConfigValidator

# Core imports
from .order_manager import SmartOrderManager
from .position_manager import AdvancedPositionManager
from .execution_analytics import ExecutionAnalytics

# Import config with try/except for flexibility
try:
    from ..config.live_trading_config import get_config
except ImportError:
    from config.live_trading_config import get_config

if TYPE_CHECKING:
    from .strategy_coordinator import StrategyCoordinator

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = 'paper'
    LIVE = 'live'
    SIMULATION = 'simulation'


class EngineState(Enum):
    """Engine state enumeration."""
    STOPPED = 'stopped'
    STARTING = 'starting'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPING = 'stopping'
    ERROR = 'error'


class LiveTradingEngine:
    """Production-ready live trading execution engine with enhanced debugging."""
    
    def __init__(self, mode='paper', portfolio_manager=None, risk_manager=None,
                 order_manager=None, position_manager=None,
                 exchange_clients=None, strategy_coordinator: Optional['StrategyCoordinator'] = None,
                 market_data_pipeline: Optional[Any] = None,
                 websocket_manager: Optional[Any] = None, **kwargs):
        """
        Initialize live trading engine with pre-configured managers.
        
        Args:
            mode: Trading mode ('paper', 'live', 'simulation')
            portfolio_manager: PortfolioManager instance.
            risk_manager: RiskManager instance.
            order_manager: Pre-configured SmartOrderManager instance.
            position_manager: Pre-configured AdvancedPositionManager instance.
            exchange_clients: Dict of exchange client instances.
            strategy_coordinator: StrategyCoordinator instance.
            market_data_pipeline: MarketDataPipeline instance.
        """
        if exchange_clients is not None and not isinstance(exchange_clients, dict):
            raise TypeError(f"exchange_clients must be a dict, got {type(exchange_clients).__name__}")
            
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.exchange_clients = exchange_clients or {}
    
        self.strategy_coordinator = strategy_coordinator
        self.market_data_pipeline = market_data_pipeline
        self.websocket_manager = websocket_manager

        if kwargs:
            logger.debug("LiveTradingEngine received unused kwargs: %s", list(kwargs.keys()))
        
        # --- DEĞİŞTİRİLEN KISIM ---
        # Artık kendi yöneticilerimizi oluşturmuyoruz, dışarıdan hazır alıyoruz.
        self.order_manager = order_manager
        self.position_manager = position_manager
                     
        self.execution_analytics = ExecutionAnalytics(self.order_manager, self.position_manager)
        
        # Engine state
        self.state = EngineState.STOPPED
        try:
            self.mode = TradingMode(mode)
        except ValueError:
            logger.warning(f"Invalid trading mode '{mode}', defaulting to 'paper'.")
            self.mode = TradingMode.PAPER
        
        # Signal queue
        self.signal_queue = asyncio.Queue()
        
        # Active tracking
        self.active_orders = {}
        self.active_positions = {}
        self.trade_history = []
        
        # Background tasks
        self.tasks = []
        
        # Load configuration using unified loader (correct priority: ENV > YAML > Defaults)
        self.config = get_config()
        
        # WebSocket integration configuration
        cfg = self.config
        self.ws_config = {
            'priority_enabled': cfg.get('websocket', {}).get('priority_enabled', True),
            'max_data_age_seconds': cfg.get('websocket', {}).get('max_data_age', 60),
            'fallback_threshold': cfg.get('websocket', {}).get('fallback_threshold', 3)
        }

        # WebSocket usage statistics
        self.ws_stats = {
            'websocket_fetches': 0,
            'rest_fetches': 0,
            'websocket_failures': 0,
            'total_latency_ws': 0.0,
            'total_latency_rest': 0.0,
            'avg_latency_ws': 0.0,
            'avg_latency_rest': 0.0,
            'websocket_success_rate': 0.0,
            'last_ws_fetch_time': None,
            'last_rest_fetch_time': None,
            'consecutive_ws_failures': 0
        }

        # DCA watcher (initialized lazily)
        self.dca_watcher: Optional[DCAWatcher] = None

        # Validate universe configuration
        if 'universe' not in self.config or not self.config['universe']:
            logger.warning("⚠️ No universe config found, using defaults")
            self.config['universe'] = {
                'fixed_symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'],
                'auto_select': False
            }
        
        # Log final configuration
        fixed_symbols = self.config['universe'].get('fixed_symbols', [])
        logger.info(f"✅ Config loaded: {len(fixed_symbols)} symbols")
        logger.info(f"   Symbols: {fixed_symbols}")
        logger.info(f"   Priority: ENV > YAML > Defaults")
        
        # Universe cache for optimization
        self._cached_symbols = None
        self._universe_built = False
        
        # Performance tracking
        self._signal_count = 0  # Received from ProductionCoordinator
        self._executed_count = 0  # Track executed signals
        self._last_signal_time = None
        self.prod_canary_0_abort_reason: Optional[str] = None

        logger.info("LiveTradingEngine initialized")
        logger.info(f"  Mode: {mode}")
        exchange_client_names = list(self.exchange_clients.keys()) if self.exchange_clients else []
        logger.info(f"  Exchange clients: {exchange_client_names}")

    def set_strategy_coordinator(self, coordinator: Optional['StrategyCoordinator']) -> None:
        """Attach StrategyCoordinator reference for execution callbacks."""
        self.strategy_coordinator = coordinator
    
    async def start_live_trading(self, mode: str = 'paper') -> Dict[str, Any]:
        """
        Start live trading with all integrated systems.
        
        Args:
            mode: Trading mode ('paper', 'live', 'simulation')
            
        Returns:
            Startup result
        """
        try:
            logger.info("="*70)
            logger.info("STARTING LIVE TRADING ENGINE")
            logger.info("="*70)
            
            self.state = EngineState.STARTING
            
            # Set trading mode
            if mode == 'live':
                self.mode = TradingMode.LIVE
                logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
            elif mode == 'simulation':
                self.mode = TradingMode.SIMULATION
                logger.info("📊 Simulation mode - Using historical data")
            else:
                self.mode = TradingMode.PAPER
                logger.info("📝 Paper trading mode - No real executions")
            
            # Initialize Phase 3 components
            logger.info("\n[Phase 3.1] Initializing WebSocket connections...")
            if self.market_data_pipeline and self.market_data_pipeline.websocket_manager:
                logger.info("  ✓ WebSocket manager ready (via MarketDataPipeline)")
            else:
                logger.warning("  ⚠️  No WebSocket manager - real-time data disabled")
            
            logger.info("\n[Phase 3.2] Initializing Risk Management...")
            risk_status = await self._initialize_risk_management()
            if not risk_status['success']:
                raise RuntimeError(f"Risk management initialization failed: {risk_status['reason']}")
            logger.info("  ✓ Risk management initialized")
            
            logger.info("\n[Phase 3.3] Initializing Portfolio Management...")
            portfolio_status = await self._initialize_portfolio_management()
            if not portfolio_status['success']:
                raise RuntimeError(f"Portfolio management initialization failed: {portfolio_status['reason']}")
            logger.info("  ✓ Portfolio management initialized")
            
            logger.info("\n[Phase 3.4] Starting Live Trading Components...")
            
            # REMOVED: Duplicate prefetch (Issue #259 followup fix)
            # Historical data is now fetched ONCE in production_coordinator.initialize_core_systems()
            # via market_data_pipeline.prime_data_buffers_async()
            # OLD CODE: await self._prefetch_historical_data()
            logger.info("[Phase 3.4.1] Historical data prefetch skipped (already completed in Phase 1)")
            
            # Transition the engine state before background loops execute so they observe RUNNING.
            self.state = EngineState.RUNNING

            # Restore open positions from snapshot before loops start (restart safety)
            if self.position_manager and hasattr(self.position_manager, "restore_positions_from_snapshot"):
                try:
                    restore_result = await self.position_manager.restore_positions_from_snapshot(
                        exchange_clients=(self.exchange_clients if self.mode == TradingMode.LIVE else None),
                        reconcile_with_exchange=(self.mode == TradingMode.LIVE),
                    )
                    # Seed engine active_positions so monitoring loop tracks restored positions.
                    restored_positions = getattr(self.position_manager, "positions", {}) or {}
                    for pid, pos in restored_positions.items():
                        self.active_positions.setdefault(pid, pos)
                    logger.info(
                        "[SNAPSHOT] restore_result=%s active_positions_seeded=%s",
                        restore_result,
                        len(restored_positions),
                    )
                except Exception as exc:
                    logger.error("[SNAPSHOT] Restore failed: %s", exc, exc_info=True)

            self._initialize_dca_watcher()

            # Start signal processing
            signal_task = asyncio.create_task(self._signal_processing_loop())
            self.tasks.append(signal_task)
            logger.info("  ✓ Signal processing started")

            # Start strategy coordinator bridge (CRITICAL for signal flow)
            if self.strategy_coordinator:
                bridge_task = asyncio.create_task(self._strategy_coordinator_bridge_loop())
                self.tasks.append(bridge_task)
                logger.info("  ✓ Strategy coordinator bridge started")
            else:
                logger.warning("  ⚠️ No StrategyCoordinator attached - bridge not started")

            # Start position monitoring
            position_task = asyncio.create_task(self._position_monitoring_loop())
            self.tasks.append(position_task)
            logger.info("  ✓ Position monitoring started")

            # Start order management
            order_task = asyncio.create_task(self._order_management_loop())
            self.tasks.append(order_task)
            logger.info("  ✓ Order management started")

            # Start performance reporting
            perf_task = asyncio.create_task(self._performance_reporting_loop())
            self.tasks.append(perf_task)
            logger.info("  ✓ Performance reporting started")

            # Yield control so newly created tasks can progress before we return.
            await asyncio.sleep(0)

            if self.dca_watcher and self.strategy_coordinator:
                dca_task = asyncio.create_task(self._dca_watch_loop())
                self.tasks.append(dca_task)
                logger.info("  ? DCA watcher loop started")
            
            logger.info("\n" + "="*70)
            logger.info("✓ LIVE TRADING ENGINE STARTED SUCCESSFULLY")
            logger.info("="*70)
            
            return {
                'success': True,
                'state': self.state.value,
                'mode': self.mode.value,
                'active_tasks': len(self.tasks),
                'execution_mode': 'passive',  # Changed from universe_size
                'signal_source': 'ProductionCoordinator'
            }
            
        except Exception as e:
            logger.error(f"Error starting live trading: {e}")
            self.state = EngineState.ERROR
            return {
                'success': False,
                'reason': str(e),
                'state': self.state.value
            }
    
    async def stop_live_trading(self) -> Dict[str, Any]:
        """
        Stop live trading gracefully.
        Issue #134: Enhanced with exit summary logging.
        
        Returns:
            Shutdown result
        """
        try:
            logger.info("Stopping live trading engine...")
            self.state = EngineState.STOPPING

            # VST full-bot canary safety: always attempt to close positions before halting background tasks.
            # This keeps the canary window minimal-risk and ensures evidence JSON can show flat final state.
            if is_vst_fullbot_canary_enabled() and self.active_positions:
                logger.warning(
                    "[VST-FULLBOT-CANARY] Shutdown: attempting to close %s open position(s)",
                    len(self.active_positions),
                )
                for position_id in list(self.active_positions.keys()):
                    try:
                        await self._execute_position_exit(position_id, {"exit_reason": "shutdown"})
                    except Exception as exc:
                        logger.error(
                            "[VST-FULLBOT-CANARY] Shutdown close failed for %s: %s",
                            position_id,
                            exc,
                            exc_info=True,
                        )
             
            # Cancel all background tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.tasks.clear()
            self.state = EngineState.STOPPED
            
            # Log exit summary statistics only when we actually have data
            if self.position_manager:
                closed_trades = getattr(self.position_manager, 'closed_positions', [])
                if closed_trades:
                    self.position_manager.log_exit_summary()
                else:
                    logger.info("No closed positions captured yet; exit summary will be emitted after position closure.")
            
            logger.info("Live trading engine stopped")
            logger.info(f"  Total signals generated: {self._signal_count}")
            logger.info(f"  Total signals executed: {self._executed_count}")
            
            return {
                'success': True,
                'state': self.state.value,
                'total_signals': self._signal_count,
                'total_executed': self._executed_count
            }
            
        except Exception as e:
            logger.error(f"Error stopping live trading: {e}")
            return {
                'success': False,
                'reason': str(e)
            }

    def _resolve_execution_config(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve per-position execution config using precedence:
          Signal Overrides > Strategy Profile (config) > Global Defaults

        Backward compatible:
          - If new keys are missing, uses existing global trailing/DCA config.
          - Accepts legacy override aliases: trailing_stop_config, dca_config.
        """
        cfg = self.config if isinstance(self.config, dict) else {}

        def _as_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _normalize_dynamic_steps(value: Any) -> List[Dict[str, float]]:
            if not isinstance(value, list):
                return []
            steps: List[Dict[str, float]] = []
            for raw in value:
                if not isinstance(raw, dict):
                    continue
                try:
                    activation_pnl = raw.get("activation_pnl", raw.get("activation_pnl_pct", None))
                    new_delta_pct = raw.get("new_delta_pct", None)
                    if activation_pnl is None or new_delta_pct is None:
                        continue
                    activation_pnl_f = float(activation_pnl)
                    new_delta_f = float(new_delta_pct)
                    if activation_pnl_f < 0:
                        activation_pnl_f = 0.0
                    if new_delta_f < 0:
                        new_delta_f = 0.0
                    steps.append(
                        {
                            "activation_pnl": activation_pnl_f,
                            "new_delta_pct": new_delta_f,
                        }
                    )
                except (TypeError, ValueError):
                    continue
            steps.sort(key=lambda s: s.get("activation_pnl", 0.0))
            return steps

        def _normalize_breakeven_lock(value: Any) -> Optional[Dict[str, Any]]:
            if value is True:
                value = {"enabled": True}
            if not isinstance(value, dict):
                return None
            enabled = bool(value.get("enabled", True))
            activation_pct = value.get("activation_pct")
            buffer_bps = value.get("buffer_bps")
            try:
                activation_pct = float(activation_pct) if activation_pct is not None else None
            except (TypeError, ValueError):
                activation_pct = None
            try:
                buffer_bps = float(buffer_bps) if buffer_bps is not None else None
            except (TypeError, ValueError):
                buffer_bps = None
            return {
                "enabled": enabled,
                "activation_pct": max(0.0, activation_pct) if activation_pct is not None else None,
                "buffer_bps": max(0.0, buffer_bps) if buffer_bps is not None else None,
            }

        # --- Global defaults (fallback) ---
        trailing_global = _as_dict(_as_dict(cfg.get("position_management")).get("trailing_stop"))
        global_trailing_enabled = bool(trailing_global.get("trailing_stop_enabled", False))
        global_delta = _safe_float(trailing_global.get("trailing_stop_distance", 0.02), 0.02)
        global_activation = _safe_float(trailing_global.get("activation_threshold", 0.0), 0.0)
        global_dynamic_steps = _normalize_dynamic_steps(trailing_global.get("dynamic_steps"))

        dca_global = _as_dict(cfg.get("dca"))
        dca_strategy = _as_dict(dca_global.get("strategy"))
        global_dca_enabled = bool(dca_global.get("enabled", False))
        global_dca_max_layers = _safe_int(dca_strategy.get("max_layers", 0), 0)
        global_dca_step_pct = _safe_float(dca_strategy.get("step_pct", 0.0), 0.0)

        resolved: Dict[str, Any] = {
            "profile": None,
            "trailing_stop": {
                "enabled": global_trailing_enabled,
                "delta_pct": max(0.0, global_delta),
                "activation_threshold_pct": max(0.0, global_activation),
                "dynamic_steps": list(global_dynamic_steps),
                "trigger_source": None,
                "breakeven_lock": None,
            },
            "dca": {
                "enabled": global_dca_enabled,
                "max_layers": max(0, global_dca_max_layers),
                "step_pct": max(0.0, global_dca_step_pct),
            },
        }

        # --- Strategy profile (config) ---
        strategy_name = (signal.get("strategy_name") or signal.get("strategy") or "").strip()
        strat_cfg = _as_dict(_as_dict(cfg.get("strategies")).get(strategy_name))
        profile_name = (strat_cfg.get("execution_profile") or "").strip() if strat_cfg else ""

        if profile_name:
            profiles = _as_dict(cfg.get("execution_profiles"))
            profile_cfg = _as_dict(profiles.get(profile_name))
            if profile_cfg:
                resolved["profile"] = profile_name
                profile_ts = _as_dict(profile_cfg.get("trailing_stop"))
                if profile_ts:
                    if "enabled" in profile_ts:
                        resolved["trailing_stop"]["enabled"] = bool(profile_ts.get("enabled"))
                    if "delta_pct" in profile_ts:
                        resolved["trailing_stop"]["delta_pct"] = max(0.0, _safe_float(profile_ts.get("delta_pct"), resolved["trailing_stop"]["delta_pct"]))
                    if "activation_threshold_pct" in profile_ts:
                        resolved["trailing_stop"]["activation_threshold_pct"] = max(0.0, _safe_float(profile_ts.get("activation_threshold_pct"), resolved["trailing_stop"]["activation_threshold_pct"]))
                    if "dynamic_steps" in profile_ts:
                        resolved["trailing_stop"]["dynamic_steps"] = _normalize_dynamic_steps(profile_ts.get("dynamic_steps"))
                    if "trigger_source" in profile_ts:
                        resolved["trailing_stop"]["trigger_source"] = str(profile_ts.get("trigger_source") or "").strip() or None
                    if "breakeven_lock" in profile_ts:
                        resolved["trailing_stop"]["breakeven_lock"] = _normalize_breakeven_lock(profile_ts.get("breakeven_lock"))

                profile_dca = _as_dict(profile_cfg.get("dca"))
                if profile_dca:
                    if "enabled" in profile_dca:
                        resolved["dca"]["enabled"] = bool(profile_dca.get("enabled"))
                    if "max_layers" in profile_dca:
                        resolved["dca"]["max_layers"] = max(0, _safe_int(profile_dca.get("max_layers"), resolved["dca"]["max_layers"]))
                    if "step_pct" in profile_dca:
                        resolved["dca"]["step_pct"] = max(0.0, _safe_float(profile_dca.get("step_pct"), resolved["dca"]["step_pct"]))

        # --- Signal overrides (highest precedence) ---
        overrides = _as_dict(signal.get("execution"))

        # Legacy alias support
        if isinstance(signal.get("trailing_stop_config"), dict):
            overrides = {**overrides, "trailing_stop": {**_as_dict(overrides.get("trailing_stop")), **_as_dict(signal.get("trailing_stop_config"))}}
        if isinstance(signal.get("dca_config"), dict):
            overrides = {**overrides, "dca": {**_as_dict(overrides.get("dca")), **_as_dict(signal.get("dca_config"))}}

        ts_override = _as_dict(overrides.get("trailing_stop"))
        if ts_override:
            if "enabled" in ts_override:
                resolved["trailing_stop"]["enabled"] = bool(ts_override.get("enabled"))
            if "delta_pct" in ts_override:
                resolved["trailing_stop"]["delta_pct"] = max(0.0, _safe_float(ts_override.get("delta_pct"), resolved["trailing_stop"]["delta_pct"]))
            if "dynamic_steps" in ts_override:
                resolved["trailing_stop"]["dynamic_steps"] = _normalize_dynamic_steps(ts_override.get("dynamic_steps"))
            if "trigger_source" in ts_override:
                resolved["trailing_stop"]["trigger_source"] = str(ts_override.get("trigger_source") or "").strip() or None
            if "breakeven_lock" in ts_override:
                resolved["trailing_stop"]["breakeven_lock"] = _normalize_breakeven_lock(ts_override.get("breakeven_lock"))
            # Support activation_threshold as alias for activation_threshold_pct
            if "activation_threshold_pct" in ts_override:
                resolved["trailing_stop"]["activation_threshold_pct"] = max(0.0, _safe_float(ts_override.get("activation_threshold_pct"), resolved["trailing_stop"]["activation_threshold_pct"]))
            elif "activation_threshold" in ts_override:
                resolved["trailing_stop"]["activation_threshold_pct"] = max(0.0, _safe_float(ts_override.get("activation_threshold"), resolved["trailing_stop"]["activation_threshold_pct"]))
            elif "activation_price" in ts_override:
                # Best-effort conversion using signal entry price (fill may differ; this is still useful in paper mode).
                entry = _safe_float(signal.get("entry"), 0.0)
                activation_price = _safe_float(ts_override.get("activation_price"), 0.0)
                side = str(signal.get("side") or "").lower()
                if entry > 0 and activation_price > 0:
                    if side in ("sell", "short"):
                        resolved["trailing_stop"]["activation_threshold_pct"] = max(0.0, (entry - activation_price) / entry)
                    else:
                        resolved["trailing_stop"]["activation_threshold_pct"] = max(0.0, (activation_price - entry) / entry)

        dca_override = _as_dict(overrides.get("dca"))
        if dca_override:
            if "enabled" in dca_override:
                resolved["dca"]["enabled"] = bool(dca_override.get("enabled"))
            if "max_layers" in dca_override:
                resolved["dca"]["max_layers"] = max(0, _safe_int(dca_override.get("max_layers"), resolved["dca"]["max_layers"]))
            if "step_pct" in dca_override:
                resolved["dca"]["step_pct"] = max(0.0, _safe_float(dca_override.get("step_pct"), resolved["dca"]["step_pct"]))

        return resolved
    
    async def execute_signal(self, signal: Dict, allocation_size: Optional[float] = None) -> Dict[str, Any]:
        """Execute trading signal with full pipeline integration."""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            signal_id = signal.get('signal_id')
            intent = signal.get('intent', INTENT_ENTRY)

            # Resolve per-position execution config (Signal Overrides > Strategy Profile > Global Defaults)
            execution_cfg = self._resolve_execution_config(signal)

            sizing_meta = signal.get('sizing_meta') or {}
            risk_assessment_payload = signal.get('risk_assessment')
            risk_assessment_metrics = {}
            if isinstance(risk_assessment_payload, dict):
                risk_assessment_metrics = risk_assessment_payload.get('metrics') or {}
                if not sizing_meta:
                    sizing_meta = risk_assessment_metrics.get('sizing_meta') or {}
            raw_ppo_mult = sizing_meta.get('ppo_position_multiplier')
            if raw_ppo_mult is None:
                raw_ppo_mult = signal.get('ppo_position_multiplier', 1.0)
            try:
                ppo_multiplier = float(raw_ppo_mult or 1.0)
            except (TypeError, ValueError):
                ppo_multiplier = 1.0
            try:
                engine_multiplier = float(signal.get('position_multiplier', 1.0) or 1.0)
            except (TypeError, ValueError):
                engine_multiplier = 1.0
            combined_multiplier = ppo_multiplier * engine_multiplier

            # ---------------------------------------------------------------------
            # Price SSOT (Single Source of Truth):
            # - Execution price is determined upstream (signal) and can optionally
            #   include a slippage/limit buffer BEFORE any risk calculations.
            # - OrderManager must not fetch its own ticker price for limit pricing.
            # ---------------------------------------------------------------------
            execution_params = signal.get('execution_params') if isinstance(signal.get('execution_params'), dict) else {}
            side_norm = str(signal.get('side', '') or '').lower()
            order_type_hint = (
                execution_params.get('type')
                or execution_params.get('order_type')
                or self.config.get('trading', {}).get('order_type', 'limit')
            )
            order_type_norm = str(order_type_hint or 'limit').lower()
            if order_type_norm == 'limit':
                # Keep legacy behavior (0.1% maker offset) but apply it BEFORE risk sizing/validation.
                cfg_trading = self.config.get('trading', {}) if isinstance(self.config, dict) else {}
                raw_offset = (
                    execution_params.get('slippage_buffer')
                    or execution_params.get('price_offset')
                    or cfg_trading.get('slippage_buffer')
                    or cfg_trading.get('limit_price_offset')
                    or 0.001
                )
                try:
                    price_offset = float(raw_offset or 0.0)
                except (TypeError, ValueError):
                    price_offset = 0.001
                if price_offset < 0:
                    price_offset = 0.0

                if not signal.get('_execution_price_locked'):
                    try:
                        base_entry = float(signal.get('entry') or 0.0)
                    except (TypeError, ValueError):
                        base_entry = 0.0
                    if base_entry > 0:
                        adjusted_entry = (
                            base_entry * (1 - price_offset)
                            if side_norm in ('buy', 'long')
                            else base_entry * (1 + price_offset)
                        )
                        delta = adjusted_entry - base_entry

                        signal.setdefault('entry_raw', base_entry)
                        signal['entry'] = adjusted_entry
                        signal['limit_price'] = adjusted_entry
                        signal['execution_price'] = adjusted_entry
                        signal['execution_price_offset'] = price_offset
                        signal['_execution_price_locked'] = True

                        # Keep stop/target distances consistent with the new entry for risk/RR.
                        for k in ('stop', 'stop_loss', 'target', 'take_profit'):
                            if k not in signal:
                                continue
                            try:
                                val = float(signal.get(k) or 0.0)
                            except (TypeError, ValueError):
                                continue
                            if val > 0:
                                shifted = val + delta
                                if shifted > 0:
                                    signal[k] = shifted
             
            # [EXECUTION START] Log signal execution start
            logger.info(f"[EXECUTION-START] Processing signal for {symbol}")
            
            # Enhanced logging for adaptive signals
            if signal.get('is_adaptive'):
                logger.info(f"🎯 Executing ADAPTIVE signal for {symbol}")
                if signal.get('adaptive_threshold'):
                    logger.info(f"  Adaptive RSI threshold: {signal['adaptive_threshold']:.1f}")
            else:
                logger.info(f"📊 Executing signal for {symbol}")
            
            # Log signal details
            logger.info(f"  Strategy: {signal.get('strategy_name', 'unknown')}")  
            logger.info(f"  Side: {signal.get('side', 'unknown').upper()}")
            logger.info(f"  Entry: ${signal.get('entry', 0):.2f}")
            logger.info(f"  Reason: {signal.get('reason', 'N/A')}")
            if signal.get('entry_raw') and signal.get('execution_price_offset') is not None:
                try:
                    logger.info(
                        "  Execution pricing: base=%.2f offset=%.3f%% limit=%.2f",
                        float(signal.get('entry_raw')),
                        float(signal.get('execution_price_offset')) * 100.0,
                        float(signal.get('entry')),
                    )
                except Exception:
                    pass

            if any(abs(val - 1.0) > 1e-6 for val in (combined_multiplier, ppo_multiplier, engine_multiplier)):
                logger.info(
                    "  Position size multiplier: %.2f (PPO=%.2f, engine=%.2f)",
                    combined_multiplier,
                    ppo_multiplier,
                    engine_multiplier,
                )

            # Prefer planner-approved sizing when present
            planner_size = risk_assessment_metrics.get('final_position_size') or signal.get('planner_planned_qty')
            planner_notional = risk_assessment_metrics.get('final_notional')
            if planner_size:
                signal.setdefault('position_size', planner_size)
                signal.setdefault('amount', planner_size)
            if planner_notional:
                signal.setdefault('notional', planner_notional)

            planner_active = bool(signal.get('planner_active') or planner_notional or planner_size)
            if planner_active:
                signal['planner_active'] = True
            # Planner invariant (RISK_SIZE_PLANNER_ENABLED=true): execution uses planner_planned_notional/qty (or risk_assessment.metrics.final_*), skips multipliers, and passes the already-capped notional into validate_new_position so PositionSizeRule cannot see a larger value than the planner cap.

            # If planner provides a notional budget, convert it into units using the (possibly adjusted) entry price
            # so the executed notional matches the planner cap (and risk sees the same entry).
            if planner_active and signal.get('notional'):
                try:
                    planned_notional = float(signal.get('notional'))
                    planned_entry = float(signal.get('entry') or 0.0)
                    if planned_notional > 0 and planned_entry > 0:
                        planned_qty = planned_notional / planned_entry
                        signal['position_size'] = planned_qty
                        signal['amount'] = planned_qty
                except (TypeError, ValueError):
                    pass

            # Step 1: Determine portfolio allocation (planner-aware)
            strategy_name = signal.get('strategy', 'default')
            if allocation_size is not None:
                position_size = allocation_size
            elif planner_active:
                position_size = signal.get('position_size') or signal.get('amount') or 0.0
                if abs(engine_multiplier - 1.0) > 1e-6:
                    logger.info(
                        "  Skipping engine multiplier: planner_active",
                        extra={'engine_multiplier': engine_multiplier, 'ppo_multiplier': ppo_multiplier},
                    )
            else:
                # Calculate position size based on risk
                position_size = await self.risk_manager.calculate_position_size(signal)

                # Apply adaptive/engine position multiplier if configured
                if abs(engine_multiplier - 1.0) > 1e-6:
                    position_size *= engine_multiplier
                    logger.info(
                        "  Applied engine multiplier: %.2f (PPO=%.2f, total=%.2f)",
                        engine_multiplier,
                        ppo_multiplier,
                        combined_multiplier,
                    )

            if position_size <= 0:
                logger.warning("Position size is zero or negative")
                return {
                    'success': False,
                    'reason': 'Invalid position size',
                    'stage': 'position_sizing'
                }

            entry_price = signal.get('entry', 0)
            if planner_active and signal.get('notional'):
                try:
                    notional_value = float(signal.get('notional'))
                except (TypeError, ValueError):
                    notional_value = position_size * entry_price
            else:
                notional_value = position_size * entry_price

            signal['position_size'] = position_size
            signal['amount'] = position_size
            signal['notional'] = notional_value
            logger.info(f"  ✓ Position size prepared: {position_size:.6f} (notional=${notional_value:.2f})")

            try:
                ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                run_id = get_current_run_id()
                logger.info(
                    "trade_execution_size_debug %s",
                    {
                        'event': 'trade_execution_size_debug',
                        'timestamp': ts,
                        'run_id': run_id,
                        'symbol': symbol,
                        'timeframe': signal.get('timeframe') or signal.get('tf'),
                        'strategy_name': signal.get('strategy_name') or signal.get('strategy'),
                        'entry_price': entry_price,
                        'final_position_size': position_size,
                        'final_notional': notional_value,
                        'planner_final_notional': signal.get('planner_planned_notional') or signal.get('notional'),
                        'planner_final_position_size': signal.get('planner_planned_qty') or signal.get('position_size'),
                        'volume_bucket_at_entry': signal.get('volume_bucket'),
                        'position_size_multiplier_from_volume': signal.get('position_size_multiplier'),
                        'planner_active': planner_active,
                    },
                )
            except Exception:
                pass

            # Step 2: Risk validation (Phase 3.2) now runs against final size
            risk_validation = await self.risk_manager.validate_new_position(signal, self.portfolio_manager)

            if not risk_validation[0]:  # is_valid
                logger.warning(f"❌ Risk validation failed: {risk_validation[1]}")
                risk_metrics = risk_validation[2]

                # Enhanced logging for capital limit failures
                if 'current_exposure' in risk_metrics:
                    logger.warning(f"   Current Exposure: ${risk_metrics.get('current_exposure', 0):.2f}")
                    logger.warning(f"   Attempted Position Value: ${risk_metrics.get('new_position_value', 0):.2f}")
                    logger.warning(f"   Capital Limit: ${risk_metrics.get('capital_limit', 0):.2f}")

                return {
                    'success': False,
                    'reason': f"Risk validation failed: {risk_validation[1]}",
                    'stage': 'risk_validation',
                    'risk_metrics': risk_metrics
                }

            risk_metrics = risk_validation[2]
            logger.info(f"  ✓ Risk validation passed: {risk_metrics}")
            
            # Step 3: Select optimal exchange (Phase 1)
            exchange = signal.get('exchange', list(self.exchange_clients.keys())[0] if self.exchange_clients else None)
            if not exchange or exchange not in self.exchange_clients:
                logger.error(f"Exchange not available: {exchange}")
                return {
                    'success': False,
                    'reason': f'Exchange not available: {exchange}',
                    'stage': 'exchange_selection'
                }
            
            # CRITICAL FIX: Add exchange to signal to ensure position tracking has valid exchange
            # This prevents "Exchange not available: unknown" error during shutdown position closure
            signal['exchange'] = exchange
            
            # Step 4: Determine execution algorithm
            # Calculate notional value once for use in algorithm selection and logging
            notional_value = signal.get('notional', position_size * signal.get('entry', 0))
            
            # ✅ FIX: Respect ORDER_TYPE configuration from environment/config
            config_order_type = self.config.get('trading', {}).get('order_type', 'limit')
            
            # Paper mode and live mode both should respect the configured order type
            if config_order_type:
                execution_algo = config_order_type.lower()
                logger.info(f"  ✓ Using configured order type: {execution_algo}")
            else:
                # Fallback to execution analytics if no config
                urgency = signal.get('urgency', 'normal')
                execution_algo = self.execution_analytics.get_best_execution_algorithm(notional_value, urgency)
                logger.info(f"  ✓ Execution algorithm selected: {execution_algo}")
            
            logger.info(f"  Notional value: ${notional_value:.2f}")
            
            # Step 5: Execute order (Phase 3.4)
            if self.mode == TradingMode.LIVE:
                logger.warning("  ⚠️  Executing LIVE order")
            else:
                logger.info(f"  📝 Executing {self.mode.value} order")
            
            order_request = {
                'symbol': symbol,
                'side': signal.get('side', 'buy'),
                'amount': position_size,
                'exchange': exchange,
                'signal': signal,
                # SSOT limit price (pre-risk, pre-validation) – OrderManager must not re-fetch ticker to price limits.
                'limit_price': signal.get('limit_price') or signal.get('execution_price') or signal.get('entry'),
                'execution_params': execution_params,
            }
            
            # Reverse intent handling: if this signal is marked as a
            # reverse and references an existing position, try to close
            # that position BEFORE opening the new one. This keeps
            # trade history and exposure clean while still respecting
            # intent-aware risk gating.
            if intent == INTENT_REVERSE:
                reverse_from = signal.get('reverse_from_position_id')
                if reverse_from:
                    logger.info(
                        "[REVERSE] Attempting to close existing position %s on %s before opening reverse.",
                        reverse_from,
                        symbol,
                    )
                    try:
                        close_result = await self._execute_position_exit(reverse_from, {'exit_reason': 'reverse'})

                        if not close_result.get('success'):
                            logger.error(
                                "[REVERSE] Failed to close source position %s: %s",
                                reverse_from,
                                close_result.get('reason'),
                            )
                            return {
                                'success': False,
                                'reason': f"Failed to close source position {reverse_from}: {close_result.get('reason')}",
                                'stage': 'reverse_close',
                            }

                        logger.info(
                            "[REVERSE] Successfully closed source position %s; proceeding to open reverse.",
                            reverse_from,
                        )
                    except Exception as reverse_error:
                        logger.error("[REVERSE] Exception while closing source position %s: %s", reverse_from, reverse_error, exc_info=True)
                        return {
                            'success': False,
                            'reason': f"Exception while closing source position {reverse_from}: {reverse_error}",
                            'stage': 'reverse_close',
                        }
            elif intent == INTENT_FORCE_SWAP:
                swap_target = signal.get('swap_target_id')
                if swap_target:
                    logger.warning(
                        "⚡ [FORCE-SWAP] Closing weakest position %s on %s before opening new extreme entry.",
                        swap_target,
                        symbol,
                    )
                    try:
                        close_result = await self._execute_position_exit(swap_target, {'exit_reason': 'force_swap'})

                        if not close_result.get('success'):
                            logger.error(
                                "⚠️ [FORCE-SWAP] Failed to close swap target %s: %s",
                                swap_target,
                                close_result.get('reason'),
                            )
                            return {
                                'success': False,
                                'reason': f"Failed to close swap target {swap_target}: {close_result.get('reason')}",
                                'stage': 'force_swap_close',
                            }

                        logger.info(
                            "⚡ [FORCE-SWAP] Successfully closed %s; continuing with new position.",
                            swap_target,
                        )
                    except Exception as swap_error:
                        logger.error("[FORCE-SWAP] Exception while closing %s: %s", swap_target, swap_error, exc_info=True)
                        return {
                            'success': False,
                            'reason': f"Exception while closing swap target {swap_target}: {swap_error}",
                            'stage': 'force_swap_close',
                        }
                else:
                    logger.warning("⚠️ [FORCE-SWAP] intent detected but no swap_target_id provided; proceeding without pre-close.")

            execution_result = await self.order_manager.place_order(order_request, execution_algo)
            
            if not execution_result.get('success'):
                logger.error(f"❌ Order execution failed: {execution_result.get('reason')}")
                return {
                    'success': False,
                    'reason': execution_result.get('reason'),
                    'stage': 'order_execution'
                }
            
            logger.info(f"  ✓ Order executed: {execution_result.get('order_id')}")
            
            # Step 6: Open position tracking (Phase 3.4)
            position_result = await self.position_manager.open_position(signal, execution_result)
            
            if not position_result.get('success'):
                logger.error(f"Position tracking failed: {position_result.get('reason')}")
                return {
                    'success': False,
                    'reason': position_result.get('reason'),
                    'stage': 'position_tracking'
                }
            
            position_id = position_result['position_id']
            position = position_result.get('position')

            # -------------------------------------------------------------
            # Stage-4: Production Canary-0 (hard stop only) - fail-fast if
            # we cannot confirm exchange-native hard stop placement.
            # -------------------------------------------------------------
            if is_prod_canary_0_enabled():
                hard_stop_required = os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
                trailing_flag = os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
                exchange_name = str((position or {}).get("exchange") or "").lower()
                hard_stop_order_id = (position or {}).get("native_hard_stop_order_id")

                if hard_stop_required and not trailing_flag and exchange_name == "bingx" and not hard_stop_order_id:
                    failure_reason = (position or {}).get("native_hard_stop_place_failed_reason") or "missing_native_hard_stop_order_id"
                    abort_reason = f"native_hard_stop_missing:{failure_reason}"
                    self.prod_canary_0_abort_reason = abort_reason

                    if isinstance(position, dict):
                        position["prod_canary_0_abort"] = True
                        position["prod_canary_0_abort_reason"] = abort_reason

                    logger.critical(
                        "[PROD-CANARY-0] Native HARD_STOP missing after entry; closing position immediately. position_id=%s reason=%s",
                        position_id,
                        abort_reason,
                    )

                    # Ensure _execute_position_exit sees the position.
                    self.active_positions[position_id] = position
                    close_result = await self._execute_position_exit(
                        position_id,
                        {"exit_reason": "native_hard_stop_place_failed"},
                    )
                    return {
                        "success": False,
                        "stage": "prod_canary_0_abort",
                        "reason": abort_reason,
                        "position_id": position_id,
                        "close_result": close_result,
                    }

            # Persist resolved execution settings on the position for restart safety + DCA gating.
            if position_id and self.position_manager and hasattr(self.position_manager, "attach_execution_config"):
                try:
                    self.position_manager.attach_execution_config(position_id, execution_cfg)
                except Exception as exc:
                    logger.warning("Execution config attach failed for %s: %s", position_id, exc)

            # Apply trailing-stop settings per-position (Signal/Profile override > Global fallback).
            if position and isinstance(execution_cfg, dict):
                ts_cfg = execution_cfg.get("trailing_stop") if isinstance(execution_cfg.get("trailing_stop"), dict) else {}
                if ts_cfg.get("enabled", False):
                    trailing_distance = float(ts_cfg.get("delta_pct", 0.02) or 0.02)
                    activation_threshold = float(ts_cfg.get("activation_threshold_pct", 0.0) or 0.0)
                    try:
                        if hasattr(self.position_manager, "configure_trailing_stop"):
                            enable_result = self.position_manager.configure_trailing_stop(
                                position_id,
                                enabled=True,
                                delta_pct=trailing_distance,
                                activation_threshold_pct=activation_threshold,
                            )
                        else:
                            enable_result = self.position_manager.enable_trailing_stop(
                                position_id,
                                trailing_distance,
                                activation_threshold=activation_threshold
                            )
                    except Exception as exc:
                        enable_result = {"success": False, "reason": str(exc)}

                    if not enable_result.get("success"):
                        logger.warning(
                            "Trailing stop enable failed for %s: %s",
                            position_id,
                            enable_result.get("reason")
                        )
            logger.info(f"  ✓ Position opened: {position_id}")
            
            # Store in active positions
            self.active_positions[position_id] = position_result['position']
            
            # Increment executed count
            self._executed_count += 1
            
            # Record in trade history
            trade_record = {
                'timestamp': datetime.now(timezone.utc),
                'signal': signal,
                'execution_result': execution_result,
                'position_id': position_id,
                'risk_metrics': risk_metrics
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"✅ Signal execution completed for {symbol}")
            logger.info(f"📊 Total executed: {self._executed_count}")
            logger.info("="*50)

            lifecycle_error = None

            # Notify strategy coordinator for lifecycle tracking
            if signal_id and self.strategy_coordinator:
                execution_summary = {
                    'order': execution_result,
                    'position': position_result,
                    'mode': self.mode.value,
                    'completed_at': datetime.now(timezone.utc)
                }
                try:
                    self.strategy_coordinator.mark_signal_executed(signal_id, execution_summary)
                except Exception as callback_error:
                    cleanup_state = 'pending'
                    cleanup_method = None
                    cleanup_errors: List[str] = []

                    lifecycle_error = {
                        'error': str(callback_error),
                        'stage': 'lifecycle_callback'
                    }

                    logger.error(
                        f"Failed to mark signal {signal_id} as executed: {callback_error}",
                        exc_info=True
                    )

                    discard_method = getattr(self.strategy_coordinator, 'discard_active_signal', None)

                    if callable(discard_method):
                        try:
                            discard_method(signal_id)
                            cleanup_state = 'discarded'
                            cleanup_method = 'discard_active_signal'
                        except Exception as cleanup_error:
                            cleanup_errors.append(str(cleanup_error))
                            logger.error(
                                "Failed to discard active signal %s after callback error: %s",
                                signal_id,
                                cleanup_error,
                                exc_info=True
                            )
                    else:
                        cleanup_errors.append('discard_active_signal_unavailable')

                    if cleanup_state != 'discarded':
                        try:
                            active_signals = getattr(self.strategy_coordinator, 'active_signals', None)
                            if isinstance(active_signals, dict) and signal_id in active_signals:
                                active_signals.pop(signal_id, None)
                                cleanup_state = 'discarded'
                                cleanup_method = 'direct_active_signals_pop'
                                logger.warning(
                                    "Signal %s removed from coordinator via direct fallback after lifecycle callback failure",
                                    signal_id
                                )
                            elif cleanup_state == 'pending':
                                cleanup_state = 'not_found'
                        except Exception as fallback_error:
                            cleanup_errors.append(str(fallback_error))
                            logger.error(
                                "Fallback removal failed for signal %s after lifecycle callback error: %s",
                                signal_id,
                                fallback_error,
                                exc_info=True
                            )
                            cleanup_state = 'failed'

                    lifecycle_error['cleanup'] = cleanup_state

                    if cleanup_method:
                        lifecycle_error['cleanup_method'] = cleanup_method

                    if cleanup_errors:
                        lifecycle_error['cleanup_error'] = cleanup_errors if len(cleanup_errors) > 1 else cleanup_errors[0]

            result = {
                'success': True,
                'position_id': position_id,
                'order_id': execution_result.get('order_id'),
                'execution_result': execution_result,
                'position_result': position_result
            }

            if lifecycle_error:
                result['lifecycle_error'] = lifecycle_error

            return result
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)
            return {
                'success': False,
                'reason': str(e),
                'stage': 'execution_error'
            }
    
    async def _signal_processing_loop(self):
        """Background task for processing signals from queue ONLY - Pure execution mode."""
        logger.info("Signal processing loop started (execution-only mode)")
        logger.info("  Market scanning: DISABLED")
        logger.info("  Signal source: ProductionCoordinator only")
        
        # Import monitor for adaptive signals
        from core.adaptive_monitor import adaptive_monitor
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    # ✅ ONLY: Process queued signals
                    signal = await asyncio.wait_for(
                        self.signal_queue.get(),
                        timeout=1.0
                    )
                    
                    # [STAGE: RECEIVED] Signal received from queue
                    logger.info(f"[STAGE:RECEIVED] 📤 Signal received from queue: {signal.get('symbol', 'unknown')}")
                    self._signal_count += 1
                    self._last_signal_time = datetime.now(timezone.utc)
                    
                    # Execute signal
                    result = await self.execute_signal(signal)
                    
                    if result['success']:
                        # [STAGE: EXECUTED] Signal successfully executed
                        logger.info(f"[STAGE:EXECUTED] ✅ Signal executed: {signal.get('symbol')} - Position opened")
                    else:
                        logger.warning(f"⚠️ Signal execution failed: {result.get('reason')}")
                    
                    # Monitor adaptive signals
                    if signal and signal.get('is_adaptive'):
                        adaptive_monitor.record_adaptive_signal(signal.get('symbol'), signal)
                        
                except asyncio.TimeoutError:
                    # Normal: Queue is empty, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error executing signal: {e}", exc_info=True)
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("Signal processing loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in signal processing loop: {e}", exc_info=True)

    async def _dca_watch_loop(self):
        """Background loop to emit DCA scale-in signals via the coordinator."""
        if not self.dca_watcher or not self.strategy_coordinator:
            return
        poll = getattr(self.dca_watcher, "poll_interval", 15.0) or 15.0
        try:
            while self.state == EngineState.RUNNING:
                try:
                    signals = await self.dca_watcher.run_once()
                    for sig in signals or []:
                        base_strategy = (
                            (sig.get("strategy_name") if isinstance(sig, dict) else None)
                            or ((sig.get("meta") or {}).get("base_strategy") if isinstance(sig, dict) else None)
                            or "dca_watcher"
                        )
                        await self.strategy_coordinator.process_strategy_signal(base_strategy, sig)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("[DCA] Watch loop error: %s", exc, exc_info=True)
                await asyncio.sleep(poll)
        except asyncio.CancelledError:
            logger.info("[DCA] Watch loop cancelled")
        except Exception as exc:
            logger.error("[DCA] Watch loop terminated: %s", exc, exc_info=True)

    async def _strategy_coordinator_bridge_loop(self):
        """
        Bridge task to transfer signals from StrategyCoordinator queue to LiveTradingEngine queue.
        This implements the proper signal flow: StrategyCoordinator → LiveTradingEngine
        
        Signal Flow:
        1. StrategyCoordinator validates and queues signal
        2. This bridge transfers it to LiveTradingEngine queue  
        3. LiveTradingEngine._signal_processing_loop() executes it
        """
        logger.info("📌 [BRIDGE] Strategy Coordinator bridge loop started")
        logger.info("📌 [BRIDGE] Monitoring StrategyCoordinator.signal_queue for signals")
        
        bridge_stats = {
            'signals_transferred': 0,
            'transfer_errors': 0,
            'last_transfer_time': None
        }
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    # Check if coordinator exists
                    if not self.strategy_coordinator:
                        logger.warning("📌 [BRIDGE] No StrategyCoordinator attached, waiting...")
                        await asyncio.sleep(5)
                        continue
                    
                    # Get signal from StrategyCoordinator queue
                    dispatcher = getattr(self.strategy_coordinator, 'try_dispatch_next', None)
                    if callable(dispatcher):
                        signal_data = await dispatcher(timeout=1.0)
                    else:
                        signal_data = await asyncio.wait_for(
                            self.strategy_coordinator.get_next_signal(),
                            timeout=1.0
                        )
                    
                    if signal_data:
                        success = await self._forward_signal_from_coordinator(signal_data, bridge_stats)
                        if success:
                            coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                            engine_queue_size = self.signal_queue.qsize()
                            logger.info(
                                f"📌 [BRIDGE-QUEUES] Coordinator: {coordinator_queue_size} | Engine: {engine_queue_size}"
                            )
                        continue
                    else:
                        continue
                        
                except asyncio.TimeoutError:
                    # Normal timeout, no signal available
                    continue
                    
                except Exception as e:
                    bridge_stats['transfer_errors'] += 1
                    logger.error(f"📌 [BRIDGE-ERROR] Error in bridge loop: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Brief pause on error
                    
        except asyncio.CancelledError:
            logger.info(f"📌 [BRIDGE] Bridge loop cancelled. Stats: {bridge_stats}")
            raise
            
        except Exception as e:
            logger.critical(f"📌 [BRIDGE-FATAL] Fatal error in bridge loop: {e}", exc_info=True)
            raise
    
    async def _forward_signal_from_coordinator(self, signal_data: Dict[str, Any], bridge_stats: Optional[Dict[str, Any]] = None) -> bool:
        """Normalize coordinator payloads and enqueue them for execution."""
        if not signal_data:
            return False

        signal_id = signal_data.get('signal_id', 'unknown')
        symbol = signal_data.get('signal', {}).get('symbol', 'unknown')

        logger.info(f"📌 [BRIDGE-RECEIVE] Got signal {signal_id} for {symbol} from StrategyCoordinator")

        enriched_signal = {
            **(signal_data.get('signal') or {}),
            'signal_id': signal_id,
            'risk_assessment': signal_data.get('risk_assessment'),
            'routing': signal_data.get('routing'),
            'from_coordinator': True,
            'bridge_timestamp': datetime.now(timezone.utc)
        }

        if enriched_signal.get('ml_enhanced'):
            if isinstance(bridge_stats, dict):
                bridge_stats['ml_enhanced'] = bridge_stats.get('ml_enhanced', 0) + 1
            logger.info(f"🧠 [ML-BRIDGE] Signal {signal_id} is ML-enhanced")

            if hasattr(self, 'rl_agent') and self.rl_agent:
                try:
                    import numpy as np
                    state = enriched_signal.get('rl_state', np.zeros(50))
                    action = ['buy', 'hold', 'sell'].index(enriched_signal.get('side', 'hold'))
                    enriched_signal['rl_state'] = state
                    enriched_signal['rl_action'] = action
                except Exception as exc:
                    logger.debug(f"RL state prep failed: {exc}")

        if enriched_signal.get('ml_blocked'):
            if isinstance(bridge_stats, dict):
                bridge_stats['ml_blocked'] = bridge_stats.get('ml_blocked', 0) + 1
            logger.info(f"🧠 [ML-BRIDGE] Signal {signal_id} blocked by ML - skipping")
            return False

        await self.signal_queue.put(enriched_signal)

        if isinstance(bridge_stats, dict):
            bridge_stats['signals_transferred'] += 1
            bridge_stats['last_transfer_time'] = datetime.now(timezone.utc)
            logger.info(f"📌 [BRIDGE-STATS] Total transferred: {bridge_stats['signals_transferred']}")

        logger.info(f"📌 [BRIDGE-TRANSFER] Signal {signal_id} transferred to LiveTradingEngine queue")
        return True

    async def trigger_coordinator_drain(self, timeout: float = 0.0) -> bool:
        """Manually nudge the coordinator bridge when execution slots free up."""
        if not self.strategy_coordinator:
            return False

        dispatcher = getattr(self.strategy_coordinator, 'try_dispatch_next', None)
        if not callable(dispatcher):
            return False

        try:
            payload = await dispatcher(timeout=timeout)
        except Exception as exc:
            logger.error(f"📌 [BRIDGE] Failed to drain coordinator queue: {exc}")
            return False

        if not payload:
            return False

        return await self._forward_signal_from_coordinator(payload)

    async def _prefetch_historical_data(self):
        """
        Prefetches historical data by delegating the task to the MarketDataPipeline.
        """
        try:
            # Artık doğrudan kendi pipeline nesnesini kullanıyor.
            if not self.market_data_pipeline:
                logger.error("[PREFETCH] MarketDataPipeline not available to LiveTradingEngine. Cannot prefetch data.")
                return
    
            # Get symbols and timeframes from the central config.
            symbols = self.config.get('universe', {}).get('fixed_symbols', [])
            timeframes_str = self.config.get('websocket', {}).get('stream_timeframes', '1m,5m,30m,1h,4h')
            
            # Timeframe'leri doğru formatta parse et
            if isinstance(timeframes_str, list):
                timeframes = [item.strip() for sublist in (tf.split(',') for tf in timeframes_str) for item in sublist]
            else:
                timeframes = [tf.strip() for tf in timeframes_str.split(',')]
    
            if not symbols or not timeframes:
                logger.warning("[PREFETCH] No symbols or timeframes configured to prefetch.")
                return
    
            logger.info(f"[PREFETCH] Delegating historical data prefetch to MarketDataPipeline for {len(symbols)} symbols...")
            
            # Doğrudan kendi pipeline nesnesindeki metodu çağır.
            await self.market_data_pipeline.prime_data_buffers_async(symbols, timeframes)
            
            logger.info("[PREFETCH] Delegation to MarketDataPipeline complete.")
        
        except Exception as e:
            logger.error(f"[PREFETCH] Fatal error during historical data prefetch: {e}", exc_info=True)
            logger.warning("[PREFETCH] Continuing without pre-fetched data. Signal generation may be delayed.")

    # --- GERİYE DÖNÜK UYUMLULUK METODU ---
    def prefetch_data(self):
        """
        Backwards-compatible public wrapper for _prefetch_historical_data.
        This ensures that external callers expecting `prefetch_data` do not break.
        It simply returns the awaitable coroutine from the internal method.
        """
        logger.debug("[PREFETCH-WRAPPER] Public `prefetch_data` called. Delegating to internal `_prefetch_historical_data`.")
        return self._prefetch_historical_data()
    
    def _determine_default_exchange(self, symbol: str) -> Optional[str]:
        """Determine default exchange for a generated signal."""

        if symbol and ':' in symbol:
            exchange_hint = symbol.split(':')[-1].lower()
            if exchange_hint in self.exchange_clients:
                return exchange_hint

        if self.exchange_clients:
            return next(iter(self.exchange_clients.keys()))

        return self.config.get('execution', {}).get('default_exchange')
    
    def get_websocket_stats(self):
        """Get comprehensive WebSocket statistics."""
        stats = self.ws_stats.copy()
        if self.market_data_pipeline and self.market_data_pipeline.websocket_manager:
            stats['connection_health'] = self.market_data_pipeline.websocket_manager.get_connection_health()
        total = stats['websocket_fetches'] + stats['rest_fetches']
        stats['websocket_usage_ratio'] = (stats['websocket_fetches'] / total * 100) if total > 0 else 0.0
        if stats['avg_latency_rest'] > 0 and stats['avg_latency_ws'] > 0:
            stats['latency_improvement_pct'] = ((stats['avg_latency_rest'] - stats['avg_latency_ws']) / stats['avg_latency_rest'] * 100)
        else:
            stats['latency_improvement_pct'] = 0.0
        return stats
    
    def _log_websocket_performance(self):
        """
        Log WebSocket performance metrics.
        Displays usage ratio, latencies, and performance improvement.
        """
        stats = self.get_websocket_stats()
        logger.info(
            f"[WS-PERFORMANCE]\n"
            f"  Usage Ratio: {stats['websocket_usage_ratio']:.1f}%\n"
            f"  WS Latency: {stats['avg_latency_ws']:.1f}ms\n"
            f"  REST Latency: {stats['avg_latency_rest']:.1f}ms\n"
            f"  Improvement: {stats['latency_improvement_pct']:.1f}%"
        )
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from exchange with enhanced error handling and bulk support.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '30m', '1h', '4h')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None on error
        """
        # Try each exchange client until successful
        for exchange_name, client in self.exchange_clients.items():
            try:
                # ✅ DÜZELTME 9: Bulk fetch support for large data requests
                if hasattr(client, 'fetch_ohlcv_bulk') and limit > 500:
                    # Use bulk fetch for large requests
                    logger.debug(f"Using bulk fetch for {symbol} {timeframe} ({limit} candles)")
                    # Assuming fetch_ohlcv_bulk is synchronous and returns a list, not DataFrame
                    data_list = client.fetch_ohlcv_bulk(symbol, timeframe=timeframe, target_limit=limit)
                    if data_list:
                        # Convert list to DataFrame here
                        df = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        return df
                elif hasattr(client, 'ohlcv'):
                    # Use standard method, which is async and returns a DataFrame
                    df = await client.ohlcv(symbol, timeframe, limit=limit)
                    if df is not None and not df.empty:
                        logger.debug(f"✅ Fetched {len(df)} candles for {symbol} {timeframe} from {exchange_name}")
                        return df
                else:
                    logger.warning(f"Exchange client {exchange_name} does not support OHLCV fetching")
                    continue
                    
            except Exception as e:
                logger.debug(f"Could not fetch {symbol} {timeframe} from {exchange_name}: {e}")
                continue
        
        logger.warning(f"❌ Failed to fetch {symbol} {timeframe} from all exchanges")
        return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price using centralized MarketDataPipeline.
        
        This method now delegates to MarketDataPipeline.get_latest_price() which
        handles WebSocket-first approach with REST fallback internally.
        This eliminates direct websocket_manager access and ensures consistent
        data flow across the entire system.
        """
        # Check if MarketDataPipeline is available
        if not self.market_data_pipeline:
            logger.error(f"❌ CRITICAL: MarketDataPipeline not available for {symbol}")
            return None
        
        # Use centralized price fetching from MarketDataPipeline
        # It handles WebSocket-first with REST fallback internally
        try:
            price = await self.market_data_pipeline.get_latest_price(symbol, timeframe='1m')
            
            if price is not None and price > 0:
                # Update statistics - track successful fetch
                self.ws_stats['websocket_fetches'] += 1
                self.ws_stats['consecutive_ws_failures'] = 0
                return price
            else:
                # Price fetch failed from all sources
                logger.error(f"❌ CRITICAL: Could not fetch price for {symbol} from MarketDataPipeline (all sources failed)")
                self.ws_stats['websocket_failures'] += 1
                self.ws_stats['consecutive_ws_failures'] += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception while fetching price for {symbol}: {e}", exc_info=True)
            self.ws_stats['websocket_failures'] += 1
            self.ws_stats['consecutive_ws_failures'] += 1
            return None


    async def _position_monitoring_loop(self):
        """Enhanced position monitoring with real-time P&L and exit checking."""
        logger.info("Position monitoring loop started")
        
        interval = 10
        try:
            raw_interval = (
                (self.config or {})
                .get("position_management", {})
                .get("position_monitoring_loop_interval_s", interval)
            )
            interval = max(1, int(raw_interval))
        except Exception:
            interval = 10
            logger.warning(
                "Invalid position_management.position_monitoring_loop_interval_s; using default=%ss",
                interval,
            )
        
        try:
            while self.state == EngineState.RUNNING:
                if not self.active_positions:
                    await asyncio.sleep(interval)
                    continue
                
                logger.debug(f"Monitoring {len(self.active_positions)} active positions")
                
                total_unrealized_pnl = 0.0
                positions_closed_count = 0
                
                for position_id in list(self.active_positions.keys()):
                    try:
                        position = self.active_positions.get(position_id)
                        if not position or not isinstance(position, dict):
                            logger.warning(f"Invalid or missing position data for ID: {position_id}. Removing from active list.")
                            self.active_positions.pop(position_id, None)
                            continue
                        
                        symbol = position.get('symbol')
                        # *** KÖK NEDEN DÜZELTMESİ: Savunmacı Kodlama ***
                        if not isinstance(symbol, str):
                            logger.error(f"CRITICAL: Position {position_id} has a non-string symbol: {symbol} (type: {type(symbol)}). Skipping.")
                            continue

                        entry_price = position.get('entry_price', 0)
                        exchange = position.get('exchange')
                        if exchange == 'unknown':
                            exchange = None

                        closed_price = await self._get_current_price(symbol)

                        trigger_price = None
                        trigger_source = 'unknown'
                        trigger_fallback = 'none'
                        trigger_age_ms = None

                        if self.market_data_pipeline:
                            trigger_cfg = self.config.get('trigger_price', {}) if isinstance(self.config, dict) else {}
                            execution_cfg = position.get("execution") if isinstance(position.get("execution"), dict) else {}
                            trailing_cfg = execution_cfg.get("trailing_stop") if isinstance(execution_cfg.get("trailing_stop"), dict) else {}
                            trigger_source_cfg = trailing_cfg.get("trigger_source") or trigger_cfg.get('source', 'mid') or 'mid'
                            trigger_price, trigger_source, trigger_fallback = self.market_data_pipeline.get_live_trigger_price(
                                symbol,
                                timeframe='1m',
                                source=trigger_source_cfg,
                                exchange=exchange,
                                forming_close=closed_price,
                                side=position.get('side')
                            )

                            ws_exchange = exchange or (
                                next(iter(self.market_data_pipeline.exchanges.keys()))
                                if self.market_data_pipeline.exchanges else None
                            )
                            collector = self.market_data_pipeline.websocket_manager.collector \
                                if self.market_data_pipeline.websocket_manager else None
                            if collector and ws_exchange:
                                sample = collector.get_latest_ticker_sample(ws_exchange, symbol)
                                if sample and sample.get('timestamp'):
                                    try:
                                        trigger_age_ms = max(
                                            0.0,
                                            (datetime.now(timezone.utc) - sample['timestamp']).total_seconds() * 1000
                                        )
                                    except Exception:
                                        trigger_age_ms = None

                        exit_price = trigger_price
                        trigger_source_final = trigger_source
                        trigger_fallback_final = trigger_fallback
                        fallback_reason = None

                        if exit_price is None or exit_price <= 0:
                            fallback_reason = 'trigger_price_missing'
                        else:
                            ticker_stale_ms = int(
                                (self.config.get('websocket', {}).get('ticker_stale_ms', 5000) or 5000)
                                if isinstance(self.config, dict) else 5000
                            )
                            if trigger_age_ms is not None and trigger_age_ms > ticker_stale_ms:
                                fallback_reason = 'trigger_price_stale'
                            if trigger_source == 'forming_close':
                                fallback_reason = fallback_reason or 'trigger_price_fallback_forming_close'

                        if fallback_reason:
                            if closed_price is None or closed_price <= 0:
                                logger.warning(
                                    f"Invalid or unavailable price for {symbol}, skipping P&L update for position {position_id} "
                                    f"(fallback_reason={fallback_reason})."
                                )
                                continue
                            exit_price = closed_price
                            trigger_source_final = 'closed_close_fallback'
                            if not trigger_fallback_final or trigger_fallback_final == 'none':
                                trigger_fallback_final = fallback_reason
                            logger.warning(
                                f"[EXIT-PRICE-FALLBACK] {position_id} {symbol} "
                                f"reason={fallback_reason} closed_price=${closed_price:.2f}"
                            )

                        if exit_price is None or exit_price <= 0:
                            logger.warning(f"Invalid or unavailable price for {symbol}, skipping P&L update for position {position_id}.")
                            continue

                        position['exit_price'] = exit_price
                        position['closed_price'] = closed_price
                        position['trigger_price_source'] = trigger_source_final
                        position['trigger_price_fallback'] = trigger_fallback_final
                        position['trigger_price_ts_or_age_ms'] = (
                            int(trigger_age_ms) if trigger_age_ms is not None else None
                        )
                        position['current_price'] = exit_price

                        pnl_result = await self.position_manager.monitor_position_pnl(position_id, exit_price)
                        
                        if pnl_result.get('success'):
                            unrealized_pnl = pnl_result.get('unrealized_pnl', 0)
                            pnl_pct = pnl_result.get('pnl_pct', 0)
                            
                            logger.info(
                                f"💰 [P&L-UPDATE] {position_id} | {symbol} | Entry: ${entry_price:.2f}, "
                                f"Current: ${exit_price:.2f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
                            )
                            total_unrealized_pnl += unrealized_pnl

                        exit_health_interval_s = 30
                        now_ts = time.time()
                        last_exit_health_log_ts = position.get('last_exit_health_log_ts', 0.0) or 0.0
                        if (now_ts - last_exit_health_log_ts) >= exit_health_interval_s:
                            trailing_enabled = bool(position.get('trailing_stop_enabled', False))
                            trailing_active = bool(position.get('trailing_stop_activated', False))
                            stop_loss = position.get('stop_loss', 0) or 0
                            take_profit = position.get('take_profit', 0) or 0
                            logger.info(
                                f"[EXIT-HEALTH] {position_id} symbol={symbol} side={position.get('side')} "
                                f"exit_price={exit_price:.4f} trigger_source={trigger_source_final} "
                                f"fallback_chain={trigger_fallback_final} closed_price={closed_price} "
                                f"stop={stop_loss:.4f} tp={take_profit:.4f} "
                                f"trailing_enabled={trailing_enabled} trailing_active={trailing_active} "
                                f"poll_interval_s={interval} trigger_age_ms={position.get('trigger_price_ts_or_age_ms')}"
                            )
                            position['last_exit_health_log_ts'] = now_ts
                        
                        exit_check = await self.position_manager.manage_position_exits(position_id)
                        
                        if exit_check.get('should_exit'):
                            exit_reason = exit_check.get('exit_reason')
                            exit_emoji = '🛑' if 'stop_loss' in exit_reason else '🎯'
                            
                            logger.warning(
                                f"{exit_emoji} [EXIT-SIGNAL] {position_id} for {symbol} due to {exit_reason.upper()}"
                            )
                            
                            await self._execute_position_exit(position_id, exit_check)
                            positions_closed_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error monitoring position {position_id}: {e}", exc_info=True)
                        continue
                
                if self.active_positions:
                    logger.info(
                        f"📊 [MONITORING-SUMMARY] Active: {len(self.active_positions)}, "
                        f"Total Unrealized P&L: ${total_unrealized_pnl:+.2f}, "
                        f"Closed this cycle: {positions_closed_count}"
                    )
                
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info("Position monitoring loop cancelled")
    
    async def _order_management_loop(self):
        """Background task for managing active orders."""
        logger.info("Order management loop started")
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    if self.active_orders:
                        logger.debug(f"Managing {len(self.active_orders)} active orders")
                    
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in order management loop: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("Order management loop cancelled")
    
    async def _performance_reporting_loop(self):
        """Background task for performance reporting."""
        logger.info("Performance reporting loop started")
        
        interval = self.config.get('monitoring', {}).get('performance_report_interval', 3600)
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    report = self.execution_analytics.generate_execution_report('1h')
                    
                    if report['success']:
                        logger.info("📊 Performance report generated")
                        logger.info(f"   Total trades: {report.get('total_trades', 0)}")
                        logger.info(f"   Win rate: {report.get('win_rate', 0):.2%}")
                        logger.info(f"   Average P&L: {report.get('avg_pnl', 0):.2%}")
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in performance reporting: {e}")
                    await asyncio.sleep(interval)
                    
        except asyncio.CancelledError:
            logger.info("Performance reporting loop cancelled")
    
    async def _execute_position_exit(self, position_id: str, exit_signal: Dict) -> Dict[str, Any]:
        """Execute position exit based on exit signal."""
        try:
            logger.info(f"Executing exit for position {position_id}")

            position = self.active_positions.get(position_id)
            if not position:
                logger.warning(f"Position not found: {position_id}")
                return {'success': False, 'reason': 'position_not_found'}

            exit_reason = exit_signal.get('exit_reason') or 'manual'

            logger.info(f"  Symbol: {position.get('symbol')}")
            logger.info(f"  Side: {position.get('side')}")
            logger.info(f"  Amount: {position.get('amount')}")
            logger.info(f"  Exit reason: {exit_reason}")

            if exit_signal.get("skip_market_exit"):
                raw_exit_price = exit_signal.get("exit_price") or position.get("current_price") or position.get("entry_price") or 0
                try:
                    exit_price = float(raw_exit_price or 0.0)
                except (TypeError, ValueError):
                    exit_price = 0.0

                logger.warning("🟦 [NATIVE EXIT] Skipping market order; closing locally: %s", position_id)

                close_result = await self.position_manager.close_position(
                    position_id,
                    exit_price,
                    exit_reason,
                )

                if not close_result.get("success"):
                    logger.error(f"Failed to close position: {close_result.get('reason')}")
                    return {'success': False, 'reason': close_result.get('reason')}

                self.active_positions.pop(position_id, None)
                return {
                    "success": True,
                    "skip_market_exit": True,
                    "exit_price": exit_price,
                    "close_result": close_result,
                }

            # Defense-in-depth: if native conditional order ids exist, confirm the position is still open on
            # the exchange right before placing a market close (race: native stop triggers between exit-check and order placement).
            try:
                exchange_name = str(position.get("exchange") or "").lower()
                has_native_ids = bool(position.get("native_hard_stop_order_id") or position.get("native_trailing_stop_order_id"))
                if exchange_name == "bingx" and has_native_ids and is_real_execution_enabled():
                    checker = getattr(self.position_manager, "_bingx_is_position_open_on_exchange", None)
                    if callable(checker):
                        is_open = await checker(position)
                        if is_open is False:
                            raw_exit_price = exit_signal.get("exit_price") or position.get("current_price") or position.get("entry_price") or 0
                            try:
                                exit_price = float(raw_exit_price or 0.0)
                            except (TypeError, ValueError):
                                exit_price = 0.0

                            logger.warning("?? [NATIVE EXIT] Preflight: exchange already flat; skipping market order: %s", position_id)

                            close_result = await self.position_manager.close_position(
                                position_id,
                                exit_price,
                                exit_reason,
                            )

                            if not close_result.get("success"):
                                logger.error(f"Failed to close position: {close_result.get('reason')}")
                                return {'success': False, 'reason': close_result.get('reason')}

                            self.active_positions.pop(position_id, None)
                            return {
                                "success": True,
                                "skip_market_exit": True,
                                "preflight_skip_market_exit": True,
                                "exit_price": exit_price,
                                "close_result": close_result,
                            }
            except Exception as exc:
                logger.warning("?? [NATIVE EXIT] Preflight check failed (continuing): %s", exc)

            exit_order = {
                'symbol': position['symbol'],
                'side': 'sell' if str(position.get('side') or '').lower() in {'long', 'buy'} else 'buy',
                'amount': position['amount'],
                'exchange': position['exchange']
            }

            # Safer close semantics for BingX hedge mode when real execution is enabled.
            try:
                exchange_name = str(position.get("exchange") or "").lower()
                side_raw = str(position.get("side") or "").lower().strip()
                position_side = "LONG" if side_raw in {"long", "buy"} else ("SHORT" if side_raw in {"short", "sell"} else None)
                if exchange_name == "bingx" and is_real_execution_enabled() and position_side:
                    exit_order["execution_params"] = {"reduceOnly": True, "positionSide": position_side}
            except Exception:
                pass

            execution_result = await self.order_manager.place_order(exit_order, 'market')

            if not execution_result.get('success'):
                logger.error(f"Exit order failed: {execution_result.get('reason')}")
                return {'success': False, 'reason': execution_result.get('reason')}

            exit_price = execution_result.get('avg_price')
            if not exit_price:
                exit_price = position.get('current_price') or position.get('entry_price') or 0

            close_result = await self.position_manager.close_position(
                position_id,
                exit_price,
                exit_reason,
                exit_order_id=execution_result.get("order_id"),
            )

            if not close_result.get('success'):
                logger.error(f"Failed to close position: {close_result.get('reason')}")
                return {'success': False, 'reason': close_result.get('reason')}

            logger.info(f"✅ Position closed successfully: {position_id}")
            logger.info(f"   Exit price: ${exit_price:.2f}")
            logger.info(
                f"   P&L: ${close_result.get('realized_pnl', 0):+.2f} ({close_result.get('return_pct', 0):+.2f}%)"
            )

            self.active_positions.pop(position_id, None)
            return {'success': True, 'exit_price': exit_price, 'close_result': close_result}

        except Exception as e:
            logger.error(f"Error executing position exit: {e}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
    async def _initialize_risk_management(self) -> Dict[str, Any]:
        """Initialize risk management systems."""
        try:
            if self.risk_manager:
                logger.info("  Risk manager initialized")
                return {'success': True}
            else:
                logger.warning("  No risk manager provided")
                return {'success': True}
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    async def _initialize_portfolio_management(self) -> Dict[str, Any]:
        """Initialize portfolio management systems."""
        try:
            if self.portfolio_manager:
                logger.info("  Portfolio manager initialized")
                
                if hasattr(self.portfolio_manager, 'strategies'):
                    strategies = self.portfolio_manager.strategies
                    logger.info(f"  Registered strategies: {list(strategies.keys())}")
                    
                    adaptive_count = sum(1 for name in strategies.keys() if 'adaptive' in name.lower())
                    if adaptive_count > 0:
                        logger.info(f"  🎯 Adaptive strategies: {adaptive_count}")
                        
                return {'success': True}
            else:
                logger.warning("  No portfolio manager provided")
                return {'success': True}
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def _initialize_dca_watcher(self) -> None:
        """Initialize DCA watcher if enabled (disabled by default)."""
        try:
            dca_cfg = self.config.get('dca') if isinstance(self.config, dict) else {}
            if not dca_cfg or not dca_cfg.get('enabled', False):
                logger.info("[DCA] Watcher disabled by config.")
                return
            if not self.position_manager:
                logger.info("[DCA] PositionManager missing; watcher not started.")
                return
            self.dca_watcher = DCAWatcher(
                cfg=self.config,
                position_manager=self.position_manager,
                market_data_pipeline=self.market_data_pipeline,
                portfolio_manager=self.portfolio_manager,
                logger=logger,
            )
            logger.info("[DCA] Watcher initialized (enabled).")
        except Exception as exc:
            logger.error("[DCA] Failed to initialize watcher: %s", exc, exc_info=True)
            self.dca_watcher = None
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status with enhanced metrics."""
        status = {
            'state': self.state.value,
            'mode': self.mode.value,
            'active_positions': len(self.active_positions),
            'active_orders': len(self.active_orders),
            'total_trades': len(self.trade_history),
            'signals_received': self._signal_count,
            'signals_executed': self._executed_count,
            'total_signals': self._signal_count,
            'last_signal_time': self._last_signal_time.isoformat() if self._last_signal_time else None,
            'active_tasks': len([t for t in self.tasks if not t.done()]),
            'signal_queue_size': self.signal_queue.qsize(),
            'universe_size': len(self._cached_symbols) if self._cached_symbols else 0,
            'config': {
                'fixed_symbols': len(self.config.get('universe', {}).get('fixed_symbols', [])),
                'auto_select': self.config.get('universe', {}).get('auto_select', False)
            }
        }
        
        if self.order_manager and hasattr(self.order_manager, 'get_execution_statistics'):
            status['execution_stats'] = self.order_manager.get_execution_statistics()
            
        if self.position_manager and hasattr(self.position_manager, 'get_position_summary'):
            status['position_summary'] = self.position_manager.get_position_summary()
        
        return status
