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
    
    async def execute_signal(self, signal: Dict, allocation_size: Optional[float] = None) -> Dict[str, Any]:
        """Execute trading signal with full pipeline integration."""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            signal_id = signal.get('signal_id')
            intent = signal.get('intent')

            sizing_meta = signal.get('sizing_meta') or {}
            risk_assessment_payload = signal.get('risk_assessment')
            if not sizing_meta and isinstance(risk_assessment_payload, dict):
                metrics = risk_assessment_payload.get('metrics') or {}
                sizing_meta = metrics.get('sizing_meta') or {}
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

            if any(abs(val - 1.0) > 1e-6 for val in (combined_multiplier, ppo_multiplier, engine_multiplier)):
                logger.info(
                    "  Position size multiplier: %.2f (PPO=%.2f, engine=%.2f)",
                    combined_multiplier,
                    ppo_multiplier,
                    engine_multiplier,
                )
            
            # Step 1: Risk validation (Phase 3.2)
            # FIX: Pass PortfolioManager object, not dict
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
            
            # Step 2: Portfolio allocation check (Phase 3.3)
            strategy_name = signal.get('strategy', 'default')
            if allocation_size is None:
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
            else:
                position_size = allocation_size
            
            if position_size <= 0:
                logger.warning("Position size is zero or negative")
                return {
                    'success': False,
                    'reason': 'Invalid position size',
                    'stage': 'position_sizing'
                }
            
            signal['position_size'] = position_size
            logger.info(f"  ✓ Position size calculated: {position_size:.6f}")
            
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
            notional_value = position_size * signal.get('entry', 0)
            
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
                'signal': signal
            }
            
            # Reverse intent handling: if this signal is marked as a
            # reverse and references an existing position, try to close
            # that position BEFORE opening the new one. This keeps
            # trade history and exposure clean while still respecting
            # intent-aware risk gating.
            if intent == 'reverse':
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
            elif intent == 'force_swap':
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
                        
                        current_price = await self._get_current_price(symbol)
                        
                        if current_price is None or current_price <= 0:
                            logger.warning(f"Invalid or unavailable price for {symbol}, skipping P&L update for position {position_id}.")
                            continue
                        
                        pnl_result = await self.position_manager.monitor_position_pnl(position_id, current_price)
                        
                        if pnl_result.get('success'):
                            unrealized_pnl = pnl_result.get('unrealized_pnl', 0)
                            pnl_pct = pnl_result.get('pnl_pct', 0)
                            
                            logger.info(
                                f"💰 [P&L-UPDATE] {position_id} | {symbol} | Entry: ${entry_price:.2f}, "
                                f"Current: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
                            )
                            total_unrealized_pnl += unrealized_pnl
                        
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

            exit_order = {
                'symbol': position['symbol'],
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'amount': position['amount'],
                'exchange': position['exchange']
            }

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
                exit_reason
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
