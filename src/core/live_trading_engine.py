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
    from ..config.live_trading_config import LiveTradingConfiguration
except ImportError:
    from config.live_trading_config import LiveTradingConfiguration

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
                 websocket_manager=None, exchange_clients=None, strategy_coordinator: Optional['StrategyCoordinator'] = None):
        """
        Initialize live trading engine.
        
        Args:
            mode: Trading mode ('paper', 'live', 'simulation')
            portfolio_manager: PortfolioManager from Phase 3.3
            risk_manager: RiskManager from Phase 3.2
            websocket_manager: WebSocketManager from Phase 3.1
            exchange_clients: Dict of exchange client instances from Phase 1
        """
        # VALIDATION EKLENDİ ✅
        if exchange_clients is not None and not isinstance(exchange_clients, dict):
            raise TypeError(f"exchange_clients must be a dict, got {type(exchange_clients).__name__}")
            
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.ws_manager = websocket_manager
        self.exchange_clients = exchange_clients or {}

        # Coordinator reference (assigned lazily when running under ProductionCoordinator)
        self.strategy_coordinator: Optional['StrategyCoordinator'] = strategy_coordinator
        
        # Initialize sub-managers
        self.order_manager = SmartOrderManager(risk_manager, exchange_clients)
        self.position_manager = AdvancedPositionManager(portfolio_manager, risk_manager, websocket_manager)
        self.execution_analytics = ExecutionAnalytics(self.order_manager, self.position_manager)
        
        # Engine state
        self.state = EngineState.STOPPED
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
        self.config = LiveTradingConfiguration.load(log_summary=False)
        
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
            if self.ws_manager:
                logger.info("  ✓ WebSocket manager ready")
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
            
            # [Phase 3.4.1] Prefetch historical data for indicator warmup
            logger.info("\n[Phase 3.4.1] Prefetching historical data for indicator warmup...")
            await self._prefetch_historical_data()
            logger.info("  ✓ Historical data prefetch complete")
            
            # Transition the engine state before background loops execute so they observe RUNNING.
            self.state = EngineState.RUNNING

            # Start signal processing
            signal_task = asyncio.create_task(self._signal_processing_loop())
            self.tasks.append(signal_task)
            logger.info("  ✓ Signal processing started")

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
            
            # Log exit summary statistics
            if self.position_manager:
                self.position_manager.log_exit_summary()
            
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
            
            # [EXECUTION START] Log signal execution start
            logger.info(f"[EXECUTION-START] Processing signal for {symbol}")
            
            # Enhanced logging for adaptive signals
            if signal.get('is_adaptive'):
                logger.info(f"🎯 Executing ADAPTIVE signal for {symbol}")
                if signal.get('adaptive_threshold'):
                    logger.info(f"  Adaptive RSI threshold: {signal['adaptive_threshold']:.1f}")
                if signal.get('position_multiplier'):
                    logger.info(f"  Position size multiplier: {signal['position_multiplier']:.2f}")
            else:
                logger.info(f"📊 Executing signal for {symbol}")
            
            # Log signal details
            logger.info(f"  Strategy: {signal.get('strategy', 'unknown')}")
            logger.info(f"  Side: {signal.get('side', 'unknown').upper()}")
            logger.info(f"  Entry: ${signal.get('entry', 0):.2f}")
            logger.info(f"  Reason: {signal.get('reason', 'N/A')}")
            
            # Step 1: Risk validation (Phase 3.2)
            portfolio_state = self.portfolio_manager.portfolio_state if self.portfolio_manager else {}
            risk_validation = await self.risk_manager.validate_new_position(signal, portfolio_state)
            
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
                
                # Apply adaptive position sizing if available
                if signal.get('position_multiplier'):
                    position_size *= signal['position_multiplier']
                    logger.info(f"  Applied position multiplier: {signal['position_multiplier']:.2f}")
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
            
            # Step 4: Determine execution algorithm
            notional_value = position_size * signal.get('entry', 0)
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
    
    async def _prefetch_historical_data(self):
        """
        Prefetch historical OHLCV data for all symbols before starting signal generation.
        
        This ensures indicators have sufficient data for calculation from the start,
        preventing the "cold start" problem where no signals are generated due to
        insufficient bars for RSI/ATR/EMA calculations.
        
        Critical for short sessions (<30 min) where WebSocket accumulation is too slow.
        
        Note: Symbols must be set via _cached_symbols before calling this method.
        """
        try:
            # Use cached symbols that should be set by ProductionCoordinator
            symbols = self._cached_symbols or []
            
            if not symbols:
                logger.warning("[PREFETCH] No symbols to prefetch (no symbols configured)")
                return
            
            logger.info(f"[PREFETCH] Fetching historical data for {len(symbols)} symbols...")
            logger.info(f"[PREFETCH] Symbols: {', '.join(symbols)}")
            
            prefetch_count = 0
            failed_count = 0
            
            for symbol in symbols:
                try:
                    logger.debug(f"[PREFETCH] Fetching {symbol}...")
                    
                    # Fetch historical data for all required timeframes
                    # Using 200 bars to ensure sufficient data for all indicators
                    df_30m = self._get_ohlcv_with_priority(symbol, '30m', limit=200)
                    df_1h = self._get_ohlcv_with_priority(symbol, '1h', limit=200)
                    df_4h = self._get_ohlcv_with_priority(symbol, '4h', limit=200)
                    
                    # Check if we got sufficient data
                    if df_30m is not None and len(df_30m) >= 14:
                        prefetch_count += 1
                        bars_30m = len(df_30m)
                        bars_1h = len(df_1h) if df_1h is not None else 0
                        bars_4h = len(df_4h) if df_4h is not None else 0
                        
                        logger.info(f"  ✓ {symbol}: {bars_30m} bars (30m), {bars_1h} bars (1h), {bars_4h} bars (4h)")
                        
                        # Verify indicators can be calculated
                        if bars_30m < 14:
                            logger.warning(f"  ⚠️ {symbol}: Insufficient bars for RSI/ATR ({bars_30m} < 14)")
                    else:
                        failed_count += 1
                        bar_count = len(df_30m) if df_30m is not None else 0
                        logger.warning(f"  ❌ {symbol}: Insufficient data - only {bar_count} bars")
                
                except Exception as e:
                    failed_count += 1
                    logger.error(f"  ❌ {symbol}: Prefetch failed - {e}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.05)
            
            logger.info("")
            logger.info(f"[PREFETCH] Complete: {prefetch_count} success, {failed_count} failed out of {len(symbols)} symbols")
            
            if prefetch_count == 0:
                logger.error("[PREFETCH] ⚠️ WARNING: No symbols have sufficient historical data!")
                logger.error("[PREFETCH] Signal generation may be delayed or impossible in short sessions.")
            elif failed_count > 0:
                logger.warning(f"[PREFETCH] ⚠️ {failed_count} symbols failed to prefetch - may have delayed signal generation")
            else:
                logger.info("[PREFETCH] ✅ All symbols ready for signal generation with full indicator data")
        
        except Exception as e:
            logger.error(f"[PREFETCH] Fatal error during historical data prefetch: {e}", exc_info=True)
            logger.warning("[PREFETCH] Continuing anyway - signal generation may be delayed")
    
    def _determine_default_exchange(self, symbol: str) -> Optional[str]:
        """Determine default exchange for a generated signal."""

        if symbol and ':' in symbol:
            exchange_hint = symbol.split(':')[-1].lower()
            if exchange_hint in self.exchange_clients:
                return exchange_hint

        if self.exchange_clients:
            return next(iter(self.exchange_clients.keys()))

        return self.config.get('execution', {}).get('default_exchange')

    def _get_ohlcv_with_priority(self, symbol: str, timeframe: str, limit: int = 100):
        """
        Get OHLCV data with WebSocket priority and REST fallback.
        
        Logic:
        1. Try WebSocket if enabled and available
        2. Check data freshness via ws_manager.is_data_fresh()
        3. Validate data has enough candles (>= limit)
        4. Record metrics (latency, success)
        5. Fallback to REST if WebSocket fails
        6. Return pandas DataFrame with OHLCV
        """
        start_time = time.time()
        
        # Try WebSocket first
        if self.ws_config['priority_enabled'] and self.ws_manager:
            try:
                if self.ws_manager.is_data_fresh(symbol, timeframe, self.ws_config['max_data_age_seconds']):
                    ws_data = self.ws_manager.get_latest_data(symbol, timeframe)
                    if ws_data and 'ohlcv' in ws_data and len(ws_data['ohlcv']) >= limit:
                        latency = (time.time() - start_time) * 1000
                        self._record_ws_fetch(latency, success=True)
                        logger.debug(f"📡 [WS-DATA] {symbol} {timeframe} - {len(ws_data['ohlcv'])} candles, {latency:.1f}ms")
                        return pd.DataFrame(ws_data['ohlcv'][-limit:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                logger.debug(f"WebSocket fetch failed: {e}")
                self._record_ws_fetch(0, success=False)
        
        # Fallback to REST
        logger.debug(f"🌐 [REST-DATA] {symbol} {timeframe}")
        return self._fetch_ohlcv_rest(symbol, timeframe, limit, start_time)

    def _fetch_ohlcv_rest(self, symbol: str, timeframe: str, limit: int, start_time: float):
        """Fetch OHLCV from REST with metrics."""
        for exchange_name, client in self.exchange_clients.items():
            try:
                ohlcv = client.ohlcv(symbol, timeframe, limit=limit)
                latency = (time.time() - start_time) * 1000
                self._record_rest_fetch(latency, success=True)
                logger.debug(f"🌐 [REST-SUCCESS] {symbol} - {latency:.1f}ms")
                return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                logger.error(f"REST fetch failed: {e}")
                self._record_rest_fetch(0, success=False)
        return None

    def _record_ws_fetch(self, latency_ms: float, success: bool):
        """Record WebSocket fetch metrics."""
        if success:
            self.ws_stats['websocket_fetches'] += 1
            self.ws_stats['total_latency_ws'] += latency_ms
            self.ws_stats['avg_latency_ws'] = self.ws_stats['total_latency_ws'] / self.ws_stats['websocket_fetches']
            self.ws_stats['last_ws_fetch_time'] = time.time()
            self.ws_stats['consecutive_ws_failures'] = 0
        else:
            self.ws_stats['websocket_failures'] += 1
            self.ws_stats['consecutive_ws_failures'] += 1
        
        total = self.ws_stats['websocket_fetches'] + self.ws_stats['websocket_failures']
        if total > 0:
            self.ws_stats['websocket_success_rate'] = (self.ws_stats['websocket_fetches'] / total * 100)

    def _record_rest_fetch(self, latency_ms: float, success: bool):
        """Record REST fetch metrics."""
        if success:
            self.ws_stats['rest_fetches'] += 1
            self.ws_stats['total_latency_rest'] += latency_ms
            self.ws_stats['avg_latency_rest'] = self.ws_stats['total_latency_rest'] / self.ws_stats['rest_fetches']
            self.ws_stats['last_rest_fetch_time'] = time.time()

    def get_websocket_stats(self):
        """Get comprehensive WebSocket statistics."""
        stats = self.ws_stats.copy()
        if self.ws_manager:
            stats['connection_health'] = self.ws_manager.get_connection_health()
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
                    data = client.fetch_ohlcv_bulk(symbol, timeframe=timeframe, target_limit=limit)
                elif hasattr(client, 'ohlcv'):
                    # Use standard method
                    data = client.ohlcv(symbol, timeframe, limit=limit)
                else:
                    logger.warning(f"Exchange client {exchange_name} does not support OHLCV fetching")
                    continue
                
                if data and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        data,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    logger.debug(f"✅ Fetched {len(df)} candles for {symbol} {timeframe} from {exchange_name}")
                    return df
                    
            except Exception as e:
                logger.debug(f"Could not fetch {symbol} {timeframe} from {exchange_name}: {e}")
                continue
        
        logger.warning(f"❌ Failed to fetch {symbol} {timeframe} from all exchanges")
        return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price with WebSocket/REST fallback.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        # Try WebSocket first
        if self.ws_manager:
            try:
                ws_data = self.ws_manager.get_latest_data(symbol, '1m')
                if ws_data and ws_data.get('ohlcv'):
                    latest_candle = ws_data['ohlcv'][-1]
                    return float(latest_candle[4])  # Close price
            except Exception as e:
                logger.debug(f"WebSocket price fetch failed: {e}")
        
        # Fallback to REST API
        for exchange_name, client in self.exchange_clients.items():
            try:
                ticker = client.fetch_ticker(symbol)
                price = ticker.get('last', ticker.get('close', 0))
                if price > 0:
                    return float(price)
            except Exception as e:
                logger.debug(f"REST price fetch failed for {exchange_name}: {e}")
        
        logger.warning(f"Could not fetch price for {symbol}")
        return None
    
    async def _position_monitoring_loop(self):
        """Enhanced position monitoring with real-time P&L and exit checking."""
        logger.info("Position monitoring loop started")
        
        # Reduce interval to 10 seconds for faster response
        interval = 10
        
        try:
            while self.state == EngineState.RUNNING:
                if not self.active_positions:
                    await asyncio.sleep(interval)
                    continue
                
                logger.debug(f"Monitoring {len(self.active_positions)} active positions")
                
                # Track summary statistics
                total_unrealized_pnl = 0.0
                positions_closed_count = 0
                
                for position_id in list(self.active_positions.keys()):
                    try:
                        position = self.active_positions.get(position_id)
                        if not position:
                            continue
                        
                        symbol = position.get('symbol')
                        entry_price = position.get('entry_price', 0)
                        
                        # Fetch current price
                        current_price = await self._get_current_price(symbol)
                        
                        if current_price is None or current_price <= 0:
                            logger.warning(f"Invalid price for {symbol}, skipping")
                            continue
                        
                        # Update P&L
                        pnl_result = await self.position_manager.monitor_position_pnl(
                            position_id,
                            current_price
                        )
                        
                        if pnl_result.get('success'):
                            unrealized_pnl = pnl_result.get('unrealized_pnl', 0)
                            pnl_pct = pnl_result.get('pnl_pct', 0)
                            
                            # Enhanced P&L logging
                            logger.info(
                                f"💰 [P&L-UPDATE] {position_id}\n"
                                f"   Symbol: {symbol}\n"
                                f"   Entry: ${entry_price:.2f}\n"
                                f"   Current: ${current_price:.2f}\n"
                                f"   Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
                            )
                            
                            # Track total unrealized P&L
                            total_unrealized_pnl += unrealized_pnl
                        
                        # Check exit conditions
                        exit_check = await self.position_manager.manage_position_exits(position_id)
                        
                        if exit_check.get('should_exit'):
                            exit_reason = exit_check.get('exit_reason')
                            exit_emoji = '🛑' if exit_reason == 'stop_loss' else '🎯'
                            
                            # Enhanced exit logging
                            logger.warning(
                                f"{exit_emoji} [EXIT-SIGNAL] {position_id}\n"
                                f"   Symbol: {symbol}\n"
                                f"   Reason: {exit_reason.upper()}\n"
                                f"   Entry: ${entry_price:.2f}\n"
                                f"   Exit: ${current_price:.2f}\n"
                                f"   P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)"
                            )
                            
                            # Execute exit
                            await self._execute_position_exit(position_id, exit_check)
                            positions_closed_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error monitoring position {position_id}: {e}")
                        continue
                
                # Summary logging at end of monitoring loop
                if self.active_positions:
                    logger.info(
                        f"📊 [MONITORING-SUMMARY]\n"
                        f"   Active Positions: {len(self.active_positions)}\n"
                        f"   Total Unrealized P&L: ${total_unrealized_pnl:.2f}\n"
                        f"   Positions Closed This Cycle: {positions_closed_count}"
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
                    
                    # Monitor active orders for timeouts, partial fills, etc.
                    # Implementation would check order status and take action
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in order management loop: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("Order management loop cancelled")
    
    async def _performance_reporting_loop(self):
        """Background task for performance reporting."""
        logger.info("Performance reporting loop started")
        
        # Default interval if not in config
        interval = self.config.get('monitoring', {}).get('performance_report_interval', 3600)
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    # Generate performance report
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
    
    async def _execute_position_exit(self, position_id: str, exit_signal: Dict):
        """Execute position exit based on exit signal."""
        try:
            logger.info(f"Executing exit for position {position_id}")
            
            position = self.active_positions.get(position_id)
            if not position:
                logger.warning(f"Position not found: {position_id}")
                return
            
            # Log exit details
            logger.info(f"  Symbol: {position.get('symbol')}")
            logger.info(f"  Side: {position.get('side')}")
            logger.info(f"  Amount: {position.get('amount')}")
            logger.info(f"  Exit reason: {exit_signal.get('exit_reason')}")
            
            # Create exit order
            exit_order = {
                'symbol': position['symbol'],
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'amount': position['amount'],
                'exchange': position['exchange']
            }
            
            # Execute exit order
            execution_result = await self.order_manager.place_order(exit_order, 'market')
            
            if execution_result.get('success'):
                # Close position
                exit_price = execution_result.get('avg_price', 0)
                close_result = await self.position_manager.close_position(
                    position_id,
                    exit_price,
                    exit_signal.get('exit_reason')
                )
                
                if close_result['success']:
                    logger.info(f"✅ Position closed successfully: {position_id}")
                    logger.info(f"   Exit price: ${exit_price:.2f}")
                    logger.info(f"   P&L: {close_result.get('pnl', 0):.2%}")
                    
                    # Remove from active positions
                    if position_id in self.active_positions:
                        del self.active_positions[position_id]
                else:
                    logger.error(f"Failed to close position: {close_result.get('reason')}")
            else:
                logger.error(f"Exit order failed: {execution_result.get('reason')}")
                
        except Exception as e:
            logger.error(f"Error executing position exit: {e}", exc_info=True)
    
    async def _initialize_risk_management(self) -> Dict[str, Any]:
        """Initialize risk management systems."""
        try:
            # Risk manager should already be initialized
            if self.risk_manager:
                logger.info("  Risk manager initialized")
                return {'success': True}
            else:
                logger.warning("  No risk manager provided")
                return {'success': True}  # Allow running without risk manager in paper mode
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    async def _initialize_portfolio_management(self) -> Dict[str, Any]:
        """Initialize portfolio management systems."""
        try:
            # Portfolio manager should already be initialized
            if self.portfolio_manager:
                logger.info("  Portfolio manager initialized")
                
                # Log registered strategies if any
                if hasattr(self.portfolio_manager, 'strategies'):
                    strategies = self.portfolio_manager.strategies
                    logger.info(f"  Registered strategies: {list(strategies.keys())}")
                    
                    # Check for adaptive strategies
                    adaptive_count = sum(1 for name in strategies.keys() if 'adaptive' in name.lower())
                    if adaptive_count > 0:
                        logger.info(f"  🎯 Adaptive strategies: {adaptive_count}")
                        
                return {'success': True}
            else:
                logger.warning("  No portfolio manager provided")
                return {'success': True}  # Allow running without portfolio manager in paper mode
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
            'signals_received': self._signal_count,  # Fixed: was 'total_signals', now correctly named
            'signals_executed': self._executed_count,  # Track executed signals
            'total_signals': self._signal_count,  # Keep for backward compatibility
            'last_signal_time': self._last_signal_time.isoformat() if self._last_signal_time else None,
            'active_tasks': len([t for t in self.tasks if not t.done()]),
            'signal_queue_size': self.signal_queue.qsize(),
            'universe_size': len(self._cached_symbols) if self._cached_symbols else 0,
            'config': {
                'fixed_symbols': len(self.config.get('universe', {}).get('fixed_symbols', [])),
                'auto_select': self.config.get('universe', {}).get('auto_select', False)
            }
        }
        
        # Add manager stats if available
        if self.order_manager and hasattr(self.order_manager, 'get_execution_statistics'):
            status['execution_stats'] = self.order_manager.get_execution_statistics()
            
        if self.position_manager and hasattr(self.position_manager, 'get_position_summary'):
            status['position_summary'] = self.position_manager.get_position_summary()
        
        return status
