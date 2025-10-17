"""
Live Trading Engine.
Production-ready live trading execution engine with enhanced debugging and adaptive strategies.
"""
import os
import yaml
import asyncio
import logging
import inspect
import pandas as pd
from typing import Dict, List, Optional, Any
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

# Core imports
from .order_manager import SmartOrderManager
from .position_manager import AdvancedPositionManager
from .execution_analytics import ExecutionAnalytics

# Import config with try/except for flexibility
try:
    from ..config.live_trading_config import LiveTradingConfiguration
except ImportError:
    from config.live_trading_config import LiveTradingConfiguration

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
                 websocket_manager=None, exchange_clients=None):
        """
        Initialize live trading engine.
        
        Args:
            mode: Trading mode ('paper', 'live', 'simulation')
            portfolio_manager: PortfolioManager from Phase 3.3
            risk_manager: RiskManager from Phase 3.2
            websocket_manager: WebSocketManager from Phase 3.1
            exchange_clients: Dict of exchange client instances from Phase 1
        """
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.ws_manager = websocket_manager
        self.exchange_clients = exchange_clients or {}
        
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
        
        # Load configuration
        self.config = LiveTradingConfiguration.get_all_configs()

        # ✅ DÜZELTME 2: Universe config'i güvenli yükle
        config_path = os.getenv("CONFIG_PATH", "config/config.example.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
                
            # Universe config'i güvenli yükle ve validate et
            if 'universe' in yaml_config:
                self.config['universe'] = yaml_config['universe']
                
                # Type validation ve default değerler
                if not isinstance(self.config['universe'].get('fixed_symbols'), list):
                    self.config['universe']['fixed_symbols'] = []
                
                # auto_select default False olmalı
                if 'auto_select' not in self.config['universe']:
                    self.config['universe']['auto_select'] = False
                    
                logger.info(f"✅ Universe config loaded from YAML")
                logger.info(f"   fixed_symbols: {len(self.config['universe'].get('fixed_symbols', []))} symbols")
                logger.info(f"   auto_select: {self.config['universe'].get('auto_select')}")
            else:
                # Default universe config
                self.config['universe'] = {
                    'fixed_symbols': [],
                    'auto_select': False
                }
                logger.info("ℹ️ Using default universe config (no fixed symbols)")
                
        except Exception as e:
            logger.warning(f"Could not load universe from YAML: {e}, using defaults")
            self.config['universe'] = {'fixed_symbols': [], 'auto_select': False}
        
        # Universe cache for optimization
        self._cached_symbols = None
        self._universe_built = False
        
        # Performance tracking
        self._signal_count = 0
        self._last_signal_time = None
        
        logger.info("LiveTradingEngine initialized")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Exchange clients: {list(exchange_clients.keys())}")
    
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
            
            self.state = EngineState.RUNNING
            
            logger.info("\n" + "="*70)
            logger.info("✓ LIVE TRADING ENGINE STARTED SUCCESSFULLY")
            logger.info("="*70)
            
            return {
                'success': True,
                'state': self.state.value,
                'mode': self.mode.value,
                'active_tasks': len(self.tasks),
                'universe_size': len(self._get_scan_symbols())
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
            
            logger.info("Live trading engine stopped")
            logger.info(f"  Total signals generated: {self._signal_count}")
            
            return {
                'success': True,
                'state': self.state.value,
                'total_signals': self._signal_count
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
                return {
                    'success': False,
                    'reason': f"Risk validation failed: {risk_validation[1]}",
                    'stage': 'risk_validation'
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
            logger.info("="*50)
            
            return {
                'success': True,
                'position_id': position_id,
                'order_id': execution_result.get('order_id'),
                'execution_result': execution_result,
                'position_result': position_result
            }
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)
            return {
                'success': False,
                'reason': str(e),
                'stage': 'execution_error'
            }
    
    async def _signal_processing_loop(self):
        """Background task for processing signals from queue and generating new signals."""
        logger.info("Signal processing loop started")
        
        # Import indicators
        try:
            from ..core.indicators import add_indicators
        except ImportError:
            from core.indicators import add_indicators
        
        # Import market regime analyzer
        try:
            from ..core.market_regime import MarketRegimeAnalyzer
        except ImportError:
            from core.market_regime import MarketRegimeAnalyzer
        
        # Initialize regime analyzer
        regime_analyzer = MarketRegimeAnalyzer()
        
        # Track last scan time to avoid too frequent scans
        last_scan_time = 0
        scan_interval = 30  # 30 seconds between market scans
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    current_time = asyncio.get_event_loop().time()
                    
                    # Market scanning phase (every 30 seconds)
                    if current_time - last_scan_time >= scan_interval:
                        logger.debug("🔍 Market scan starting...")
                        last_scan_time = current_time
                        
                        # Get symbols to scan
                        symbols = self._get_scan_symbols()
                        logger.info(f"🔍 Scanning {len(symbols)} symbols")
                        
                        for symbol in symbols:
                            try:
                                logger.info(f"[PROCESSING] Symbol: {symbol}")
                                
                                # Fetch OHLCV data for multiple timeframes
                                df_30m = await self._fetch_ohlcv(symbol, '30m', limit=200)
                                df_1h = await self._fetch_ohlcv(symbol, '1h', limit=200)
                                df_4h = await self._fetch_ohlcv(symbol, '4h', limit=200)
                                
                                if df_30m is None or len(df_30m) == 0:
                                    logger.debug(f"⚠️ No data for {symbol}, skipping")
                                    continue
                                
                                # Log data fetching result
                                last_close = df_30m['close'].iloc[-1] if len(df_30m) > 0 else 0
                                logger.info(f"[DATA] {symbol}: 30m={len(df_30m)} bars, last_close=${last_close:.2f}")
                                
                                # Add technical indicators
                                indicator_config = self.config.get('indicators', {})
                                df_30m = add_indicators(df_30m, indicator_config)
                                
                                if df_1h is not None and len(df_1h) > 0:
                                    df_1h = add_indicators(df_1h, indicator_config)
                                if df_4h is not None and len(df_4h) > 0:
                                    df_4h = add_indicators(df_4h, indicator_config)
                                
                                # ✅ DÜZELTME 3: Enhanced indicator logging
                                if 'rsi' in df_30m.columns and len(df_30m) > 0:
                                    last_rsi = df_30m['rsi'].iloc[-1]
                                    last_atr = df_30m['atr'].iloc[-1] if 'atr' in df_30m.columns else 0
                                    last_ema21 = df_30m['ema21'].iloc[-1] if 'ema21' in df_30m.columns else 0
                                    last_ema50 = df_30m['ema50'].iloc[-1] if 'ema50' in df_30m.columns else 0
                                    
                                    logger.info(f"[INDICATORS] {symbol}: RSI={last_rsi:.1f}, ATR=${last_atr:.2f}, "
                                              f"EMA21=${last_ema21:.2f}, EMA50=${last_ema50:.2f}")
                                
                                # Perform market regime analysis if we have all timeframes
                                regime_data = None
                                if df_1h is not None and df_4h is not None and len(df_1h) > 0 and len(df_4h) > 0:
                                    regime_data = regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
                                    logger.info(f"📊 {symbol} Regime Analysis:")
                                    logger.info(f"   Trend: {regime_data.get('trend', 'unknown')}")
                                    logger.info(f"   Momentum: {regime_data.get('momentum', 'unknown')}")
                                    logger.info(f"   Volatility: {regime_data.get('volatility', 'unknown')}")
                                
                                # Run strategies registered in portfolio manager
                                if self.portfolio_manager and hasattr(self.portfolio_manager, 'strategies'):
                                    for strategy_name, strategy in self.portfolio_manager.strategies.items():
                                        try:
                                            logger.info(f"[STRATEGY-CHECK] Running {strategy_name} for {symbol}")
                                            
                                            # ✅ DÜZELTME 4: Enhanced adaptive detection and logging
                                            is_adaptive = False
                                            adaptive_info = ""
                                            adaptive_threshold = None
                                            position_multiplier = 1.0
                                            
                                            # Check if this is an adaptive strategy
                                            if 'adaptive' in strategy_name.lower() or hasattr(strategy, 'get_adaptive_rsi_threshold'):
                                                is_adaptive = True
                                                
                                                # Get adaptive parameters if available
                                                if regime_data and hasattr(strategy, 'get_adaptive_rsi_threshold'):
                                                    adaptive_threshold = strategy.get_adaptive_rsi_threshold(regime_data)
                                                    adaptive_info = f" (adaptive RSI: {adaptive_threshold:.1f})"
                                                    logger.info(f"🎯 [ADAPTIVE] {strategy_name} using dynamic RSI threshold: {adaptive_threshold:.1f}")
                                                    
                                                    # Get position size multiplier if available
                                                    if hasattr(strategy, 'calculate_dynamic_position_size'):
                                                        volatility = regime_data.get('volatility', 'normal')
                                                        position_multiplier = strategy.calculate_dynamic_position_size(volatility)
                                                        logger.info(f"   Position multiplier: {position_multiplier:.2f} (volatility: {volatility})")
                                            
                                            # ✅ DÜZELTME 5: Debug log BEFORE strategy call
                                            if len(df_30m) > 0:
                                                last_30m_row = df_30m.iloc[-1]
                                                logger.info(f"[DEBUG-PRE] Calling {strategy_name} for {symbol}")
                                                logger.info(f"[DEBUG-PRE] Data: RSI={last_30m_row.get('rsi', 0):.2f}, "
                                                          f"Close=${last_30m_row.get('close', 0):.2f}, "
                                                          f"ATR=${last_30m_row.get('atr', 0):.4f}")
                                            
                                            signal = None
                                            
                                            # Check if strategy has signal method
                                            if not hasattr(strategy, 'signal'):
                                                logger.debug(f"Strategy {strategy_name} has no signal method, skipping")
                                                continue
                                            
                                            # Determine strategy requirements by inspecting method signature
                                            sig = inspect.signature(strategy.signal)
                                            params = list(sig.parameters.keys())
                                            
                                            # Check for specific parameter names
                                            has_regime_param = 'regime_data' in params
                                            has_df_1h_param = 'df_1h' in params
                                            
                                            # Call strategy with appropriate parameters
                                            try:
                                                if has_regime_param:
                                                    # Adaptive strategy with regime awareness
                                                    if has_df_1h_param and df_1h is not None:
                                                        signal = strategy.signal(df_30m, df_1h, regime_data)
                                                    else:
                                                        signal = strategy.signal(df_30m, regime_data)
                                                else:
                                                    # Base strategy without regime awareness
                                                    if has_df_1h_param and df_1h is not None:
                                                        signal = strategy.signal(df_30m, df_1h)
                                                    else:
                                                        signal = strategy.signal(df_30m)
                                            except TypeError as te:
                                                # Fallback: try calling with just df_30m
                                                logger.warning(f"Strategy {strategy_name} parameter mismatch: {te}")
                                                try:
                                                    signal = strategy.signal(df_30m)
                                                except Exception as e2:
                                                    logger.error(f"Strategy {strategy_name} failed: {e2}")
                                                    continue
                                            
                                            # ✅ DÜZELTME 6: Debug log AFTER strategy call
                                            if signal:
                                                logger.info(f"✅ [SIGNAL] {strategy_name} generated signal for {symbol}")
                                            else:
                                                if 'rsi' in df_30m.columns and len(df_30m) > 0:
                                                    last_rsi = df_30m['rsi'].iloc[-1]
                                                    if is_adaptive and adaptive_threshold:
                                                        logger.debug(f"[DEBUG-POST] {strategy_name} returned None for {symbol} "
                                                                   f"(RSI={last_rsi:.1f}, Adaptive threshold={adaptive_threshold:.1f})")
                                                    else:
                                                        logger.debug(f"[DEBUG-POST] {strategy_name} returned None for {symbol} (RSI={last_rsi:.1f})")
                                                else:
                                                    logger.debug(f"[DEBUG-POST] {strategy_name} returned None for {symbol}")
                                            
                                            # If signal generated, enrich and add to queue
                                            if signal:
                                                # ✅ DÜZELTME 7: Complete signal enrichment
                                                # Add adaptive information
                                                if is_adaptive:
                                                    signal['is_adaptive'] = True
                                                    signal['adaptive_info'] = adaptive_info
                                                    if adaptive_threshold:
                                                        signal['adaptive_threshold'] = adaptive_threshold
                                                    if position_multiplier != 1.0:
                                                        signal['position_multiplier'] = position_multiplier
                                                
                                                # CRITICAL: Add required metadata
                                                signal['symbol'] = symbol  # ✅ MUTLAKA EKLENMELI
                                                signal['strategy'] = strategy_name
                                                signal['timestamp'] = datetime.now(timezone.utc).isoformat()
                                                
                                                # Add current price (entry) - CRITICAL
                                                if len(df_30m) > 0:
                                                    signal['entry'] = float(df_30m['close'].iloc[-1])  # ✅ KRİTİK
                                                    
                                                    # Add ATR for stop loss calculation
                                                    if 'atr' in df_30m.columns:
                                                        signal['atr'] = float(df_30m['atr'].iloc[-1])  # ✅ ÖNEMLİ
                                                
                                                # Add regime data if available
                                                if regime_data:
                                                    signal['regime_data'] = regime_data
                                                
                                                # Add to signal queue
                                                await self.signal_queue.put(signal)
                                                self._signal_count += 1
                                                self._last_signal_time = datetime.now(timezone.utc)
                                                
                                                # Log signal details
                                                logger.info(f"📊 Signal Details:")
                                                logger.info(f"   Symbol: {symbol}")
                                                logger.info(f"   Strategy: {strategy_name}{adaptive_info}")
                                                logger.info(f"   Side: {signal.get('side', 'unknown').upper()}")
                                                logger.info(f"   Entry: ${signal.get('entry', 0):.2f}")
                                                logger.info(f"   Reason: {signal.get('reason', 'N/A')}")
                                                if is_adaptive:
                                                    logger.info(f"   🎯 ADAPTIVE: RSI threshold={adaptive_threshold:.1f}")
                                                    if position_multiplier != 1.0:
                                                        logger.info(f"   Position multiplier: {position_multiplier:.2f}")
                                        
                                        except Exception as e:
                                            logger.error(f"Error running strategy {strategy_name} for {symbol}: {e}", exc_info=True)
                                            continue
                                
                            except Exception as e:
                                logger.error(f"Error scanning {symbol}: {e}")
                                continue
                        
                        logger.info(f"✓ Market scan completed. Signals in queue: {self.signal_queue.qsize()}")
                    
                    # Signal processing phase (check queue with short timeout)
                    try:
                        # Check if there are signals in queue to process
                        signal = await asyncio.wait_for(
                            self.signal_queue.get(),
                            timeout=1.0  # 1 second timeout
                        )
                        
                        # Process signal
                        logger.info(f"📤 Processing signal from queue: {signal.get('symbol', 'unknown')}")
                        result = await self.execute_signal(signal)
                        
                        if result['success']:
                            logger.info(f"✅ Signal processed successfully: {signal.get('symbol')}")
                        else:
                            logger.warning(f"⚠️ Signal processing failed: {result.get('reason')}")
                    
                    except asyncio.TimeoutError:
                        # No signals in queue, continue scanning loop
                        await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                        continue
                
                except asyncio.CancelledError:
                    logger.info("Signal processing loop cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error in signal processing loop: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Wait before retrying after error
                    
        except asyncio.CancelledError:
            logger.info("Signal processing loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in signal processing loop: {e}", exc_info=True)
    
    def _get_scan_symbols(self) -> List[str]:
        """Get symbols to scan - optimized for fixed symbols mode."""
        
        # Cache kontrolü
        if self._cached_symbols:
            return self._cached_symbols
        
        # Config'den direkt oku
        cfg = self.config
        universe_cfg = cfg.get('universe', {})
        
        # Sabit liste var mı?
        fixed_symbols = universe_cfg.get('fixed_symbols', [])
        auto_select = universe_cfg.get('auto_select', False)
        
        # ✅ DÜZELTME 8: Güvenli kontrol ve type checking
        if fixed_symbols and isinstance(fixed_symbols, list) and not auto_select:
            # Direkt kullan, market yükleme YOK!
            logger.info(f"[UNIVERSE] ✅ Using {len(fixed_symbols)} FIXED symbols (no market loading)")
            
            # Validate symbols
            valid_symbols = []
            for symbol in fixed_symbols:
                if isinstance(symbol, str) and '/' in symbol:
                    valid_symbols.append(symbol)
                else:
                    logger.warning(f"[UNIVERSE] Invalid symbol format: {symbol}")
            
            self._cached_symbols = valid_symbols
            
            # Exchange client'lara bildir (eğer varsa)
            if self.exchange_clients:
                for client in self.exchange_clients.values():
                    if hasattr(client, 'set_required_symbols'):
                        client.set_required_symbols(valid_symbols)
            
            logger.info(f"[UNIVERSE] Symbols: {', '.join(valid_symbols[:5])}..." if valid_symbols else "No symbols")
            return valid_symbols
        
        # Auto-select mode (önerilmez)
        if auto_select:
            logger.warning("[UNIVERSE] Auto-select mode enabled (will load all markets)")
            
            # Import universe builder
            try:
                from universe import build_universe
                
                # Build universe using exchange clients
                universe_dict = build_universe(self.exchange_clients, cfg)
                
                # Flatten all symbols from all exchanges
                all_symbols = []
                for exchange_symbols in universe_dict.values():
                    all_symbols.extend(exchange_symbols)
                
                # Remove duplicates
                unique_symbols = list(set(all_symbols))
                
                # Set required symbols on all clients
                if self.exchange_clients:
                    for client in self.exchange_clients.values():
                        if hasattr(client, 'set_required_symbols'):
                            client.set_required_symbols(unique_symbols)
                
                # Cache the result
                self._cached_symbols = unique_symbols
                
                logger.info(f"[UNIVERSE] Auto-selected {len(unique_symbols)} symbols")
                return unique_symbols
                
            except ImportError:
                logger.error("[UNIVERSE] Universe builder not available")
            except Exception as e:
                logger.error(f"[UNIVERSE] Error building universe: {e}")
        
        # Fallback to default symbols
        logger.warning("[UNIVERSE] Using default symbols (config not properly set)")
        default_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT',
            'LTC/USDT:USDT', 'AVAX/USDT:USDT'
        ]
        self._cached_symbols = default_symbols
        return default_symbols
    
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
    
    async def _position_monitoring_loop(self):
        """Background task for monitoring active positions."""
        logger.info("Position monitoring loop started")
        
        # Default interval if not in config
        interval = self.config.get('monitoring', {}).get('position_check_interval', 30)
        
        try:
            while self.state == EngineState.RUNNING:
                try:
                    if self.active_positions:
                        logger.debug(f"Monitoring {len(self.active_positions)} active positions")
                        
                    for position_id in list(self.active_positions.keys()):
                        # Check position status
                        exit_check = await self.position_manager.manage_position_exits(position_id)
                        
                        if exit_check.get('should_exit'):
                            logger.info(f"📤 Exit signal for {position_id}: {exit_check.get('exit_reason')}")
                            # Handle position exit
                            await self._execute_position_exit(position_id, exit_check)
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring positions: {e}")
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
            'total_signals': self._signal_count,
            'last_signal_time': self._last_signal_time.isoformat() if self._last_signal_time else None,
            'active_tasks': len([t for t in self.tasks if not t.done()]),
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
