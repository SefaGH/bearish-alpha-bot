"""
Production Coordinator - Phase 3 Orchestration Layer
Manages the complete production trading system with all phases integrated.
"""

import logging
import asyncio
import inspect
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import os
import yaml
import pandas as pd

# Phase 1: Multi-Exchange Framework
from .multi_exchange import build_clients_from_env
from .ccxt_client import CcxtClient

# Phase 2: Market Intelligence  
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # src/ dizinini ekle
from core.market_regime import MarketRegimeAnalyzer
from .performance_monitor import PerformanceMonitor
from .websocket_manager import WebSocketManager
from config.live_trading_config import LiveTradingConfiguration

# Performance Monitor için basit fallback
class RealTimePerformanceMonitor:
    """Basit performance monitor fallback."""
    def __init__(self):
        self.trades = []
        self.metrics = {}
        self.performance_history = {}
        self.optimization_feedback = {}
    
    def record_trade(self, trade_data):
        self.trades.append(trade_data)
    
    def get_metrics(self):
        return self.metrics
    
    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Production uyumluluğu için gerekli metod."""
        return {
            'strategy': strategy_name,
            'status': 'active',
            'metrics': {},
            'trade_count': 0
        }

# Phase 3.1-3.3: Risk & Portfolio Management
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .strategy_coordinator import StrategyCoordinator
from .circuit_breaker import CircuitBreakerSystem

# Phase 3.4: Live Trading Components
from .live_trading_engine import LiveTradingEngine

# Strategy imports - DÜZELTILDI
from strategies.adaptive_ob import AdaptiveOversoldBounce
from strategies.adaptive_str import AdaptiveShortTheRip

# Triple-fallback import strategy for maximum compatibility:
# 1. Direct utils import (when src/ is on sys.path)
# 2. Absolute src.utils import (when repo root is on sys.path)
# 3. Relative import (when imported as package module)
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
except ModuleNotFoundError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
    except ModuleNotFoundError as e:
        # Option 3: Relative import (package context)
        if e.name in ('src', 'src.utils', 'src.utils.pnl_calculator'):
            from ..utils.pnl_calculator import calculate_unrealized_pnl, calculate_pnl_percentage
        else:
            # Unknown module missing, re-raise
            raise
# Phase 4: ML Components (optional)
try:
    from ml.regime_predictor import RegimePredictor
    from ml.strategy_optimizer import StrategyOptimizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProductionCoordinator:
    """Coordinate all Phase 3 components for production deployment."""
    
    def __init__(self):
        """Initialize production coordinator."""
        # Phase 3 components (will be initialized)
        self.websocket_manager = None  # Phase 3.1
        self.risk_manager = None       # Phase 3.2
        self.portfolio_manager = None  # Phase 3.3
        self.trading_engine = None     # Phase 3.4
        self.strategy_coordinator = None
        self.circuit_breaker = None
        
        # Phase 2 components
        self.market_regime_analyzer = None  # Phase 2
        self.performance_monitor = None     # Phase 2
        
        # Phase 1 components
        self.exchange_clients = {}          # Phase 1
        
        # Registered strategies
        self.strategies = {}  # strategy_name -> strategy_instance
        self.strategy_capabilities = {}  # strategy_name -> {supports_regime_data, is_async}
        
        # System state
        self.is_running = False
        self.is_initialized = False
        self.emergency_stop_triggered = False
        self.loop_interval = 30  # Ana döngü bekleme süresi
        self.active_symbols = []  # Takip edilen semboller
        self.processed_symbols_count = 0  # İşlenen sembol sayacı
        
        # Signal lifecycle tracking
        self.signal_lifecycle = {}  # signal_id -> {stage, timestamp, details}
        
        # Configuration
        self.config = LiveTradingConfiguration.load()
        
        # Market regime analyzer başlat
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        logger.info("✅ Market regime analyzer initialized")
        
        logger.info("ProductionCoordinator created")

    def _setup_websocket_connections(self) -> bool:
        """
        Setup WebSocket connections with proper validation and error handling.
        
        Returns:
            True if any streams were started successfully, False otherwise
        """
        # Step 1: Validate Prerequisites
        if not self.exchange_clients:
            logger.error("[WS] ERROR: No exchange clients available")
            return False
        
        # Get symbols with fallback
        fixed_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
        if not fixed_symbols:
            # Fallback to default symbols
            fixed_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
            logger.warning(f"[WS] No symbols configured, using defaults: {fixed_symbols}")
        
        logger.info(f"[WS] Setting up WebSocket streams for {len(fixed_symbols)} symbols")
        
        # Step 2: Initialize Manager
        try:
            if not self.websocket_manager:
                self.websocket_manager = WebSocketManager(exchanges=self.exchange_clients)
                logger.info("[WS] WebSocket manager created")
        except Exception as e:
            logger.error(f"[WS] ERROR: Failed to create WebSocket manager: {type(e).__name__}: {str(e)}")
            return False
        
        # Get timeframes from config
        ws_config = self.config.get('websocket', {})
        timeframes = ws_config.get('stream_timeframes', ['1m', '5m'])
        
        # Step 3: Setup Streams with Limits
        total_streams_started = 0
        
        for exchange_name in self.exchange_clients.keys():
            # Get stream limit for this exchange
            stream_limit = self._get_stream_limit(exchange_name)
            logger.info(f"[WS] {exchange_name}: stream limit = {stream_limit}")
            
            # Calculate required streams
            required_streams = len(fixed_symbols) * len(timeframes)
            
            # Determine symbols to use (respect limit)
            if required_streams > stream_limit:
                max_symbols = stream_limit // len(timeframes)
                symbols_to_use = fixed_symbols[:max_symbols]
                logger.warning(
                    f"[WS] {exchange_name}: Required {required_streams} streams > limit {stream_limit}, "
                    f"using only first {max_symbols} symbols"
                )
            else:
                symbols_to_use = fixed_symbols
            
            # Start streams for this exchange
            exchange_streams_started = 0
            for symbol in symbols_to_use:
                for tf in timeframes:
                    try:
                        self.websocket_manager.start_ohlcv_stream(exchange_name, symbol, tf)
                        exchange_streams_started += 1
                    except Exception as e:
                        logger.error(f"[WS] ERROR: Failed to start stream {exchange_name}:{symbol}:{tf} - {str(e)}")
            
            total_streams_started += exchange_streams_started
            
            if exchange_streams_started > 0:
                logger.info(f"[WS] {exchange_name}: Started {exchange_streams_started} streams successfully")
                # Update active symbols for this exchange
                if not hasattr(self, 'active_symbols') or not self.active_symbols:
                    self.active_symbols = symbols_to_use
            else:
                logger.error(f"[WS] ERROR: {exchange_name}: No streams started")
        
        # Step 4: Return Status
        if total_streams_started == 0:
            logger.error("[WS] ERROR: No WebSocket streams were started")
            return False
        
        logger.info(f"[WS] ✅ Setup complete: {total_streams_started} total streams started")
        return True
    
    def _get_stream_limit(self, exchange_name: str) -> int:
        """
        Get stream limit for a specific exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Maximum number of streams allowed for this exchange
        """
        # Per-exchange stream limits
        stream_limits = {
            'bingx': 10,
            'binance': 20,
            'kucoinfutures': 15,
        }
        
        # Try to get from config first
        ws_config = self.config.get('websocket', {})
        max_streams_config = ws_config.get('max_streams_per_exchange', {})
        
        # Return config value, then hardcoded limit, then default
        return max_streams_config.get(
            exchange_name,
            stream_limits.get(exchange_name, 10)  # Default to 10
        )

    def _get_top_volume_symbols(self, limit=20):
        """Get top volume symbols from exchanges."""
        # Basit implementasyon
        default_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT'
        ]
        return default_symbols[:limit]

    async def process_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Process a single symbol for signals with market regime analysis.
        
        Args:
            symbol: Trading symbol to process
            
        Returns:
            Signal dictionary if found, None otherwise
        """
        try:
            self.processed_symbols_count += 1
            
            # WebSocket'ten veri almayı dene
            df_30m = None
            df_1h = None
            df_4h = None
            
            if self.websocket_manager:
                try:
                    # get_latest_data metodunu kullan
                    data_30m = self.websocket_manager.get_latest_data(symbol, '30m')
                    data_1h = self.websocket_manager.get_latest_data(symbol, '1h')
                    data_4h = self.websocket_manager.get_latest_data(symbol, '4h')
                    
                    # OHLCV verilerini DataFrame'e dönüştür
                    if data_30m and 'ohlcv' in data_30m:
                        df_30m = self._ohlcv_to_dataframe(data_30m['ohlcv'])
                    if data_1h and 'ohlcv' in data_1h:
                        df_1h = self._ohlcv_to_dataframe(data_1h['ohlcv'])
                    if data_4h and 'ohlcv' in data_4h:
                        df_4h = self._ohlcv_to_dataframe(data_4h['ohlcv'])
                        
                except AttributeError:
                    logger.debug(f"WebSocketManager missing get_latest_data method")
                    pass
            
            # REST API fallback
            if df_30m is None and self.exchange_clients:
                # İlk mevcut exchange'i kullan
                for exchange_name, client in self.exchange_clients.items():
                    try:
                        df_30m = await self._fetch_ohlcv(client, symbol, '30m')
                        df_1h = await self._fetch_ohlcv(client, symbol, '1h')
                        df_4h = await self._fetch_ohlcv(client, symbol, '4h')
                        break
                    except Exception as e:
                        logger.debug(f"REST API fetch failed for {symbol} on {exchange_name}: {e}")
                        continue
            
            # Veri yoksa skip
            if df_30m is None or df_1h is None or df_4h is None:
                logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # ===== MARKET REGIME ANALYSIS =====
            metadata = {}
            
            if self.market_regime_analyzer:
                try:
                    # Regime analizi yap
                    regime = self.market_regime_analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
                    
                    # Her 10 sembolde bir recommendations logla
                    if self.processed_symbols_count % 10 == 0:
                        recommendations = self.market_regime_analyzer.get_regime_recommendations(df_30m, df_1h, df_4h)
                        if recommendations:
                            logger.info(f"\n📊 MARKET REGIME for {symbol}:")
                            for rec in recommendations[:3]:
                                logger.info(f"  • {rec}")
                    
                    # Strategy uygunluğunu kontrol et
                    ob_favorable, ob_reason = self.market_regime_analyzer.is_favorable_for_strategy(
                        'oversold_bounce', df_30m, df_1h, df_4h
                    )
                    str_favorable, str_reason = self.market_regime_analyzer.is_favorable_for_strategy(
                        'short_the_rip', df_30m, df_1h, df_4h
                    )
                    
                    # Uygun olmayan durumları logla ama atlamak yerine metadata'ya ekle
                    if not ob_favorable and not str_favorable:
                        logger.debug(f"{symbol}: Regime not ideal - OB: {ob_reason}, STR: {str_reason}")
                    
                    # Metadata'ya ekle
                    metadata = {
                        'regime': regime,
                        'ob_favorable': ob_favorable,
                        'str_favorable': str_favorable,
                        'ob_reason': ob_reason,
                        'str_reason': str_reason
                    }
                    
                except Exception as e:
                    logger.debug(f"Regime analysis failed for {symbol}: {e}")
            
            # ===== STRATEGY SIGNALS =====
            signal = None
            
            # Execute registered strategies
            count = len(self.strategies)
            logger.info(f"🎯 Registered strategies count: {count}")
            
            if count:
                logger.info(f"🔍 Executing {count} strategies for {symbol}")
                
                for strategy_name, strategy_instance in self.strategies.items():
                    logger.info(f"  → Calling {strategy_name}...")
                    try:
                        # Call strategy's signal method
                        strategy_signal = None
                        
                        # Get cached capabilities
                        capabilities = self.strategy_capabilities.get(strategy_name, {})
                        
                        # Check if strategy has signal method
                        if hasattr(strategy_instance, 'signal'):
                            # Use cached regime_data support check
                            if capabilities.get('supports_regime_data', False):
                                # Adaptive strategies take regime_data parameter
                                strategy_signal = strategy_instance.signal(df_30m, df_1h, regime_data=metadata.get('regime'), symbol=symbol)
                            else:
                                # Standard strategies
                                strategy_signal = strategy_instance.signal(df_30m, df_1h)
                        elif hasattr(strategy_instance, 'generate_signal'):
                            # Mock or test strategies - use cached async check
                            # Use runtime check to verify if generate_signal is a coroutine function
                            if inspect.iscoroutinefunction(strategy_instance.generate_signal):
                                strategy_signal = await strategy_instance.generate_signal()
                            else:
                                strategy_signal = strategy_instance.generate_signal()
                        
                        if strategy_signal:
                            strategy_signal['strategy'] = strategy_name
                            logger.info(f"📊 Signal from {strategy_name} for {symbol}: {strategy_signal}")
                            signal = strategy_signal
                            break  # Use first signal found
                            
                    except Exception as e:
                        logger.error(f"❌ {strategy_name} error for {symbol}: {e}", exc_info=True)
            else:
                # Fallback: Use default strategies if none registered
                signals_config = self.config.get('signals', {})
                
                # Check AdaptiveOversoldBounce
                if signals_config.get('oversold_bounce', {}).get('enable', True):
                    # Sadece regime uygunsa veya ignore_regime true ise
                    ignore_regime = signals_config.get('oversold_bounce', {}).get('ignore_regime', False)
                    
                    if metadata.get('ob_favorable', True) or ignore_regime:
                        try:
                            ob_config = signals_config.get('oversold_bounce', {})
                            ob = AdaptiveOversoldBounce(ob_config, self.market_regime_analyzer)
                            
                            # Adaptive strateji farklı parametre alıyor
                            signal = ob.signal(df_30m, df_1h, regime_data=metadata.get('regime'), symbol=symbol)
                            
                            if signal:
                                signal['strategy'] = 'adaptive_ob'
                                logger.info(f"📈 Adaptive OB signal for {symbol}: {signal}")
                                
                        except Exception as e:
                            logger.debug(f"AdaptiveOB error for {symbol}: {e}")
                
                # Check AdaptiveShortTheRip (sadece signal yoksa)
                if not signal and signals_config.get('short_the_rip', {}).get('enable', True):
                    ignore_regime = signals_config.get('short_the_rip', {}).get('ignore_regime', False)
                    
                    if metadata.get('str_favorable', True) or ignore_regime:
                        try:
                            str_config = signals_config.get('short_the_rip', {})
                            strp = AdaptiveShortTheRip(str_config, self.market_regime_analyzer)
                            
                            signal = strp.signal(df_30m, df_1h, regime_data=metadata.get('regime'), symbol=symbol)
                            
                            if signal:
                                signal['strategy'] = 'adaptive_str'
                                logger.info(f"📉 Adaptive STR signal for {symbol}: {signal}")
                                
                        except Exception as e:
                            logger.debug(f"AdaptiveSTR error for {symbol}: {e}")
            
            # Signal'e metadata ekle
            if signal:
                signal['metadata'] = metadata
                signal['symbol'] = symbol
                signal['timestamp'] = datetime.now(timezone.utc)

            # Monitor adaptive signals
            if signal and signal.get('is_adaptive'):
                from core.adaptive_monitor import adaptive_monitor
                adaptive_monitor.record_adaptive_signal(symbol, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None 

    async def _process_trading_loop(self):
        """Main trading loop processing with timeout protection."""
        for symbol in self.active_symbols:
            try:
                # Add timeout protection and capture the returned signal
                signal = await asyncio.wait_for(
                    self.process_symbol(symbol),
                    timeout=30.0
                )
    
                # If a signal was generated, submit it for execution
                if signal:
                    logger.info(f"Submitting signal for {symbol} to execution engine.")
                    submission_result = await self.submit_signal(signal)
                    if not submission_result.get('success'):
                        logger.warning(f"Failed to submit signal for {symbol}: {submission_result.get('reason')}")
    
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout processing {symbol} - skipping")
                continue
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

    async def _initialize_production_system(self) -> bool:
        """Legacy wrapper - just calls the public method."""
        # Public metodu çağır ve sonucunu döndür
        result = await self.initialize_production_system()
        return result.get('success', False)

    async def initialize_production_system(self, 
                                          exchange_clients: Optional[Dict] = None,
                                          portfolio_config: Optional[Dict] = None,
                                          mode: str = 'paper',
                                          trading_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize production system with all components.
        """
        logger.info("Initializing Production Trading System...")
        
        try:
            # Store provided configuration
            if exchange_clients:
                self.exchange_clients = exchange_clients
            
            # Skip WebSocket init if already provided externally
            if not self.websocket_manager and not hasattr(self, 'skip_ws_init'):
                self.websocket_manager = WebSocketManager(exchanges=self.exchange_clients)
            
            # Initialize components with graceful fallbacks
            try:
                self.performance_monitor = PerformanceMonitor()
            except:
                self.performance_monitor = RealTimePerformanceMonitor()
            
            # ÇÖZÜM: portfolio_config'i doğrudan RiskManager'a geç!
            portfolio_config = portfolio_config or {}
            
            # Config'den risk parametrelerini al ve birleştir
            config = self.config
            risk_config = config.get('risk', {})
            
            # RiskManager için portfolio_config hazırla
            risk_manager_config = {
                'equity_usd': float(
                    portfolio_config.get('equity_usd') or 
                    risk_config.get('equity_usd', 100)  # Config'den veya default 100
                ),
                'per_trade_risk_pct': float(risk_config.get('per_trade_risk_pct', 0.01)),
                'daily_loss_limit_pct': float(risk_config.get('daily_loss_limit_pct', 0.02)),
                'risk_usd_cap': float(risk_config.get('risk_usd_cap', 5)),
                'max_notional_per_trade': float(risk_config.get('max_notional_per_trade', 20)),
                'max_portfolio_risk': 0.02,  # RiskManager defaults
                'max_position_size': 0.10,
                'max_drawdown': 0.15,
                'max_correlation': 0.70
            }
            
            # Risk Manager - DOĞRU PARAMETRELERLE
            logger.info(f"Initializing RiskManager with equity_usd: ${risk_manager_config['equity_usd']}")
            self.risk_manager = RiskManager(
                portfolio_config=risk_manager_config,  # Artık doğru format!
                websocket_manager=self.websocket_manager,
                performance_monitor=self.performance_monitor
            )
            
            # Portfolio Manager
            self.portfolio_manager = PortfolioManager(
                risk_manager=self.risk_manager,
                performance_monitor=self.performance_monitor,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients
            )
            self.portfolio_manager.cfg = self.config
            
            # Other components...
            self.strategy_coordinator = StrategyCoordinator(self.portfolio_manager, self.risk_manager)
            self.circuit_breaker = CircuitBreakerSystem(self.portfolio_manager, self.risk_manager)
            
            # LiveTradingEngine
            self.trading_engine = LiveTradingEngine(
                mode=mode,
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients,
                strategy_coordinator=self.strategy_coordinator
            )
            
            # Set active symbols with multi-tier fallback
            if trading_symbols:
                self.active_symbols = trading_symbols
                logger.info(f"✓ Active symbols set from parameter: {len(trading_symbols)} symbols")
            else:
                # Fallback 1: Try loading from config
                config_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
                if config_symbols and isinstance(config_symbols, list):
                    self.active_symbols = config_symbols
                    logger.info(f"✓ Active symbols loaded from config: {len(config_symbols)} symbols")
                # Fallback 2: Try getting from engine (if already created)
                elif self.trading_engine:
                    try:
                        self.active_symbols = self.trading_engine._get_scan_symbols()
                        logger.info(f"✓ Active symbols loaded from engine: {len(self.active_symbols)} symbols")
                    except Exception as e:
                        logger.error(f"❌ Failed to get symbols from engine: {e}")
                        self.active_symbols = []
                else:
                    logger.warning("⚠️ No active symbols configured! Trading loop will be idle.")
                    self.active_symbols = []
            
            # Log final result
            if self.active_symbols:
                logger.info(f"📊 Final active symbols ({len(self.active_symbols)}): {self.active_symbols[:3]}...")
            else:
                logger.error("❌ CRITICAL: No active symbols! Bot cannot trade!")
            
            self.is_initialized = True
            
            # SUCCESS LOG
            logger.info(f"✅ System initialized with portfolio value: ${self.risk_manager.portfolio_value:.2f}")
            
            components = ['websocket_manager', 'performance_monitor', 'risk_manager', 
                         'portfolio_manager', 'strategy_coordinator', 'circuit_breaker', 
                         'trading_engine']
            
            return {'success': True, 'components': components}
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return {'success': False, 'reason': str(e)}
            
    async def run_production_loop(self, mode: str = 'paper', duration: Optional[float] = None, 
                                  continuous: bool = False):
        """Main production trading loop."""
        try:
            if not self.is_initialized:
                raise RuntimeError("Production system not initialized. Call initialize_production_system() first.")
            
            logger.info("="*70)
            logger.info("STARTING PRODUCTION TRADING LOOP")
            logger.info("="*70)
            
            # Start the live trading engine
            start_result = await self.trading_engine.start_live_trading(mode=mode)
            
            if not start_result['success']:
                raise RuntimeError(f"Failed to start trading engine: {start_result.get('reason')}")
            
            self.is_running = True
            
            # Start queue monitoring task
            self._monitoring_task = asyncio.create_task(self._monitor_signal_queues())
            
            logger.info("\n🚀 Production trading loop active")
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Duration: {'Indefinite' if duration is None else f'{duration}s'}")
            logger.info(f"   Continuous Mode: {'ENABLED (Never stops, auto-recovers)' if continuous else 'DISABLED'}")
            logger.info(f"   Active Symbols: {len(self.active_symbols)}")
            
            # Main loop
            start_time = datetime.now(timezone.utc)
            last_recommendation_time = start_time
            recommendation_interval = 300  # Her 5 dakikada bir recommendations
            loop_iteration = 0
            
            # ✅ EKLE: Trading loop başlangıç logu
            logger.info("")
            logger.info("="*70)
            logger.info("🔄 STARTING TRADING LOOP ITERATIONS")
            logger.info("="*70)
            logger.info(f"   Loop interval: {self.loop_interval}s")
            logger.info(f"   Symbols to process: {len(self.active_symbols)}")
            if duration:
                logger.info(f"   Will run for: {duration}s")
            else:
                logger.info(f"   Will run: Indefinitely")
            logger.info("="*70)
            logger.info("")
            
            while self.is_running:
                try:
                    loop_iteration += 1
                    # ✅ DEĞİŞTİR: logger.debug → logger.info
                    logger.info(f"🔁 [ITERATION {loop_iteration}] Processing {len(self.active_symbols)} symbols...")
                    
                    # Check emergency conditions
                    if self.emergency_stop_triggered:
                        logger.critical("Emergency stop triggered - shutting down")
                        break
                    
                    # Check circuit breaker
                    breaker_status = await self.circuit_breaker.check_circuit_breaker()
                    if breaker_status.get('tripped'):
                        severity = breaker_status.get('severity', 'high')
                        
                        if continuous and severity != 'critical':
                            logger.warning(f"Circuit breaker tripped ({severity}): {breaker_status.get('message')}")
                            logger.warning("CONTINUOUS MODE: Bypassing non-critical breaker, continuing...")
                            await asyncio.sleep(10)
                            continue
                        else:
                            logger.critical(f"Circuit breaker tripped ({severity}): {breaker_status.get('message')}")
                            await self.handle_emergency_shutdown('circuit_breaker_tripped')
                            break
                    
                    # Check duration
                    if duration and not continuous:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        if elapsed >= duration:
                            logger.info(f"Duration {duration}s reached - stopping")
                            break
                    
                    # Process trading loop with WebSocket data
                    await self._process_trading_loop()
                    
                    # Market regime recommendations (periyodik)
                    current_time = datetime.now(timezone.utc)
                    time_since_last = (current_time - last_recommendation_time).total_seconds()
                    
                    if time_since_last >= recommendation_interval:
                        if self.market_regime_analyzer and self.active_symbols:
                            try:
                                # İlk sembol için recommendations al
                                test_symbol = self.active_symbols[0]
                                
                                if self.websocket_manager:
                                    df_30m = self.websocket_manager.get_latest_data(test_symbol, '30m')
                                    df_1h = self.websocket_manager.get_latest_data(test_symbol, '1h')
                                    df_4h = self.websocket_manager.get_latest_data(test_symbol, '4h')
                                    
                                    if df_30m is not None and df_1h is not None and df_4h is not None:
                                        recommendations = self.market_regime_analyzer.get_regime_recommendations(
                                            df_30m, df_1h, df_4h
                                        )
                                        
                                        if recommendations:
                                            logger.info("\n" + "="*50)
                                            logger.info("📊 MARKET RECOMMENDATIONS UPDATE:")
                                            for rec in recommendations[:5]:
                                                logger.info(f"  • {rec}")
                                            logger.info("="*50 + "\n")
                                        
                                        last_recommendation_time = current_time
                                
                            except Exception as e:
                                logger.debug(f"Could not get recommendations: {e}")
                    
                    # Sleep between iterations
                    logger.debug(f"🔁 Trading loop iteration {loop_iteration} completed, sleeping {self.loop_interval}s")
                    await asyncio.sleep(self.loop_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received - stopping gracefully")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in production loop: {e}", exc_info=True)
                    
                    # Auto-recovery in continuous mode
                    if continuous:
                        logger.warning("CONTINUOUS MODE: Auto-recovering from error...")
                        
                        try:
                            if self.trading_engine and not self.trading_engine.is_running:
                                logger.info("Attempting to restart trading engine...")
                                restart_result = await self.trading_engine.start_live_trading(mode=mode)
                                if restart_result['success']:
                                    logger.info("✓ Trading engine restarted successfully")
                                else:
                                    logger.error(f"Failed to restart trading engine: {restart_result.get('reason')}")
                        except Exception as restart_error:
                            logger.error(f"Error during auto-recovery: {restart_error}")
                        
                        await asyncio.sleep(5)
                        continue
                    else:
                        # Original behavior for non-continuous mode
                        if self.config.get('emergency', {}).get('enable_circuit_breaker', True):
                            await self.handle_emergency_shutdown('system_error')
                            break
                        else:
                            await asyncio.sleep(5)
                            continue
            
            # Shutdown
            logger.info("\nShutting down production trading loop...")
            await self.trading_engine.stop_live_trading()
            self.is_running = False
            
            logger.info("✓ Production trading loop stopped")
            
        except Exception as e:
            logger.error(f"Critical error in production loop: {e}", exc_info=True)
            self.is_running = False
            await self.handle_emergency_shutdown('critical_error')
    
    async def handle_emergency_shutdown(self, reason: str):
        """
        Emergency shutdown protocol.
        
        Args:
            reason: Reason for emergency shutdown
        """
        try:
            logger.critical("="*70)
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
            logger.critical(f"Reason: {reason}")
            logger.critical("="*70)
            
            self.emergency_stop_triggered = True
            
            # Step 1: Stop new signals
            logger.critical("Step 1: Stopping new signal processing...")
            self.is_running = False
            
            # Step 2: Cancel pending orders
            logger.critical("Step 2: Cancelling pending orders...")
            # Would iterate through active orders and cancel them
            
            # Step 3: Close positions (if configured)
            close_method = self.config.get('emergency', {}).get('emergency_close_method', 'market')
            logger.critical(f"Step 3: Closing positions using {close_method} method...")
            
            # Get active positions
            if self.trading_engine:
                active_positions = list(self.trading_engine.active_positions.keys())
                
                if active_positions:
                    logger.critical(f"Closing {len(active_positions)} active positions...")
                    
                    # Execute emergency protocol via circuit breaker
                    if self.circuit_breaker:
                        await self.circuit_breaker.execute_emergency_protocol('close_all', active_positions)
                else:
                    logger.critical("No active positions to close")
            
            # Step 4: Stop trading engine
            if self.trading_engine:
                logger.critical("Step 4: Stopping trading engine...")
                await self.trading_engine.stop_live_trading()
            
            # Step 5: Close WebSocket connections
            if self.websocket_manager:
                logger.critical("Step 5: Closing WebSocket connections...")
                await self.websocket_manager.close()
            
            # Step 6: Save state
            logger.critical("Step 6: Saving system state...")
            state = self.get_system_state()
            # Could save state to file here
            
            # Step 7: Generate emergency report
            logger.critical("Step 7: Generating emergency report...")
            report = self._generate_emergency_report(reason)
            
            logger.critical("\n" + "="*70)
            logger.critical("EMERGENCY SHUTDOWN COMPLETE")
            logger.critical("="*70)
            
            # Could send alert notification here
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}", exc_info=True)
    
    def _generate_emergency_report(self, reason: str) -> Dict[str, Any]:
        """Generate emergency shutdown report."""
        report = {
            'timestamp': datetime.now(timezone.utc),
            'reason': reason,
            'system_state': self.get_system_state()
        }
        
        if self.trading_engine:
            report['engine_status'] = self.trading_engine.get_engine_status()
        
        if self.portfolio_manager:
            report['portfolio_state'] = self.portfolio_manager.portfolio_state
        
        return report
    
    async def submit_signal(self, signal: Dict) -> Dict[str, Any]:
        """
        Submit trading signal to the system.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Submission result
        """
        try:
            if not self.is_running:
                return {
                    'success': False,
                    'reason': 'Trading system not running'
                }
            
            # Process through strategy coordinator
            result = await self.strategy_coordinator.process_strategy_signal(
                strategy_name=signal.get('strategy', 'unknown'),
                signal=signal
            )
            
            if result['status'] == 'accepted':
                # Get signal_id from coordinator result
                signal_id = result.get('signal_id')
                
                # [STAGE 1: GENERATED] Mark signal as generated
                self._track_signal_lifecycle(signal_id, 'generated', {'symbol': signal.get('symbol'), 'strategy': signal.get('strategy')})
                logger.info(f"[STAGE:GENERATED] Signal {signal_id} for {signal.get('symbol')}")
                
                # [STAGE 2: VALIDATED] Signal passed validation
                enriched_signal = result['enriched_signal']
                self._track_signal_lifecycle(signal_id, 'validated', {'reason': 'passed_all_checks'})
                logger.info(f"[STAGE:VALIDATED] Signal {signal_id} validated")
                
                # [STAGE 3: QUEUED] Signal added to StrategyCoordinator queue (already done in coordinator)
                self._track_signal_lifecycle(signal_id, 'queued', {'queue': 'strategy_coordinator'})
                logger.info(f"[STAGE:QUEUED] Signal {signal_id} in StrategyCoordinator queue")
                
                # Attach signal identifier so execution layer can report completion
                if signal_id and isinstance(enriched_signal, dict):
                    enriched_signal['signal_id'] = signal_id

                # [STAGE 4: FORWARDED] Forward enriched signal to LiveTradingEngine
                await self.trading_engine.signal_queue.put(enriched_signal)
                self._track_signal_lifecycle(signal_id, 'forwarded', {'queue': 'live_trading_engine'})
                logger.info(f"[STAGE:FORWARDED] Signal {signal_id} forwarded to LiveTradingEngine queue")
                
                return {'success': True, 'signal_id': signal_id}
            else:
                logger.warning(f"Signal rejected: {result.get('reason')}")
                # For rejected signals, generate a temporary ID for tracking
                signal_id = f"{signal.get('strategy', 'unknown')}_{signal.get('symbol', 'UNKNOWN')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                # [STAGE 1: GENERATED] Mark signal as generated for rejected signals
                self._track_signal_lifecycle(signal_id, 'generated', {'symbol': signal.get('symbol'), 'strategy': signal.get('strategy')})
                # [STAGE 2: REJECTED] Mark signal as rejected
                self._track_signal_lifecycle(signal_id, 'rejected', {'reason': result.get('reason')})
                return {'success': False, 'reason': result.get('reason')}
                
        except Exception as e:
            logger.error(f"Error submitting signal: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _track_signal_lifecycle(self, signal_id: str, stage: str, details: Dict = None):
        """
        Track signal through its lifecycle stages.
        
        Stages: generated -> validated -> queued -> forwarded -> executed
        
        Args:
            signal_id: Unique signal identifier
            stage: Current stage
            details: Additional details about this stage
        """
        if signal_id not in self.signal_lifecycle:
            self.signal_lifecycle[signal_id] = {
                'stages': [],
                'created_at': datetime.now(timezone.utc)
            }
        
        self.signal_lifecycle[signal_id]['stages'].append({
            'stage': stage,
            'timestamp': datetime.now(timezone.utc),
            'details': details or {}
        })
        
        # Keep only last 100 signals to avoid memory issues
        if len(self.signal_lifecycle) > 100:
            oldest_key = min(self.signal_lifecycle.keys(), 
                           key=lambda k: self.signal_lifecycle[k]['created_at'])
            del self.signal_lifecycle[oldest_key]
    
    def register_strategy(self, strategy_name: str, strategy_instance: Any, 
                         initial_allocation: float = 0.25) -> Dict[str, Any]:
        """
        Register a trading strategy with the system.
        
        Args:
            strategy_name: Unique strategy identifier
            strategy_instance: Strategy instance
            initial_allocation: Initial capital allocation
            
        Returns:
            Registration result
        """
        try:
            if not self.is_initialized:
                return {
                    'success': False,
                    'reason': 'System not initialized'
                }
            
            # Store strategy reference in coordinator
            self.strategies[strategy_name] = strategy_instance
            
            # Cache strategy capabilities to avoid repeated inspection
            capabilities = {
                'supports_regime_data': False,
                'is_async': False
            }
            
            # Check if strategy has signal method and supports regime_data
            if hasattr(strategy_instance, 'signal'):
                sig = inspect.signature(strategy_instance.signal)
                capabilities['supports_regime_data'] = 'regime_data' in sig.parameters
            
            # Check if strategy has async generate_signal method
            if hasattr(strategy_instance, 'generate_signal'):
                capabilities['is_async'] = inspect.iscoroutinefunction(strategy_instance.generate_signal)
            
            self.strategy_capabilities[strategy_name] = capabilities
            
            result = self.portfolio_manager.register_strategy(
                strategy_name=strategy_name,
                strategy_instance=strategy_instance,
                initial_allocation=initial_allocation
            )
            
            logger.info(f"Strategy registered: {strategy_name} with {initial_allocation*100}% allocation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error registering strategy: {e}")
            return {'success': False, 'reason': str(e)}
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        state = {
            'timestamp': datetime.now(timezone.utc),
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'emergency_stop': self.emergency_stop_triggered,
            'active_symbols': self.active_symbols,
            'processed_symbols': self.processed_symbols_count
        }
        
        if self.trading_engine:
            state['trading_engine'] = self.trading_engine.get_engine_status()
        
        if self.portfolio_manager:
            state['portfolio'] = self.portfolio_manager.portfolio_state
        
        if self.risk_manager:
            state['risk_limits'] = self.risk_manager.risk_limits
        
        if self.market_regime_analyzer:
            state['market_regime'] = self.market_regime_analyzer.get_current_regime()
        
        return state
    
    def _print_position_dashboard(self):
        """
        Display real-time position dashboard with P&L.
        Phase 3.4 - Issue #105: Position Dashboard
        """
        try:
            if not self.portfolio_manager or not hasattr(self.portfolio_manager, 'risk_manager'):
                return
            
            risk_manager = self.portfolio_manager.risk_manager
            active_positions = risk_manager.active_positions if hasattr(risk_manager, 'active_positions') else {}
            
            if not active_positions:
                logger.info("\n📊 POSITION DASHBOARD: No open positions")
                return
            
            logger.info("\n" + "="*70)
            logger.info("📊 POSITION DASHBOARD")
            logger.info("="*70)
            
            total_unrealized_pnl = 0.0
            for position_id, position in active_positions.items():
                symbol = position.get('symbol', 'UNKNOWN')
                side = position.get('side', 'unknown')
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)
                amount = position.get('amount', 0)
                
                # Calculate unrealized P&L
                unrealized_pnl = calculate_unrealized_pnl(side, entry_price, current_price, amount)
                
                pnl_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
                total_unrealized_pnl += unrealized_pnl
                
                # Format output
                pnl_symbol = "✅" if unrealized_pnl >= 0 else "❌"
                logger.info(f"{pnl_symbol} {symbol} {side.upper()}")
                logger.info(f"   Entry: ${entry_price:.4f} | Current: ${current_price:.4f}")
                logger.info(f"   Amount: {amount:.4f} | P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")
            
            logger.info("-"*70)
            total_symbol = "✅" if total_unrealized_pnl >= 0 else "❌"
            logger.info(f"{total_symbol} TOTAL UNREALIZED P&L: ${total_unrealized_pnl:.2f}")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Error displaying position dashboard: {e}")
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List) -> pd.DataFrame:
        """Convert OHLCV list data to DataFrame."""
        import pandas as pd
        
        if not ohlcv_data:
            return None
        
        # OHLCV formatı: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Indicators ekle (eğer yoksa)
        from core.indicators import add_indicators
        if 'rsi' not in df.columns:
            df = add_indicators(df, self.config.get('indicators', {}))
        
        return df
    
    async def _fetch_ohlcv(self, client, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Helper method to fetch OHLCV data via REST API.
        
        This method runs the blocking client.ohlcv() call in a thread pool to prevent
        blocking the async event loop, which was causing the bot to freeze.
        """
        try:
            # Run blocking I/O in thread pool to prevent event loop blocking
            rows = await asyncio.to_thread(client.ohlcv, symbol, timeframe, limit=200)
            return self._ohlcv_to_dataframe(rows)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None
    
    async def _monitor_signal_queues(self):
        """
        Monitor signal queues and log their sizes periodically.
        
        Monitors:
        - StrategyCoordinator signal queue
        - LiveTradingEngine signal queue
        """
        logger.info("Queue monitoring task started")
        
        try:
            while self.is_running:
                try:
                    # Get queue sizes
                    coordinator_queue_size = 0
                    engine_queue_size = 0
                    
                    if self.strategy_coordinator:
                        coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                    
                    if self.trading_engine:
                        engine_queue_size = self.trading_engine.signal_queue.qsize()
                    
                    # Log queue status
                    logger.info(f"📊 [QUEUE-MONITOR] StrategyCoordinator: {coordinator_queue_size} signals | LiveTradingEngine: {engine_queue_size} signals")
                    
                    # Log lifecycle summary
                    if self.signal_lifecycle:
                        total_signals = len(self.signal_lifecycle)
                        stage_counts = {}
                        for signal_data in self.signal_lifecycle.values():
                            if signal_data['stages']:
                                last_stage = signal_data['stages'][-1]['stage']
                                stage_counts[last_stage] = stage_counts.get(last_stage, 0) + 1
                        
                        logger.info(f"📊 [LIFECYCLE] Total tracked: {total_signals} | Stages: {stage_counts}")
                    
                    # Wait 30 seconds before next check
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    logger.info("Queue monitoring task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in queue monitoring: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Fatal error in queue monitoring: {e}", exc_info=True)
    
    async def stop_system(self):
        """Stop the production system gracefully."""
        logger.info("Stopping production system...")
        self.is_running = False
        
        # Cancel monitoring task
        if hasattr(self, '_monitoring_task') and self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.trading_engine:
            await self.trading_engine.stop_live_trading()
        
        if self.websocket_manager:
            await self.websocket_manager.close()
        
        logger.info("Production system stopped")
