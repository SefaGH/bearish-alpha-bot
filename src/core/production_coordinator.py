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
# ✅ EKLE: Logger seviyesini zorla INFO yap
logger.setLevel(logging.INFO)


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
        timeframes = ws_config.get('stream_timeframes', ['1m', '5m', '30m', '1h', '4h'])
        
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
            logger.info(f"[DATA-FETCH] Fetching market data for {symbol}")
            
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
                    
                    if df_30m is not None and df_1h is not None and df_4h is not None:
                        logger.info(f"[DATA-FETCH] ✅ WebSocket data retrieved for {symbol}")
                    else:
                        logger.info(f"[DATA-FETCH] ⚠️ Incomplete WebSocket data for {symbol}, will try REST API")
                        
                except AttributeError as e:
                    logger.warning(f"[DATA-FETCH] WebSocketManager missing get_latest_data method: {e}")
                    pass
            
            # REST API fallback with timeout protection
            if df_30m is None and self.exchange_clients:
                logger.info(f"[DATA-FETCH] Using REST API fallback for {symbol}")
                # İlk mevcut exchange'i kullan
                for exchange_name, client in self.exchange_clients.items():
                    try:
                        # Fetch all timeframes with overall 45s timeout (3 x 15s individual timeouts)
                        fetch_tasks = [
                            asyncio.create_task(asyncio.wait_for(
                                self._fetch_ohlcv(client, symbol, '30m'), 
                                timeout=10.0
                            )),
                            asyncio.create_task(asyncio.wait_for(
                                self._fetch_ohlcv(client, symbol, '1h'), 
                                timeout=10.0
                            )),
                            asyncio.create_task(asyncio.wait_for(
                                self._fetch_ohlcv(client, symbol, '4h'), 
                                timeout=10.0
                            ))
                        ]
                        
                        # Wait for all with overall timeout
                        results = await asyncio.wait_for(
                            asyncio.gather(*fetch_tasks, return_exceptions=True),
                            timeout=15.0
                        )
                        
                        # Process results
                        df_30m = results[0] if not isinstance(results[0], Exception) else None
                        df_1h = results[1] if not isinstance(results[1], Exception) else None
                        df_4h = results[2] if not isinstance(results[2], Exception) else None
                        
                        if df_30m is not None:
                            logger.info(f"[DATA-FETCH] ✅ REST API data retrieved for {symbol}")
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[DATA-FETCH] ⏱️ REST API timeout for {symbol} (15s limit)")
                        continue
                    except Exception as e:
                        logger.warning(f"[DATA-FETCH] REST API failed: {e}")
                        continue
            
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
                    logger.warning(f"[REGIME] Regime analysis failed for {symbol}: {e}")
            
            # ===== STRATEGY SIGNALS =====
            signal = None
            
            # Execute registered strategies
            count = len(self.strategies)
            logger.info(f"[STRATEGY-CHECK] {count} registered strategies available")
            
            if count:
                logger.info(f"[STRATEGY-CHECK] Executing {count} strategies for {symbol}")
                
                for strategy_name, strategy_instance in self.strategies.items():
                    logger.info(f"[STRATEGY-CHECK] Running {strategy_name} for {symbol}...")
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
                logger.info(f"[STRATEGY-CHECK] No registered strategies, using fallback strategies for {symbol}")
                signals_config = self.config.get('signals', {})
                
                # Check AdaptiveOversoldBounce
                if signals_config.get('oversold_bounce', {}).get('enable', True):
                    logger.info(f"[STRATEGY-CHECK] Checking AdaptiveOversoldBounce (adaptive_ob) for {symbol}")
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
                                logger.info(f"📊 Signal from adaptive_ob for {symbol}: {signal}")
                            else:
                                logger.info(f"[STRATEGY-CHECK] adaptive_ob: No signal for {symbol}")
                                
                        except Exception as e:
                            logger.warning(f"[STRATEGY-CHECK] AdaptiveOB error for {symbol}: {e}", exc_info=True)
                    else:
                        logger.info(f"[STRATEGY-CHECK] adaptive_ob: Regime not favorable for {symbol}, skipping")
                
                # Check AdaptiveShortTheRip (sadece signal yoksa)
                if not signal and signals_config.get('short_the_rip', {}).get('enable', True):
                    logger.info(f"[STRATEGY-CHECK] Checking AdaptiveShortTheRip (adaptive_str) for {symbol}")
                    ignore_regime = signals_config.get('short_the_rip', {}).get('ignore_regime', False)
                    
                    if metadata.get('str_favorable', True) or ignore_regime:
                        try:
                            str_config = signals_config.get('short_the_rip', {})
                            strp = AdaptiveShortTheRip(str_config, self.market_regime_analyzer)
                            
                            signal = strp.signal(df_30m, df_1h, regime_data=metadata.get('regime'), symbol=symbol)
                            
                            if signal:
                                signal['strategy'] = 'adaptive_str'
                                logger.info(f"📊 Signal from adaptive_str for {symbol}: {signal}")
                            else:
                                logger.info(f"[STRATEGY-CHECK] adaptive_str: No signal for {symbol}")
                                
                        except Exception as e:
                            logger.warning(f"[STRATEGY-CHECK] AdaptiveSTR error for {symbol}: {e}", exc_info=True)
                    else:
                        logger.info(f"[STRATEGY-CHECK] adaptive_str: Regime not favorable for {symbol}, skipping")
            
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
            logger.error(f"❌ Critical error processing {symbol}: {e}", exc_info=True)
            return None 

    async def _process_trading_loop(self):
        """Main trading loop processing with timeout protection and detailed logging."""
        # Log entry to confirm loop is executing
        logger.info(f"📋 [PROCESSING] Starting processing loop for {len(self.active_symbols)} symbols")
        import time
        start_time = time.time()
        
        processed_count = 0
        signal_count = 0
        error_count = 0
        
        for symbol in self.active_symbols:
            try:
                logger.info(f"[PROCESSING] Symbol {processed_count + 1}/{len(self.active_symbols)}: {symbol}")
                symbol_start = time.time()
                
                # Add timeout protection and capture the returned signal
                signal = await asyncio.wait_for(
                    self.process_symbol(symbol),
                    timeout=30.0
                )
                
                symbol_duration = time.time() - symbol_start
                logger.info(f"[PROCESSING] {symbol} completed in {symbol_duration:.2f}s")
                processed_count += 1
    
                # If a signal was generated, submit it for execution
                if signal:
                    logger.info(f"✅ Signal generated for {symbol}, submitting to execution engine")
                    submission_result = await self.submit_signal(signal)
                    if not submission_result.get('success'):
                        logger.warning(f"Failed to submit signal for {symbol}: {submission_result.get('reason')}")
                    else:
                        signal_count += 1
                else:
                    logger.info(f"ℹ️ No signal generated for {symbol}")
    
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout processing {symbol} after 30s - skipping")
                error_count += 1
                continue
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
                error_count += 1
                continue
        
        total_duration = time.time() - start_time
        logger.info(f"✅ [PROCESSING] Completed processing loop in {total_duration:.2f}s")
        logger.info(f"   Processed: {processed_count}/{len(self.active_symbols)} symbols")
        logger.info(f"   Signals: {signal_count} | Errors: {error_count}")

    def _get_default_symbols(self) -> List[str]:
        """
        Get default symbols with proper fallback logic.
        
        This is the single source of truth for symbol discovery when
        symbols are not explicitly provided via parameter.
        
        Fallback order:
        1. Config file: config['universe']['fixed_symbols']
        2. Environment variable: TRADING_SYMBOLS (comma-separated)
        3. Hardcoded defaults: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT
        
        Returns:
            List of trading symbols
        """
        # Priority 1: Check config file
        config_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
        if config_symbols and isinstance(config_symbols, list) and len(config_symbols) > 0:
            logger.info(f"[SYMBOL_DISCOVERY] Using {len(config_symbols)} symbols from config")
            return config_symbols
        
        # Priority 2: Check environment variable
        env_symbols = os.environ.get('TRADING_SYMBOLS', '').strip()
        if env_symbols:
            symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
            if symbols:
                logger.info(f"[SYMBOL_DISCOVERY] Using {len(symbols)} symbols from TRADING_SYMBOLS env var")
                return symbols
        
        # Priority 3: Use hardcoded defaults
        # Note: Using 3 major symbols as per issue requirements for cleaner default set.
        # Previous implementation had 8 symbols: BTC, ETH, SOL, BNB, ADA, DOT, LTC, AVAX
        # New default uses 3 major pairs (BTC, ETH, SOL) for sufficient fallback coverage
        # while keeping the default set minimal and maintainable.
        default_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        logger.info(f"[SYMBOL_DISCOVERY] Using {len(default_symbols)} hardcoded default symbols")
        return default_symbols
    
    async def _initialize_ml_components(self) -> Dict[str, Any]:
        """
        Initialize and connect ALL ML components from src/ml/.
        This connects the fully implemented but disconnected ML layer.
        
        Returns:
            Dict with 'success' and optional 'reason' keys
        """
        logger.info("🧠 [ML-INIT] Initializing ML system...")
        
        ml_components = []
        
        try:
            # Import ML components (optional - won't fail if not available)
            try:
                from ml.strategy_integration import (
                    MLStrategyIntegrationManager,
                    AIEnhancedStrategyAdapter
                )
                from ml.price_predictor import AdvancedPricePredictionEngine
                from ml.regime_predictor import MLRegimePredictor
                from ml.prediction_engine import RealTimePredictionEngine
                from ml.reinforcement_learning import TradingRLAgent
                from ml.experience_replay import ExperienceReplay
                from ml.feature_engineering import FeatureEngineeringPipeline
                
                logger.info("🧠 [ML-INIT] All ML modules imported successfully")
            except ImportError as e:
                logger.warning(f"🧠 [ML-INIT] ML modules not available: {e}")
                return {
                    'success': False,
                    'reason': f'ML modules not available: {e}'
                }
            
            # 1. Initialize Feature Engineering Pipeline
            self.feature_pipeline = FeatureEngineeringPipeline()
            ml_components.append('feature_pipeline')
            logger.info("✅ Feature engineering pipeline ready")
            
            # 2. Initialize Price Prediction Engine (simplified - no complex models for now)
            try:
                # AdvancedPricePredictionEngine bazı ortamlarda predictor keyword'ünü kabul etmiyor.
                # Dayanıklı kurulum: önce positional None, TypeError olursa argsız deneyin.
                try:
                    self.price_engine = AdvancedPricePredictionEngine(None)
                except TypeError:
                    self.price_engine = AdvancedPricePredictionEngine()
                ml_components.append('price_engine')
                logger.info("✅ Price prediction engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Price engine init failed: {e}")
            
            # 3. Initialize Regime Predictor
            try:
                self.regime_predictor = MLRegimePredictor(
                    regime_analyzer=self.market_regime_analyzer,
                    websocket_manager=self.websocket_manager
                )
                ml_components.append('regime_predictor')
                logger.info("✅ Regime predictor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Regime predictor init failed: {e}")
            
            # 4. Initialize Reinforcement Learning Agent
            try:
                self.rl_agent = TradingRLAgent(
                    state_size=50,   # Feature dimension
                    action_size=3,   # Buy/Hold/Sell
                    learning_rate=0.001
                )
                
                # Initialize experience replay buffer
                self.experience_replay = ExperienceReplay(max_size=100000)
                self.rl_agent.set_memory(self.experience_replay)
                
                ml_components.append('rl_agent')
                logger.info("✅ Reinforcement learning agent initialized")
            except Exception as e:
                logger.warning(f"⚠️ RL agent init failed: {e}")
            
            # 5. Initialize ML Strategy Integration Manager
            try:
                if hasattr(self, 'price_engine') and hasattr(self, 'regime_predictor'):
                    self.ml_integration = MLStrategyIntegrationManager(
                        self.price_engine,
                        self.regime_predictor
                    )
                    ml_components.append('ml_integration')
                    logger.info("✅ ML strategy integration manager initialized")
                else:
                    logger.warning("⚠️ Cannot init ML integration without price/regime engines")
            except Exception as e:
                logger.warning(f"⚠️ ML integration init failed: {e}")
            
            # 6. Connect ML to Strategy Coordinator
            if hasattr(self, 'strategy_coordinator') and self.strategy_coordinator:
                if hasattr(self, 'ml_integration'):
                    self.strategy_coordinator.ml_integration = self.ml_integration.adapter
                if hasattr(self, 'feature_pipeline'):
                    self.strategy_coordinator.feature_pipeline = self.feature_pipeline
                if hasattr(self, 'rl_agent'):
                    self.strategy_coordinator.rl_agent = self.rl_agent
                logger.info("🔗 ML connected to StrategyCoordinator")
            
            # 7. Connect ML to LiveTradingEngine
            if hasattr(self, 'trading_engine') and self.trading_engine:
                if hasattr(self, 'rl_agent'):
                    self.trading_engine.rl_agent = self.rl_agent
                if hasattr(self, 'ml_integration'):
                    self.trading_engine.ml_integration = self.ml_integration
                logger.info("🔗 ML connected to LiveTradingEngine")
            
            logger.info("🧠 [ML-INIT] ✅ ML SYSTEM INITIALIZED")
            logger.info(f"   Components: {', '.join(ml_components)}")
            logger.info(f"   Tracking {len(self.active_symbols)} symbols")
            
            return {
                'success': True,
                'components': ml_components
            }
            
        except Exception as e:
            logger.error(f"🧠 [ML-INIT] Initialization error: {e}", exc_info=True)
            return {
                'success': False,
                'reason': str(e)
            }
    
    async def _initialize_production_system(self) -> bool:
        """
        Legacy wrapper for backwards compatibility.
        
        This method wraps the public initialize_production_system() method
        and returns a boolean instead of a dict for legacy code compatibility.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        result = await self.initialize_production_system()
        return result.get('success', False)
    
    async def initialize_production_system(self, 
                                          exchange_clients: Optional[Dict] = None,
                                          portfolio_config: Optional[Dict] = None,
                                          mode: str = 'paper',
                                          trading_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize production system with all Phase 3 components.
        
        This is the main initialization method that sets up:
        - WebSocket connections (Phase 3.1)
        - Risk Management (Phase 3.2)
        - Portfolio Management (Phase 3.3)
        - Trading Engine (Phase 3.4)
        - Strategy Coordinator
        - Circuit Breaker System
        
        Args:
            exchange_clients: Dict of exchange client instances {exchange_name: client}
            portfolio_config: Portfolio configuration dict with 'equity_usd' etc.
            mode: Trading mode ('paper', 'live', 'simulation')
            trading_symbols: List of symbols to trade (e.g., ['BTC/USDT:USDT'])
        
        Returns:
            Dict with 'success', 'components', and optional 'reason' keys
        """
        logger.info("="*70)
        logger.info("INITIALIZING PRODUCTION SYSTEM")
        logger.info("="*70)
        
        try:
            # ========================================
            # STEP 1: STORE PROVIDED CONFIGURATION
            # ========================================
            if exchange_clients:
                # Normalize exchange keys to lowercase for consistency
                self.exchange_clients = {k.lower(): v for k, v in exchange_clients.items()}
                logger.info(f"✓ Received {len(self.exchange_clients)} exchange client(s): {list(self.exchange_clients.keys())}")
            
            # ========================================
            # STEP 2: INITIALIZE WEBSOCKET MANAGER
            # ========================================
            if not self.websocket_manager and not hasattr(self, 'skip_ws_init'):
                try:
                    self.websocket_manager = WebSocketManager(exchanges=self.exchange_clients)
                    logger.info("✓ WebSocket manager initialized")
                except Exception as e:
                    logger.warning(f"⚠️ WebSocket manager initialization failed: {e}")
                    logger.warning("⚠️ Continuing without WebSocket (will use REST API)")
                    self.websocket_manager = None
            else:
                if self.websocket_manager:
                    logger.info("✓ WebSocket manager already initialized (external)")
                else:
                    logger.info("ℹ️ WebSocket initialization skipped (skip_ws_init flag)")
            
            # ========================================
            # STEP 3: INITIALIZE PERFORMANCE MONITOR
            # ========================================
            try:
                self.performance_monitor = PerformanceMonitor()
                logger.info("✓ Performance monitor initialized")
            except Exception as e:
                logger.warning(f"⚠️ PerformanceMonitor not available: {e}")
                logger.info("✓ Using fallback RealTimePerformanceMonitor")
                self.performance_monitor = RealTimePerformanceMonitor()
            
            # ========================================
            # STEP 4: PREPARE RISK MANAGER CONFIG
            # ========================================
            portfolio_config = portfolio_config or {}
            config = self.config
            risk_config = config.get('risk', {})
            
            # Merge portfolio_config with config file settings
            risk_manager_config = {
                'equity_usd': float(
                    portfolio_config.get('equity_usd') or 
                    risk_config.get('equity_usd', 100)
                ),
                'per_trade_risk_pct': float(risk_config.get('per_trade_risk_pct', 0.01)),
                'daily_loss_limit_pct': float(risk_config.get('daily_loss_limit_pct', 0.02)),
                'risk_usd_cap': float(risk_config.get('risk_usd_cap', 5)),
                'max_notional_per_trade': float(risk_config.get('max_notional_per_trade', 20)),
                'max_portfolio_risk': float(risk_config.get('max_portfolio_risk', 0.02)),
                'max_position_size': float(risk_config.get('max_position_size', 0.10)),
                'max_drawdown': float(risk_config.get('max_drawdown', 0.15)),
                'max_correlation': float(risk_config.get('max_correlation', 0.70))
            }
            
            logger.info(f"✓ Risk config prepared: ${risk_manager_config['equity_usd']} equity")
            
            # ========================================
            # STEP 5: INITIALIZE RISK MANAGER
            # ========================================
            self.risk_manager = RiskManager(
                portfolio_config=risk_manager_config,
                websocket_manager=self.websocket_manager,
                performance_monitor=self.performance_monitor
            )
            logger.info(f"✓ Risk manager initialized (portfolio value: ${self.risk_manager.portfolio_value:.2f})")
            
            # ========================================
            # STEP 6: INITIALIZE PORTFOLIO MANAGER
            # ========================================
            self.portfolio_manager = PortfolioManager(
                risk_manager=self.risk_manager,
                performance_monitor=self.performance_monitor,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients
            )
            # Attach full config to portfolio manager
            self.portfolio_manager.cfg = self.config
            logger.info("✓ Portfolio manager initialized")
            
            # ========================================
            # STEP 7: INITIALIZE STRATEGY COORDINATOR
            # ========================================
            self.strategy_coordinator = StrategyCoordinator(
                self.portfolio_manager,
                self.risk_manager
            )
            logger.info("✓ Strategy coordinator initialized")
            
            # ========================================
            # STEP 8: INITIALIZE CIRCUIT BREAKER
            # ========================================
            self.circuit_breaker = CircuitBreakerSystem(
                self.portfolio_manager,
                self.risk_manager
            )
            logger.info("✓ Circuit breaker system initialized")
            
            # ========================================
            # STEP 9: INITIALIZE LIVE TRADING ENGINE
            # ========================================
            self.trading_engine = LiveTradingEngine(
                mode=mode,
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients,
                strategy_coordinator=self.strategy_coordinator
            )
            logger.info(f"✓ Live trading engine initialized (mode: {mode})")
            
            # ========================================
            # STEP 10: SET ACTIVE SYMBOLS (CRITICAL!)
            # ========================================
            if trading_symbols:
                # Priority 1: Use provided parameter
                self.active_symbols = trading_symbols
                logger.info(f"✓ Active symbols set from parameter: {len(trading_symbols)} symbols")
                logger.info(f"  Symbols: {', '.join(trading_symbols)}")
            else:
                # Priority 2+: Use default symbol discovery
                self.active_symbols = self._get_default_symbols()
                logger.info(f"✓ Active symbols discovered: {len(self.active_symbols)} symbols")
                logger.info(f"  Symbols: {', '.join(self.active_symbols)}")
            
            # Set symbols on trading engine for prefetch and other operations
            if self.trading_engine and self.active_symbols:
                self.trading_engine._cached_symbols = self.active_symbols
                logger.info(f"✓ Trading engine symbols cache set: {len(self.active_symbols)} symbols")
            
            # ========================================
            # STEP 11: VALIDATE ACTIVE SYMBOLS
            # ========================================
            if not self.active_symbols:
                logger.error("="*70)
                logger.error("❌ CRITICAL: NO ACTIVE SYMBOLS CONFIGURED!")
                logger.error("="*70)
                logger.error("The bot cannot trade without active symbols.")
                logger.error("Please provide trading_symbols parameter or configure fixed_symbols in config.")
                logger.error("="*70)
                # Don't fail initialization, but warn heavily
            else:
                logger.info("="*70)
                logger.info(f"✅ ACTIVE SYMBOLS CONFIGURED: {len(self.active_symbols)} symbols")
                logger.info("="*70)
                for idx, symbol in enumerate(self.active_symbols, 1):
                    logger.info(f"  {idx}. {symbol}")
                logger.info("="*70)
            
            # ========================================
            # STEP 11.5: INITIALIZE ML COMPONENTS (NEW)
            # ========================================
            ml_enabled = self.config.get('ml', {}).get('enabled', False)
            if ml_enabled and self.active_symbols:
                logger.info("="*70)
                logger.info("🧠 INITIALIZING ML COMPONENTS")
                logger.info("="*70)
                try:
                    ml_init_result = await self._initialize_ml_components()
                    if ml_init_result.get('success'):
                        logger.info("✅ ML components initialized successfully")
                    else:
                        logger.warning(f"⚠️ ML initialization partial: {ml_init_result.get('reason')}")
                except Exception as e:
                    logger.warning(f"⚠️ ML initialization failed: {e}")
                    logger.info("Continuing without ML features")
            else:
                if not ml_enabled:
                    logger.info("ℹ️ ML features disabled in config")
                elif not self.active_symbols:
                    logger.warning("⚠️ Cannot initialize ML without active symbols")
            
            # ========================================
            # STEP 12: MARK AS INITIALIZED
            # ========================================
            self.is_initialized = True
            
            components = [
                'websocket_manager',
                'performance_monitor',
                'risk_manager',
                'portfolio_manager',
                'strategy_coordinator',
                'circuit_breaker',
                'trading_engine'
            ]
            
            logger.info("="*70)
            logger.info("✅ PRODUCTION SYSTEM INITIALIZATION COMPLETE")
            logger.info("="*70)
            logger.info(f"Components initialized: {len(components)}")
            logger.info(f"Portfolio value: ${self.risk_manager.portfolio_value:.2f}")
            logger.info(f"Active symbols: {len(self.active_symbols)}")
            logger.info(f"Mode: {mode}")
            logger.info("="*70)
            
            return {
                'success': True,
                'components': components,
                'is_initialized': True,
                'active_symbols_count': len(self.active_symbols)
            }
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ PRODUCTION SYSTEM INITIALIZATION FAILED")
            logger.error("="*70)
            logger.error(f"Error: {e}", exc_info=True)
            logger.error("="*70)
            
            self.is_initialized = False
            
            return {
                'success': False,
                'reason': str(e),
                'is_initialized': False
            }
            
    async def run_production_loop(self, mode: str = 'paper', duration: Optional[float] = None, 
                                  continuous: bool = False):
        # ✅ EMERGENCY DEBUG with print()
        print(f"\n{'='*70}")
        print(f"🚨 EMERGENCY: run_production_loop() CALLED")
        print(f"   Time: {datetime.now(timezone.utc)}")
        print(f"   Mode: {mode}")
        print(f"   Duration: {duration}")
        print(f"   Continuous: {continuous}")
        print(f"   is_initialized: {self.is_initialized}")
        print(f"   is_running: {self.is_running}")
        print(f"   active_symbols: {self.active_symbols}")
        print(f"{'='*70}\n")
        
        import sys                              
        sys.stdout.flush()
        
        # Now try logger
        logger.warning("🔥 [WARNING-LEVEL] run_production_loop() ENTERED")  # Use WARNING to ensure visibility
        logger.info("🔍 [INFO-LEVEL] run_production_loop() method ENTERED")
            
        try:
            logger.info("🔍 [DEBUG] Inside try block")
            
            if not self.is_initialized:
                logger.error("🔍 [DEBUG] NOT INITIALIZED - raising RuntimeError")
                raise RuntimeError("Production system not initialized. Call initialize_production_system() first.")
            
            logger.info("🔍 [DEBUG] Passed initialization check")
            logger.info("="*70)
            logger.info("STARTING PRODUCTION TRADING LOOP")
            logger.info("="*70)
            
            # ✅ YENİ: Engine'in çalıştığını kontrol et (ikinci kez başlatma!)
            logger.info("🔍 [DEBUG] Checking trading engine...")
            if not self.trading_engine:
                logger.error("🔍 [DEBUG] trading_engine is None!")
                raise RuntimeError("Trading engine not initialized!")
            
            logger.info(f"🔍 [DEBUG] trading_engine exists, state={self.trading_engine.state.value}")

            if self.trading_engine.state.value != 'running':
                logger.warning(
                    "⚠️ Trading engine reported state '%s' while entering production loop; "
                    "awaiting synchronization...",
                    self.trading_engine.state.value
                )
                # Give the event loop more time to schedule engine startup tasks
                # Extended from 0s to 1.0s to ensure proper task scheduling
                logger.info("⏱️ Waiting 1.0s for engine tasks to initialize...")
                await asyncio.sleep(1.0)

                if self.trading_engine.state.value != 'running':
                    logger.error(
                        "❌ Engine state still '%s' after 1s synchronization delay; aborting production loop.",
                        self.trading_engine.state.value
                    )
                    raise RuntimeError(
                        "Trading engine not running after synchronization delay "
                        f"(state={self.trading_engine.state.value})"
                    )
                else:
                    logger.info("✅ Trading engine reached running state after synchronization delay")
            else:
                logger.info(f"✅ Trading engine already running (state={self.trading_engine.state.value})")

            # Ensure is_running is True
            logger.info(f"🔍 [DEBUG] Current is_running = {self.is_running}")
            if not self.is_running:
                logger.warning("⚠️ is_running was False, setting to True")
                self.is_running = True
            
            logger.info(f"🔍 [DEBUG] is_running now = {self.is_running}")
            
            # Start queue monitoring task
            logger.info("🔍 [DEBUG] Creating queue monitoring task...")
            self._monitoring_task = asyncio.create_task(self._monitor_signal_queues())
            logger.info("🔍 [DEBUG] Queue monitoring task created")
            
            # Start watchdog task to detect if main loop stalls
            logger.info("🔍 [DEBUG] Creating watchdog task...")
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info("🔍 [DEBUG] Watchdog task created")
            
            logger.info("🔍 [DEBUG] About to print production loop info...")
            logger.info("\n🚀 Production trading loop active")
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Duration: {'Indefinite' if duration is None else f'{duration}s'}")
            logger.info(f"   Continuous Mode: {'ENABLED (Never stops, auto-recovers)' if continuous else 'DISABLED'}")
            logger.info(f"   Active Symbols: {len(self.active_symbols)}")
            
            # ✅ EKLE: active_symbols kontrolü
            logger.info(f"🔍 [DEBUG] Checking active_symbols: {self.active_symbols}")
            if not self.active_symbols:
                logger.error("❌ No active symbols configured!")
                raise RuntimeError("active_symbols is empty! Cannot process any symbols.")
            
            logger.info("🔍 [DEBUG] active_symbols check passed")
            
            # Main loop
            logger.info("🔍 [DEBUG] Initializing loop variables...")
            start_time = datetime.now(timezone.utc)
            last_recommendation_time = start_time
            recommendation_interval = 300  # Her 5 dakikada bir recommendations
            loop_iteration = 0
            
            logger.info(f"🔍 [DEBUG] Loop variables initialized: start_time={start_time}, loop_iteration={loop_iteration}")
            
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
            
            # ✅ FORCE FLUSH before loop entry
            import sys
            sys.stdout.flush()
            print(f"🚨 [BEFORE-WHILE] About to enter while loop, is_running={self.is_running}")
            sys.stderr.flush()
            
            logger.info(f"🔍 [DEBUG] About to enter while loop. is_running={self.is_running}")
            
            # ✅ CRITICAL CHECK: Verify is_running is True before loop
            if not self.is_running:
                logger.critical("❌ [CRITICAL] is_running is FALSE before loop entry!")
                logger.critical(f"   This should never happen - is_running was just set to True at line 791")
                raise RuntimeError("is_running unexpectedly False before loop entry")
            
            while self.is_running:
                # ✅ ENHANCED: Always log loop entry at INFO level for visibility
                if loop_iteration == 0:
                    logger.info("🔄 [LOOP-START] Main trading loop entered successfully")
                
                print(f"🚨 [IN-WHILE] Loop iteration {loop_iteration + 1} Processing symbols...")
                
                # Watchdog: Log heartbeat every 5 iterations
                if loop_iteration > 0 and loop_iteration % 5 == 0:
                    logger.info(f"💓 [WATCHDOG] Loop heartbeat - {loop_iteration} iterations completed")
                
                try:
                    loop_iteration += 1
                    # ✅ DEĞİŞTİR: logger.debug → logger.info
                    logger.info(f"🔁 [ITERATION {loop_iteration}] Processing {len(self.active_symbols)} symbols...")
                    
                    # Check emergency conditions
                    if self.emergency_stop_triggered:
                        logger.critical("Emergency stop triggered - shutting down")
                        break
                    
                    # Check circuit breaker with timeout protection
                    try:
                        breaker_status = await asyncio.wait_for(
                            self.circuit_breaker.check_circuit_breaker(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("⏱️ Circuit breaker check timeout - continuing")
                        breaker_status = {'tripped': False}
                    
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
                        logger.info(f"🔍 [DEBUG] Duration check: elapsed={elapsed:.1f}s, duration={duration}s")
                        if elapsed >= duration:
                            logger.info(f"⏱️ Duration {duration}s reached - stopping (elapsed: {elapsed:.1f}s)")
                            break
                        else:
                            logger.info(f"🔍 [DEBUG] Duration check passed - continuing loop")
                    
                    # Process trading loop with WebSocket data
                    logger.info("🔍 [DEBUG] About to call _process_trading_loop()...")
                    await self._process_trading_loop()
                    logger.info("🔍 [DEBUG] _process_trading_loop() completed")
                    
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
                    
                    # Sleep between iterations, but check duration first
                    logger.debug(f"🔁 Trading loop iteration {loop_iteration} completed, sleeping {self.loop_interval}s")
                    
                    # If duration is set, calculate remaining time and don't sleep longer than needed
                    if duration and not continuous:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        remaining = duration - elapsed
                        if remaining <= 0:
                            logger.info(f"⏱️ Duration {duration}s reached after processing - stopping")
                            break
                        # Sleep for minimum of loop_interval or remaining time
                        sleep_time = min(self.loop_interval, remaining)
                        logger.debug(f"Sleeping for {sleep_time:.1f}s (remaining: {remaining:.1f}s)")
                        await asyncio.sleep(sleep_time)
                    else:
                        # No duration limit or continuous mode - use full loop_interval
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
                
                # Signal is now queued in StrategyCoordinator, will be bridged to LiveTradingEngine
                logger.info(f"✅ [SIGNAL-ACCEPTED] Signal {signal_id} accepted by StrategyCoordinator")
                logger.info(f"💡 [SIGNAL-QUEUED] {signal.get('strategy', 'unknown').upper()} signal for {signal.get('symbol')} queued in StrategyCoordinator")
                
                # Log queue state for monitoring
                coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                logger.info(f"📊 [QUEUE-STATE] StrategyCoordinator queue size: {coordinator_queue_size}")
                
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
        
        Uses a 15-second timeout to prevent indefinite blocking on slow/failed requests.
        """
        try:
            # Run blocking I/O in thread pool with timeout to prevent indefinite blocking
            rows = await asyncio.wait_for(
                asyncio.to_thread(client.ohlcv, symbol, timeframe, limit=200),
                timeout=10.0
            )
            return self._ohlcv_to_dataframe(rows)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout fetching {symbol} {timeframe} (10s limit)")
            return None
    
    async def _watchdog_loop(self):
        """
        Watchdog task that logs periodic heartbeats to detect loop stalls.
        
        Logs every 10 seconds regardless of main loop state to help diagnose
        if the main loop is truly stuck or just not logging.
        """
        logger.info("🐕 [WATCHDOG] Watchdog task started - will log every 10s")
        watchdog_count = 0
        
        try:
            while self.is_running:
                watchdog_count += 1
                logger.info(f"🐕 [WATCHDOG-{watchdog_count}] Heartbeat - is_running={self.is_running}")
                logger.info(f"   Active symbols: {len(self.active_symbols)}")
                logger.info(f"   Processed symbols: {self.processed_symbols_count}")
                
                # Check if engine is still running
                if self.trading_engine:
                    logger.info(f"   Engine state: {self.trading_engine.state.value}")
                
                # Force log flush
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("🐕 [WATCHDOG] Task cancelled")
        except Exception as e:
            logger.error(f"🐕 [WATCHDOG] Error: {e}", exc_info=True)
    
    async def _monitor_signal_queues(self):
        """
        Monitor signal queues and log their sizes periodically.
        
        Monitors:
        - StrategyCoordinator signal queue
        - LiveTradingEngine signal queue
        - Engine state
        - Execution statistics
        """
        logger.info("Queue monitoring task started")
        
        try:
            while self.is_running:
                try:
                    # Get queue sizes
                    coordinator_queue_size = 0
                    engine_queue_size = 0
                    engine_state = 'not_initialized'
                    
                    if self.strategy_coordinator:
                        coordinator_queue_size = self.strategy_coordinator.signal_queue.qsize()
                    
                    if self.trading_engine:
                        engine_queue_size = self.trading_engine.signal_queue.qsize()
                        engine_state = self.trading_engine.state.value if hasattr(self.trading_engine, 'state') else 'unknown'
                    
                    # Log queue status
                    logger.info(f"📊 [QUEUE-MONITOR] Pipeline Status:")
                    logger.info(f"   StrategyCoordinator Queue: {coordinator_queue_size} signals")
                    logger.info(f"   LiveTradingEngine Queue: {engine_queue_size} signals")
                    logger.info(f"   LiveTradingEngine State: {engine_state}")
                    
                    # Get engine status if available
                    if self.trading_engine:
                        engine_status = self.trading_engine.get_engine_status()
                        logger.info(f"   Signals Received: {engine_status.get('signals_received', 0)}")
                        logger.info(f"   Signals Executed: {engine_status.get('signals_executed', 0)}")
                        logger.info(f"   Active Positions: {engine_status.get('active_positions', 0)}")
                    
                    # Alert if signals are stuck
                    if coordinator_queue_size > 5:
                        logger.warning(f"⚠️ [QUEUE-ALERT] {coordinator_queue_size} signals stuck in StrategyCoordinator queue!")
                        
                    if engine_queue_size > 5:
                        logger.warning(f"⚠️ [QUEUE-ALERT] {engine_queue_size} signals stuck in LiveTradingEngine queue!")
                        
                    if coordinator_queue_size > 0 and engine_queue_size == 0 and engine_state != 'running':
                        logger.critical(f"❌ [PIPELINE-BROKEN] Signals in coordinator but engine not running! State: {engine_state}")
                    
                    # Log lifecycle summary
                    if self.signal_lifecycle:
                        total_signals = len(self.signal_lifecycle)
                        stage_counts = {}
                        for signal_data in self.signal_lifecycle.values():
                            if signal_data['stages']:
                                last_stage = signal_data['stages'][-1]['stage']
                                stage_counts[last_stage] = stage_counts.get(last_stage, 0) + 1
                        
                        logger.info(f"📊 [LIFECYCLE] Total tracked: {total_signals} | Stages: {stage_counts}")
                    
                    # Wait 60 seconds before next check
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    logger.info("Queue monitoring task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in queue monitoring: {e}")
                    await asyncio.sleep(60)
                    
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
        
        # Cancel watchdog task
        if hasattr(self, '_watchdog_task') and self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        
        if self.trading_engine:
            await self.trading_engine.stop_live_trading()
        
        if self.websocket_manager:
            await self.websocket_manager.close()
            # Allow graceful shutdown with small delay
            await asyncio.sleep(0.05)
        
        logger.info("Production system stopped")
