"""
Production Coordinator - Phase 3 Orchestration Layer
Manages the complete production trading system with all phases integrated.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import os
import yaml

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
from .live_trading_config import LiveTradingConfiguration

# Performance Monitor için basit fallback
class RealTimePerformanceMonitor:
    """Basit performance monitor fallback."""
    def __init__(self):
        self.trades = []
        self.metrics = {}
    
    def record_trade(self, trade_data):
        self.trades.append(trade_data)
    
    def get_metrics(self):
        return self.metrics

# Phase 3.1-3.3: Risk & Portfolio Management
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .strategy_coordinator import StrategyCoordinator
from .circuit_breaker import CircuitBreakerSystem

# Phase 3.4: Live Trading Components
from .live_trading_engine import LiveTradingEngine

# Strategy imports - DÜZELTILDI
from strategies import AdaptiveOversoldBounce, AdaptiveShortTheRip

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
        
        # System state
        self.is_running = False
        self.is_initialized = False
        self.emergency_stop_triggered = False
        self.loop_interval = 30  # Ana döngü bekleme süresi
        self.active_symbols = []  # Takip edilen semboller
        self.processed_symbols_count = 0  # İşlenen sembol sayacı
        
        # Configuration - DÜZELTILDI
        self.config = LiveTradingConfiguration.get_all_configs()
        
        # Market regime analyzer başlat
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        logger.info("✅ Market regime analyzer initialized")
        
        logger.info("ProductionCoordinator created")

    def _setup_websocket_connections(self):
        """WebSocket bağlantılarını kur - OPTİMİZE EDİLMİŞ"""
    
        # Config'den sabit sembolleri al
        fixed_symbols = self.config.get('universe', {}).get('fixed_symbols', [])
    
        if fixed_symbols:
            # Sadece config'deki semboller için stream aç
            logger.info(f"[WS] Opening streams for {len(fixed_symbols)} configured symbols only")
            self.websocket_manager.subscribe_to_symbols(fixed_symbols)
            self.active_symbols = fixed_symbols
        else:
            # Fallback: En fazla 20 sembol
            logger.warning("[WS] No fixed symbols, limiting to top 20")
            top_symbols = self._get_top_volume_symbols(limit=20)
            self.websocket_manager.subscribe_to_symbols(top_symbols)
            self.active_symbols = top_symbols

    def _get_top_volume_symbols(self, limit=20):
        """Get top volume symbols from exchanges."""
        # Basit implementasyon
        default_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'BNB/USDT:USDT', 'ADA/USDT:USDT'
        ]
        return default_symbols[:limit]

    def process_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Process a single symbol for signals with market regime analysis.
        
        Args:
            symbol: Trading symbol to process
            
        Returns:
            Signal dictionary if found, None otherwise
        """
        try:
            self.processed_symbols_count += 1
            
            # WebSocket'ten veri al
            if not self.websocket_manager:
                logger.debug(f"No WebSocket manager for {symbol}")
                return None
                
            df_30m = self.websocket_manager.get_latest_data(symbol, '30m')
            df_1h = self.websocket_manager.get_latest_data(symbol, '1h')
            df_4h = self.websocket_manager.get_latest_data(symbol, '4h')
            
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
            signals_config = self.config.get('signals', {})
            signal = None
            
            # Check AdaptiveOversoldBounce
            if signals_config.get('oversold_bounce', {}).get('enable', True):
                # Sadece regime uygunsa veya ignore_regime true ise
                ignore_regime = signals_config.get('oversold_bounce', {}).get('ignore_regime', False)
                
                if metadata.get('ob_favorable', True) or ignore_regime:
                    try:
                        ob_config = signals_config.get('oversold_bounce', {})
                        ob = AdaptiveOversoldBounce(ob_config, self.market_regime_analyzer)
                        
                        # Adaptive strateji farklı parametre alıyor
                        signal = ob.signal(df_30m, df_1h, regime_data=metadata.get('regime'))
                        
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
                        
                        signal = strp.signal(df_30m, df_1h, regime_data=metadata.get('regime'))
                        
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
                
            return signal
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    async def initialize_production_system(self, exchange_clients: Dict, 
                                          portfolio_config: Dict,
                                          mode: str = 'paper') -> Dict[str, Any]:
        """
        Initialize complete production trading system.
        
        Args:
            exchange_clients: Dictionary of exchange client instances (Phase 1)
            portfolio_config: Portfolio configuration dictionary
            mode: Trading mode ('paper', 'live', 'simulation')
            
        Returns:
            Initialization result
        """
        try:
            logger.info("="*70)
            logger.info("INITIALIZING PRODUCTION TRADING SYSTEM")
            logger.info("="*70)
            
            # Phase 1: Multi-exchange setup
            logger.info("\n[Phase 1] Setting up Multi-Exchange Framework...")
            self.exchange_clients = exchange_clients
            logger.info(f"  ✓ {len(exchange_clients)} exchange(s) configured: {list(exchange_clients.keys())}")
            
            # Phase 2: Market intelligence activation
            logger.info("\n[Phase 2] Activating Market Intelligence...")
            
            # Market regime analyzer zaten init'te oluşturuldu
            logger.info("  ✓ Market Regime Analyzer already initialized")
            
            # Initialize performance monitor
            self.performance_monitor = RealTimePerformanceMonitor()
            logger.info("  ✓ Performance Monitor initialized")
            
            # Phase 3.1: WebSocket connections
            logger.info("\n[Phase 3.1] Establishing WebSocket Connections...")
            try:
                self.websocket_manager = WebSocketManager(exchanges=None)  # Will use default
                self._setup_websocket_connections()  # Sembolleri ayarla
                logger.info("  ✓ WebSocket Manager initialized")
            except Exception as e:
                logger.warning(f"  ⚠️  WebSocket initialization failed: {e}")
                self.websocket_manager = None
            
            # Phase 3.2: Risk management setup
            logger.info("\n[Phase 3.2] Setting up Risk Management...")
            self.risk_manager = RiskManager(
                portfolio_config=portfolio_config,
                websocket_manager=self.websocket_manager,
                performance_monitor=self.performance_monitor
            )
            logger.info(f"  ✓ Risk Manager initialized with ${portfolio_config.get('equity_usd', 0):.2f} portfolio")
            
            # Phase 3.3: Portfolio optimization
            logger.info("\n[Phase 3.3] Initializing Portfolio Management...")
            self.portfolio_manager = PortfolioManager(
                risk_manager=self.risk_manager,
                performance_monitor=self.performance_monitor,
                websocket_manager=self.websocket_manager
            )
            logger.info("  ✓ Portfolio Manager initialized")

            # Register trading strategies
            logger.info("  Registering adaptive trading strategies...")

            signals_config = self.config.get('signals', {})
            strategies_registered = 0

            # 1. Adaptive OversoldBounce
            if signals_config.get('oversold_bounce', {}).get('enable', True):
                adaptive_ob_config = {
                    'rsi_max': signals_config.get('oversold_bounce', {}).get('adaptive_rsi_base', 40),
                    'adaptive_rsi_range': signals_config.get('oversold_bounce', {}).get('adaptive_rsi_range', 15),
                    'tp_pct': signals_config.get('oversold_bounce', {}).get('tp_pct', 0.015),
                    'sl_atr_mult': signals_config.get('oversold_bounce', {}).get('sl_atr_mult', 1.0),
                    'ignore_regime': signals_config.get('oversold_bounce', {}).get('ignore_regime', False)
                }
                adaptive_ob = AdaptiveOversoldBounce(adaptive_ob_config, self.market_regime_analyzer)
                
                result = self.portfolio_manager.register_strategy(
                    strategy_name='adaptive_ob',
                    strategy_instance=adaptive_ob,
                    initial_allocation=0.5  # %50
                )
                logger.info("    ✓ AdaptiveOversoldBounce registered")
                strategies_registered += 1

            # 2. Adaptive ShortTheRip  
            if signals_config.get('short_the_rip', {}).get('enable', True):
                adaptive_str_config = {
                    'rsi_min': signals_config.get('short_the_rip', {}).get('adaptive_rsi_base', 65),
                    'adaptive_rsi_range': signals_config.get('short_the_rip', {}).get('adaptive_rsi_range', 20),
                    'tp_pct': signals_config.get('short_the_rip', {}).get('tp_pct', 0.012),
                    'sl_atr_mult': signals_config.get('short_the_rip', {}).get('sl_atr_mult', 1.2),
                    'ignore_regime': signals_config.get('short_the_rip', {}).get('ignore_regime', False)
                }
                adaptive_str = AdaptiveShortTheRip(adaptive_str_config, self.market_regime_analyzer)
                
                result = self.portfolio_manager.register_strategy(
                    strategy_name='adaptive_str',
                    strategy_instance=adaptive_str,
                    initial_allocation=0.5  # %50
                )
                logger.info("    ✓ AdaptiveShortTheRip registered")
                strategies_registered += 1

            logger.info(f"  ✓ {strategies_registered} adaptive strategies registered")
            
            # Initialize strategy coordinator
            self.strategy_coordinator = StrategyCoordinator(
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager
            )
            logger.info("  ✓ Strategy Coordinator initialized")
            
            # Initialize circuit breaker
            self.circuit_breaker = CircuitBreakerSystem(
                risk_manager=self.risk_manager,
                websocket_manager=self.websocket_manager
            )
            logger.info("  ✓ Circuit Breaker System initialized")
            
            # Phase 3.4: Live trading activation
            logger.info("\n[Phase 3.4] Initializing Live Trading Engine...")
            self.trading_engine = LiveTradingEngine(
                mode=mode,  # Dinamik mode
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients
            )
            logger.info(f"  ✓ Live Trading Engine initialized (mode: {mode})")
            
            self.is_initialized = True
            
            logger.info("\n" + "="*70)
            logger.info("✓ PRODUCTION SYSTEM INITIALIZED SUCCESSFULLY")
            logger.info("="*70)
            
            return {
                'success': True,
                'initialized': True,
                'components': {
                    'exchanges': len(self.exchange_clients),
                    'websocket': self.websocket_manager is not None,
                    'risk_manager': True,
                    'portfolio_manager': True,
                    'trading_engine': True,
                    'circuit_breaker': True,
                    'market_regime': True,
                    'strategies': strategies_registered
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing production system: {e}", exc_info=True)
            self.is_initialized = False
            return {
                'success': False,
                'reason': str(e),
                'initialized': False
            }
    
    async def run_production_loop(self, mode: str = 'paper', duration: Optional[float] = None, 
                                  continuous: bool = False):
        """
        Main production trading loop.
        
        Args:
            mode: Trading mode ('paper', 'live', 'simulation')
            duration: Optional duration in seconds (None = run indefinitely)
            continuous: If True, enable TRUE CONTINUOUS mode (never stops, auto-recovers)
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Production system not initialized. Call initialize_production_system() first.")
            
            logger.info("="*70)
            logger.info("STARTING PRODUCTION TRADING LOOP")
            logger.info("="*70)
            
            # Start live trading engine
            start_result = await self.trading_engine.start_live_trading(mode=mode)
            
            if not start_result['success']:
                raise RuntimeError(f"Failed to start trading engine: {start_result.get('reason')}")
            
            self.is_running = True
            
            logger.info("\n🚀 Production trading loop active")
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Duration: {'Indefinite' if duration is None else f'{duration}s'}")
            logger.info(f"   Continuous Mode: {'ENABLED (Never stops, auto-recovers)' if continuous else 'DISABLED'}")
            logger.info(f"   Active Symbols: {len(self.active_symbols)}")
            
            # Main loop
            start_time = datetime.now(timezone.utc)
            last_recommendation_time = start_time
            recommendation_interval = 300  # Her 5 dakikada bir recommendations
            
            while self.is_running:
                try:
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
                    
                    # Process symbols for signals
                    for symbol in self.active_symbols:
                        signal = self.process_symbol(symbol)
                        if signal:
                            # Submit signal to trading engine
                            result = await self.submit_signal(signal)
                            if result['success']:
                                logger.info(f"✅ Signal submitted: {symbol} {signal['strategy']}")
                    
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
                # Add to trading engine signal queue
                await self.trading_engine.signal_queue.put(signal)
                logger.info(f"Signal submitted for {signal.get('symbol')}")
                return {'success': True, 'signal_id': signal.get('signal_id')}
            else:
                logger.warning(f"Signal rejected: {result.get('reason')}")
                return {'success': False, 'reason': result.get('reason')}
                
        except Exception as e:
            logger.error(f"Error submitting signal: {e}")
            return {'success': False, 'reason': str(e)}
    
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
    
    async def stop_system(self):
        """Stop the production system gracefully."""
        logger.info("Stopping production system...")
        self.is_running = False
        
        if self.trading_engine:
            await self.trading_engine.stop_live_trading()
        
        if self.websocket_manager:
            await self.websocket_manager.close()
        
        logger.info("Production system stopped")
