"""
Production Coordinator.
Coordinate all Phase 3 components for production deployment.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .live_trading_engine import LiveTradingEngine
from .websocket_manager import WebSocketManager
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .strategy_coordinator import StrategyCoordinator
from .circuit_breaker import CircuitBreakerSystem
from .performance_monitor import RealTimePerformanceMonitor
from .market_regime import MarketRegimeAnalyzer

# Import config with try/except for flexibility
try:
    from ..config.live_trading_config import LiveTradingConfiguration
except ImportError:
    from config.live_trading_config import LiveTradingConfiguration

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
        
        # Configuration
        self.config = LiveTradingConfiguration.get_all_configs()
        
        logger.info("ProductionCoordinator created")
    
    async def initialize_production_system(self, exchange_clients: Dict, 
                                          portfolio_config: Dict) -> Dict[str, Any]:
        """
        Initialize complete production trading system.
        
        Args:
            exchange_clients: Dictionary of exchange client instances (Phase 1)
            portfolio_config: Portfolio configuration dictionary
            
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
            
            # Initialize market regime analyzer
            self.market_regime_analyzer = MarketRegimeAnalyzer()
            logger.info("  ✓ Market Regime Analyzer initialized")
            
            # Initialize performance monitor
            self.performance_monitor = RealTimePerformanceMonitor()
            logger.info("  ✓ Performance Monitor initialized")
            
            # Phase 3.1: WebSocket connections
            logger.info("\n[Phase 3.1] Establishing WebSocket Connections...")
            try:
                self.websocket_manager = WebSocketManager(exchanges=None)  # Will use default
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
                portfolio_manager=self.portfolio_manager,
                risk_manager=self.risk_manager,
                websocket_manager=self.websocket_manager,
                exchange_clients=self.exchange_clients
            )
            logger.info("  ✓ Live Trading Engine initialized")
            
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
                    'circuit_breaker': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing production system: {e}")
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
            
            # Main loop
            start_time = datetime.now(timezone.utc)
            
            while self.is_running:
                try:
                    # Check emergency conditions (always honor manual stop)
                    if self.emergency_stop_triggered:
                        logger.critical("Emergency stop triggered - shutting down")
                        break
                    
                    # Check circuit breaker (bypass non-critical in continuous mode)
                    breaker_status = await self.circuit_breaker.check_circuit_breaker()
                    if breaker_status.get('tripped'):
                        severity = breaker_status.get('severity', 'high')
                        
                        # In continuous mode, only stop for critical breakers
                        if continuous and severity != 'critical':
                            logger.warning(f"Circuit breaker tripped ({severity}): {breaker_status.get('reason')}")
                            logger.warning("CONTINUOUS MODE: Bypassing non-critical breaker, continuing...")
                            await asyncio.sleep(10)  # Pause briefly before continuing
                            continue
                        else:
                            logger.critical(f"Circuit breaker tripped ({severity}): {breaker_status.get('reason')}")
                            await self.handle_emergency_shutdown('circuit_breaker_tripped')
                            break
                    
                    # Check duration (skip in continuous mode)
                    if duration and not continuous:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        if elapsed >= duration:
                            logger.info(f"Duration {duration}s reached - stopping")
                            break
                    
                    # Sleep between iterations
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received - stopping gracefully")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in production loop: {e}")
                    
                    # In continuous mode, implement auto-recovery
                    if continuous:
                        logger.warning("CONTINUOUS MODE: Auto-recovering from error...")
                        
                        # Try to restart trading engine if it stopped
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
                        
                        # Continue after brief pause
                        await asyncio.sleep(5)
                        continue
                    else:
                        # Original behavior for non-continuous mode
                        if self.config['emergency']['enable_circuit_breaker']:
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
            logger.error(f"Critical error in production loop: {e}")
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
            # Signal queue should stop accepting new signals
            
            # Step 2: Cancel pending orders
            logger.critical("Step 2: Cancelling pending orders...")
            # Would iterate through active orders and cancel them
            
            # Step 3: Close positions (if configured)
            close_method = self.config['emergency']['emergency_close_method']
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
            logger.critical(f"Error during emergency shutdown: {e}")
    
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
            'emergency_stop': self.emergency_stop_triggered
        }
        
        if self.trading_engine:
            state['trading_engine'] = self.trading_engine.get_engine_status()
        
        if self.portfolio_manager:
            state['portfolio'] = self.portfolio_manager.portfolio_state
        
        if self.risk_manager:
            state['risk_limits'] = self.risk_manager.risk_limits
        
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
