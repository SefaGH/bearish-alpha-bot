#!/usr/bin/env python3
"""
Live Trading Launcher for Bearish Alpha Bot

[... mevcut docstring ...]
"""

import sys
import os
import asyncio
import logging
import argparse
import time
import signal
import logging

# Production için sadece WARNING ve üstü
if os.getenv('PRODUCTION', 'false').lower() == 'true':
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
else:
    # Development/test için INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        
        logger.info("[WS-OPT] Optimized WebSocket Manager initialized")
    
    def setup_from_config(self, config: Dict[str, Any]) -> None:
        """
        Setup WebSocket configuration from config.
        
        Args:
            config: Configuration dictionary
        """
        universe_cfg = config.get('universe', {})
        self.fixed_symbols = universe_cfg.get('fixed_symbols', [])
        
        # WebSocket configuration
        ws_cfg = config.get('websocket', {
            'enabled': True,
            'max_streams_per_exchange': {
                'bingx': 10,
                'binance': 20,
                'kucoinfutures': 15,
                'default': 10
            }
        })
        
        self.max_streams_config = ws_cfg.get('max_streams_per_exchange', {})
        
        logger.info(f"[WS-OPT] Configured with {len(self.fixed_symbols)} fixed symbols")
        
        if not self.fixed_symbols:
            logger.warning("[WS-OPT] No fixed symbols configured!")
    
    async def initialize_websockets(self, exchange_clients: Dict[str, Any]) -> bool:
        """
        Initialize WebSocket connections with optimization.
        
        Args:
            exchange_clients: Dictionary of exchange clients
            
        Returns:
            True if initialization successful
        """
        try:
            # Check if we have fixed symbols
            if not self.fixed_symbols:
                logger.warning("[WS-OPT] No fixed symbols, WebSocket disabled")
                return False
            
            # Import WebSocketManager lazily
            from core.websocket_manager import WebSocketManager
            
            # Create optimized manager
            self.ws_manager = WebSocketManager(
                exchanges=exchange_clients,
                config=self.config
            )
            
            # Setup stream limits per exchange
            for exchange_name in exchange_clients.keys():
                max_streams = self.max_streams_config.get(
                    exchange_name,
                    self.max_streams_config.get('default', 10)
                )
                logger.info(f"[WS-OPT] {exchange_name}: Max streams set to {max_streams}")
            
            # Subscribe to fixed symbols with optimization
            tasks = await self._subscribe_optimized()
            
            if tasks:
                logger.info(f"[WS-OPT] ✅ WebSocket initialized with {len(tasks)} streams")
                self.is_initialized = True
                return True
            else:
                logger.warning("[WS-OPT] No WebSocket streams started")
                return False
                
        except Exception as e:
            logger.error(f"[WS-OPT] Failed to initialize WebSocket: {e}")
            return False
    
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
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket connections."""
        if self.ws_manager:
            await self.ws_manager.close()
            logger.info("[WS-OPT] WebSocket connections closed")

    async def _monitor_websocket_health(self):
        """Enhanced WebSocket health monitor with error recovery"""
        logger.info("Starting WebSocket health monitor...")
    
        consecutive_errors = 0
        max_consecutive_errors = 3
    
        while self.ws_optimizer and self.ws_optimizer.is_initialized:
            try:
                await asyncio.sleep(300)  # Her 5 dakikada bir kontrol
            
                status = await self.ws_optimizer.get_stream_status()
            
                # Parse frame hataları için özel kontrol
                error_count = status.get('error_count', 0)
                if error_count > 0:
                    logger.warning(f"⚠️ WebSocket errors detected: {error_count}")
                
                    # Eğer parse_frame hatası varsa
                    if status.get('parse_frame_errors', 0) > 0:
                        logger.error("❌ parse_frame errors detected! Attempting recovery...")
                    
                        # WebSocket'leri yeniden başlat
                        await self._restart_websockets_with_backoff()
                        consecutive_errors = 0
                        continue
            
                # Normal sağlık kontrolü
                if status['active_streams'] > 50:
                    logger.warning(f"⚠️ Too many WebSocket streams: {status['active_streams']}")
                    if self.telegram:
                        await self.telegram.send_async(
                            f"⚠️ <b>WebSocket Warning</b>\n"
                            f"Active streams: {status['active_streams']}\n"
                            f"Consider reducing symbols"
                        )
            
                elif status['active_streams'] == 0:
                    consecutive_errors += 1
                    logger.error(f"❌ No active WebSocket streams! (attempt {consecutive_errors}/{max_consecutive_errors})")
                
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical("❌ WebSocket completely failed after multiple attempts!")
                        if self.telegram:
                            await self.telegram.send_async(
                                "🛑 <b>CRITICAL</b>\n"
                                "WebSocket system failure!\n"
                                "Manual intervention required."
                            )
                        # Sistem durması gerekebilir
                        await self._emergency_shutdown("WebSocket system failure")
                    else:
                        # Yeniden başlatma denemesi
                        await self._restart_websockets_with_backoff()
            
                else:
                    # Her şey normal
                    consecutive_errors = 0
                    logger.info(f"✅ WebSocket healthy: {status['active_streams']} streams active")
                
            except Exception as e:
                logger.error(f"WebSocket monitor error: {e}")
                consecutive_errors += 1
            
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Monitor failed {max_consecutive_errors} times!")
                    break
                
                await asyncio.sleep(60)

    async def _restart_websockets_with_backoff(self):
        """Restart WebSockets with exponential backoff"""
        max_attempts = 3
        base_delay = 5  # seconds
    
        for attempt in range(max_attempts):
            try:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
            
                logger.info(f"Restarting WebSockets (attempt {attempt + 1}/{max_attempts})...")
            
                if attempt > 0:
                    logger.info(f"Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
            
                # Önce mevcut bağlantıları kapat
                await self.ws_optimizer.shutdown()
                await asyncio.sleep(2)
            
                # Yeniden başlat
                await self.ws_optimizer.initialize_websockets(self.exchange_clients)
            
                # Başarılı mı kontrol et
                await asyncio.sleep(5)  # Stabilizasyon için bekle
                status = await self.ws_optimizer.get_stream_status()
            
                if status['active_streams'] > 0:
                    logger.info(f"✅ WebSocket restart successful! {status['active_streams']} streams active")
                    if self.telegram:
                        await self.telegram.send_async(
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

# ============= End of WebSocket Optimization Manager =============


class HealthMonitor:
    """
    HEALTH MONITORING SYSTEM (Layer 3 Guardian)
    
    [... mevcut kod ...]
    """
    # Mevcut HealthMonitor sınıfı aynen kalıyor
    def __init__(self, telegram: Optional[Telegram] = None):
        """
        Initialize health monitor.
        
        Args:
            telegram: Telegram notifier for health alerts
        """
        self.telegram = telegram
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_interval = 300  # 5 minutes
        
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
        self.monitoring_active = False
        self.monitor_task = None
        
        logger.info("="*70)
        logger.info("HEALTH MONITORING SYSTEM INITIALIZED (Layer 3 Guardian)")
        logger.info("="*70)
        logger.info(f"Heartbeat Interval: {self.heartbeat_interval}s")
        logger.info("="*70)
    
    async def start_monitoring(self):
        """Start health monitoring task."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("✓ Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring task."""
        self.monitoring_active = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("✓ Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main health monitoring loop."""
        try:
            while self.monitoring_active:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                time_since_last = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
                
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
            logger.info("Health monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in health monitor: {e}")
    
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
    
    [... mevcut class docstring ...]
    """
    
    # 8 diversified trading pairs as specified
    TRADING_PAIRS = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'SOL/USDT:USDT',
        'BNB/USDT:USDT',
        'ADA/USDT:USDT',
        'DOT/USDT:USDT',
        'LTC/USDT:USDT',
        'AVAX/USDT:USDT'
    ]
    
    # Live trading configuration with real USDT
    CAPITAL_USDT = 100.0  # 100 USDT (real trading capital)
    
    # Risk parameters as specified
    RISK_PARAMS = {
        'max_position_size': 0.15,    # 15% max position size
        'stop_loss_pct': 0.05,        # 5% stop loss
        'take_profit_pct': 0.10,      # 10% take profit
        'max_drawdown': 0.15,         # 15% max drawdown
        'max_portfolio_risk': 0.02,   # 2% risk per trade
        'max_correlation': 0.70,      # 70% max correlation
    }
    
    def __init__(self, mode: str = 'live', dry_run: bool = False, 
                 infinite: bool = False, auto_restart: bool = False,
                 max_restarts: int = 1000, restart_delay: int = 30,
                 debug_mode: bool = False):
        """
        Initialize live trading launcher.
        
        [... mevcut init docstring ...]
        """
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
        
        logger.info("="*70)
        logger.info("BEARISH ALPHA BOT - LIVE TRADING LAUNCHER")
        logger.info("="*70)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Capital: {self.CAPITAL_USDT} USDT")
        logger.info(f"Exchange: BingX")
        logger.info(f"Trading Pairs: {len(self.TRADING_PAIRS)}")
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
        """
        OPTIMIZED BingX initialization - verify only 8 trading pairs, not 2528 markets.
        
        Returns:
            True if connection successful
        """
        logger.info("\n[2/8] Initializing BingX Exchange Connection...")
        
        try:
            # Create BingX client with credentials
            bingx_creds = {
                'apiKey': os.getenv('BINGX_KEY'),
                'secret': os.getenv('BINGX_SECRET'),
            }
            
            bingx_client = CcxtClient('bingx', bingx_creds)
            
            # WebSocket optimization: Set required symbols only
            bingx_client.set_required_symbols(self.TRADING_PAIRS)
            logger.info(f"✓ BingX client optimized for {len(self.TRADING_PAIRS)} symbols only")
            
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
            
            # Verify ONLY our 8 trading pairs (not all 2528 markets!)
            logger.info(f"Verifying {len(self.TRADING_PAIRS)} trading pairs...")
            verified_pairs = []
            
            for pair in self.TRADING_PAIRS:
                try:
                    ticker = bingx_client.fetch_ticker(pair)
                    verified_pairs.append(pair)
                    logger.info(f"  ✓ {pair}: ${ticker['last']:.2f}")
                except Exception as e:
                    logger.warning(f"  ❌ {pair}: {e}")
            
            if len(verified_pairs) >= 6:  # Allow for some pair failures
                logger.info(f"✓ {len(verified_pairs)}/{len(self.TRADING_PAIRS)} trading pairs verified")
                return True
            else:
                logger.error(f"Only {len(verified_pairs)}/{len(self.TRADING_PAIRS)} pairs verified")
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
        """
        Initialize adaptive trading strategies.
        
        [... mevcut kod ...]
        """
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # Initialize regime analyzer for adaptive strategies
            regime_analyzer = MarketRegimeAnalyzer()
            
            # Strategy configurations
            adaptive_ob_config = {
                'rsi_max': 30,
                'tp_pct': 0.015,
                'sl_atr_mult': 1.0
            }
            
            adaptive_str_config = {
                'rsi_min': 70,
                'tp_pct': 0.012,
                'sl_atr_mult': 1.0
            }
            
            # Adaptive Oversold Bounce strategy
            self.strategies['adaptive_ob'] = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
            logger.info("✓ Adaptive Oversold Bounce strategy initialized")
            
            # Adaptive Short The Rip strategy
            self.strategies['adaptive_str'] = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
            logger.info("✓ Adaptive Short The Rip strategy initialized")
            
            logger.info(f"\n✓ {len(self.strategies)} strategies ready for trading")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategies: {e}")
            return False
    
    async def _initialize_production_system(self) -> bool:
        """
        Initialize Phase 3 production coordinator with all components.
        
        Returns:
            True if initialization successful
        """
        logger.info("\n[6/8] Initializing Production Trading System...")
        
        try:
            # Load configuration for WebSocket optimization
            import yaml
            config_path = os.getenv('CONFIG_PATH', 'config/config.example.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Setup WebSocket optimizer with config
            self.ws_optimizer.setup_from_config(config)
            
            # Initialize WebSocket connections
            ws_initialized = await self.ws_optimizer.initialize_websockets(self.exchange_clients)
            if ws_initialized:
                logger.info("✓ WebSocket connections optimized and initialized")
            else:
                logger.warning("⚠️  WebSocket initialization failed, continuing without real-time data")
            
            # Portfolio configuration with USDT capital
            portfolio_config = {
                'equity_usd': self.CAPITAL_USDT,  # 100 USDT
                'max_portfolio_risk': self.RISK_PARAMS['max_portfolio_risk'],
                'max_position_size': self.RISK_PARAMS['max_position_size'],
                'max_drawdown': self.RISK_PARAMS['max_drawdown']
            }
            
            # Initialize production coordinator
            self.coordinator = ProductionCoordinator()
            
            # Pass WebSocket manager to coordinator if initialized
            if ws_initialized and self.ws_optimizer.ws_manager:
                self.coordinator.ws_manager = self.ws_optimizer.ws_manager
                logger.info("✓ WebSocket manager integrated with production coordinator")
            
            # Initialize complete production system
            init_result = await self.coordinator.initialize_production_system(
                exchange_clients=self.exchange_clients,
                portfolio_config=portfolio_config,
                mode=self.mode  # ✅ Mode'u ekle!
            )
            
            if not init_result['success']:
                logger.error(f"❌ Production system initialization failed: {init_result.get('reason')}")
                return False
            
            logger.info("✓ Production system initialized successfully")
            logger.info(f"  Components: {init_result['components']}")
            
            # Get WebSocket status
            if self.ws_optimizer:
                ws_status = await self.ws_optimizer.get_stream_status()
                logger.info(f"✓ WebSocket Status: {ws_status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize production system: {e}")
            return False

        async def _monitor_websocket_health(self):
            """WebSocket sağlık kontrolü loop'u - periyodik monitoring"""
            logger.info("Starting WebSocket health monitor...")
    
            while self.ws_optimizer and self.ws_optimizer.is_initialized:
                try:
                    await asyncio.sleep(300)  # Her 5 dakikada bir kontrol
            
                    status = await self.ws_optimizer.get_stream_status()
            
                    # Log current status
                    logger.info(f"[WS-MONITOR] Active streams: {status['active_streams']}/{status['total_streams']}")
            
                    # Sağlık kontrolü
                    if status['active_streams'] > 50:
                        logger.warning(f"⚠️ Too many WebSocket streams: {status['active_streams']}")
                        if self.telegram:
                            self.telegram.send(
                                f"⚠️ <b>WebSocket Warning</b>\n"
                                f"Active streams: {status['active_streams']}\n"
                                f"Consider reducing symbols"
                            )
            
                    elif status['active_streams'] == 0:
                        logger.error("❌ No active WebSocket streams!")
                        if self.telegram:
                            self.telegram.send("❌ <b>WebSocket Lost!</b> Attempting restart...")
                
                        # WebSocket'leri yeniden başlat
                        logger.info("Attempting WebSocket reconnection...")
                        await self.ws_optimizer.initialize_websockets(self.exchange_clients)
                
                        if self.ws_optimizer.is_initialized:
                            logger.info("✅ WebSocket reconnection successful")
                            if self.telegram:
                                self.telegram.send("✅ WebSocket reconnected successfully")
                        else:
                            logger.error("Failed to reconnect WebSocket")
            
                    else:
                        logger.info(f"✅ WebSocket healthy: {status['active_streams']} streams active")
                
                except Exception as e:
                    logger.error(f"WebSocket monitor error: {e}")
                    await asyncio.sleep(60)  # Hata durumunda 1 dakika bekle

        async def _shutdown(self) -> None:  # <-- Bu metod zaten var, üstüne ekliyorsunuz
            """Graceful shutdown of trading system."""
            # ... mevcut kod ...
    
    async def _register_strategies(self) -> bool:
        """
        Register trading strategies with portfolio manager.
        
        [... mevcut kod ...]
        """
        logger.info("\n[7/8] Registering Trading Strategies...")
        
        try:
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
            if self.ws_optimizer and self.ws_optimizer.is_initialized:
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
    
    async def _start_trading_loop(self, duration: Optional[float] = None) -> None:
        """Start the main trading loop with WebSocket optimization."""
        logger.info("\n" + "="*70)
        logger.info("STARTING LIVE TRADING")
        
        Args:
            duration: Optional duration in seconds (None for indefinite)
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING LIVE TRADING")
        logger.info("="*70)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Duration: {'Indefinite' if duration is None else f'{duration}s'}")
        logger.info(f"Trading Pairs: {len(self.TRADING_PAIRS)}")
        logger.info(f"WebSocket: {'OPTIMIZED' if self.ws_optimizer.is_initialized else 'DISABLED'}")
        logger.info("="*70)
        
        # Send Telegram notification
        if self.telegram:
            ws_info = "WebSocket OPTIMIZED ✅" if self.ws_optimizer.is_initialized else "REST API mode"
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
        
        try:
            # Start health monitoring if enabled
            if self.health_monitor:
                await self.health_monitor.start_monitoring()
            
            # Start production trading loop with continuous mode if enabled
            await self.coordinator.run_production_loop(
                mode=self.mode,
                duration=duration,
                continuous=self.infinite
            )
            
        except KeyboardInterrupt:
            logger.info("\n⚠ Keyboard interrupt received - initiating shutdown...")
            await self._shutdown()
            
        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
            if self.health_monitor:
                self.health_monitor.record_error(str(e))
            await self._emergency_shutdown(f"Critical error: {e}")
        
        finally:
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            # Shutdown WebSocket connections
            if self.ws_optimizer:
                await self.ws_optimizer.shutdown()
    
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
                if self.ws_optimizer and self.ws_optimizer.is_initialized:
                    ws_status = await self.ws_optimizer.get_stream_status()
                    msg += f"\nWebSocket streams: {ws_status['active_streams']}"
                self.telegram.send(msg)
            
            logger.info("="*70)
            logger.info("SHUTDOWN COMPLETE")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
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
            ws_monitor_task = None  # <-- YENİ
    
            try:
                # Start health monitoring if enabled
                if self.health_monitor:
                    await self.health_monitor.start_monitoring()
        
                # ✅ YENİ: WebSocket monitor'ü başlat
                if self.ws_optimizer and self.ws_optimizer.is_initialized:
                    ws_monitor_task = asyncio.create_task(self._monitor_websocket_health())
                    logger.info("✓ WebSocket health monitoring started")
        
                # Start production trading loop
                await self.coordinator.run_production_loop(
                    mode=self.mode,
                    duration=duration,
                    continuous=self.infinite
                )
        
            except KeyboardInterrupt:
                logger.info("\n⚠ Keyboard interrupt received - initiating shutdown...")
                await self._shutdown()
        
            except Exception as e:
                logger.error(f"❌ Critical error in trading loop: {e}")
                if self.health_monitor:
                    self.health_monitor.record_error(str(e))
                await self._emergency_shutdown(f"Critical error: {e}")
    
            finally:
                # ✅ YENİ: WebSocket monitor'ü durdur
                if ws_monitor_task and not ws_monitor_task.done():
                    logger.info("Stopping WebSocket monitor...")
                    ws_monitor_task.cancel()
                    try:
                        await ws_monitor_task
                    except asyncio.CancelledError:
                        pass
                    logger.info("✓ WebSocket monitor stopped")
        
                # Stop health monitoring
                if self.health_monitor:
                    await self.health_monitor.stop_monitoring()
        
                # Shutdown WebSocket connections
                if self.ws_optimizer:
                    await self.ws_optimizer.shutdown()

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
            
            # If dry-run, stop here
            if self.dry_run:
                logger.info("\n✓ Dry run completed successfully - no trading started")
                return 0
            
            # Start trading loop
            await self._start_trading_loop(duration)
            
            return 0
            
        except Exception as e:
            logger.critical(f"❌ Fatal error: {e}")
            await self._emergency_shutdown(f"Fatal error: {e}")
            return 1
    
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
    
    # Create and run launcher
    launcher = LiveTradingLauncher(
        mode=mode, 
        dry_run=args.dry_run,
        infinite=args.infinite,
        auto_restart=args.auto_restart,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        debug_mode=args.debug
    )
    exit_code = await launcher.run(duration=args.duration)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())
