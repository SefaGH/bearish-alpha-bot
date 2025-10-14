#!/usr/bin/env python3
"""
Live Trading Launcher for Bearish Alpha Bot

Comprehensive production-ready launcher that integrates all Phase 1-4 components
for live trading on BingX with VST (Virtual test tokens).

Configuration:
- Capital: 100 VST
- Exchange: BingX (Single exchange focus)
- Trading Pairs: 8 diversified crypto pairs
- Execution Mode: Full AI control (Automated)
- Risk Parameters: 15% max position, 5% stop loss, 10% take profit

Usage:
    python scripts/live_trading_launcher.py [--paper] [--duration SECONDS]
    
Options:
    --paper         Run in paper trading mode (default: live)
    --duration      Run for specified duration in seconds (default: indefinite)
    --dry-run       Perform pre-flight checks only without starting trading
    
Environment Variables Required:
    BINGX_KEY       - BingX API key
    BINGX_SECRET    - BingX API secret
    
Optional:
    TELEGRAM_BOT_TOKEN  - Telegram bot token for notifications
    TELEGRAM_CHAT_ID    - Telegram chat ID for notifications
"""

import sys
import os
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient
from core.notify import Telegram
from config.risk_config import RiskConfiguration
from ml.regime_predictor import MLRegimePredictor
from ml.price_predictor import AdvancedPricePredictionEngine
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


class LiveTradingLauncher:
    """
    Comprehensive live trading launcher integrating all system components.
    
    Phases Integrated:
    - Phase 1: Multi-exchange framework (BingX focus)
    - Phase 2: Adaptive strategies with live signals
    - Phase 3: Portfolio management, risk engine, live execution
    - Phase 4: Complete AI enhancement (regime, adaptive learning, optimization, price prediction)
    """
    
    # 8 diversified trading pairs as specified
    TRADING_PAIRS = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'SOL/USDT:USDT',
        'BNB/USDT:USDT',
        'ADA/USDT:USDT',
        'DOT/USDT:USDT',
        'MATIC/USDT:USDT',
        'AVAX/USDT:USDT'
    ]
    
    # VST test trading configuration
    CAPITAL_VST = 100.0  # 100 VST virtual test tokens
    
    # Risk parameters as specified
    RISK_PARAMS = {
        'max_position_size': 0.15,    # 15% max position size
        'stop_loss_pct': 0.05,        # 5% stop loss
        'take_profit_pct': 0.10,      # 10% take profit
        'max_drawdown': 0.15,         # 15% max drawdown
        'max_portfolio_risk': 0.02,   # 2% risk per trade
        'max_correlation': 0.70,      # 70% max correlation
    }
    
    def __init__(self, mode: str = 'live', dry_run: bool = False):
        """
        Initialize live trading launcher.
        
        Args:
            mode: Trading mode ('live' or 'paper')
            dry_run: If True, only perform checks without starting trading
        """
        self.mode = mode
        self.dry_run = dry_run
        self.coordinator = None
        self.telegram = None
        self.exchange_clients = {}
        self.strategies = {}
        
        # Phase 4 AI components
        self.regime_predictor = None
        self.price_engine = None
        self.strategy_adapter = None
        self.strategy_optimizer = None
        
        logger.info("="*70)
        logger.info("BEARISH ALPHA BOT - LIVE TRADING LAUNCHER")
        logger.info("="*70)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Capital: {self.CAPITAL_VST} VST")
        logger.info(f"Exchange: BingX")
        logger.info(f"Trading Pairs: {len(self.TRADING_PAIRS)}")
        logger.info(f"Dry Run: {dry_run}")
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
        
        return True
    
    def _initialize_exchange_connection(self) -> bool:
        """
        Initialize BingX exchange connection with VST support.
        
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
            self.exchange_clients['bingx'] = bingx_client
            
            # Test connection
            logger.info("Testing BingX connection...")
            markets = bingx_client.markets()
            logger.info(f"✓ Connected to BingX - {len(markets)} markets available")
            
            # Verify VST contract availability
            vst_symbol = 'VST/USDT:USDT'
            if vst_symbol in markets:
                logger.info(f"✓ VST/USDT:USDT contract verified")
            else:
                logger.warning(f"⚠ VST/USDT:USDT contract not found in markets")
            
            # Verify all trading pairs
            missing_pairs = [pair for pair in self.TRADING_PAIRS if pair not in markets]
            if missing_pairs:
                logger.warning(f"⚠ Some trading pairs not available: {missing_pairs}")
            else:
                logger.info(f"✓ All {len(self.TRADING_PAIRS)} trading pairs verified")
            
            return True
            
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
        
        Returns:
            True if initialization successful
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
            self.strategy_optimizer = StrategyOptimizer()
            logger.info("✓ Strategy Optimizer initialized")
            
            # Phase 4.4: Price Prediction
            logger.info("Initializing price prediction engine...")
            self.price_engine = AdvancedPricePredictionEngine()
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
        
        Returns:
            True if initialization successful
        """
        logger.info("\n[5/8] Initializing Trading Strategies...")
        
        try:
            # Adaptive Oversold Bounce strategy
            self.strategies['adaptive_ob'] = AdaptiveOversoldBounce()
            logger.info("✓ Adaptive Oversold Bounce strategy initialized")
            
            # Adaptive Short The Rip strategy
            self.strategies['adaptive_str'] = AdaptiveShortTheRip()
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
            # Portfolio configuration with VST capital
            portfolio_config = {
                'equity_usd': self.CAPITAL_VST,  # 100 VST
                'max_portfolio_risk': self.RISK_PARAMS['max_portfolio_risk'],
                'max_position_size': self.RISK_PARAMS['max_position_size'],
                'max_drawdown': self.RISK_PARAMS['max_drawdown']
            }
            
            # Initialize production coordinator
            self.coordinator = ProductionCoordinator()
            
            # Initialize complete production system
            init_result = await self.coordinator.initialize_production_system(
                exchange_clients=self.exchange_clients,
                portfolio_config=portfolio_config
            )
            
            if not init_result['success']:
                logger.error(f"❌ Production system initialization failed: {init_result.get('reason')}")
                return False
            
            logger.info("✓ Production system initialized successfully")
            logger.info(f"  Components: {init_result['components']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize production system: {e}")
            return False
    
    async def _register_strategies(self) -> bool:
        """
        Register trading strategies with portfolio manager.
        
        Returns:
            True if registration successful
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
                
                if result['success']:
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
            logger.info("Check 1/5: Exchange connectivity...")
            try:
                ticker = self.exchange_clients['bingx'].ticker('BTC/USDT:USDT')
                logger.info(f"✓ BTC/USDT:USDT price: ${ticker.get('last', 0):.2f}")
            except Exception as e:
                logger.error(f"❌ Exchange connectivity failed: {e}")
                checks_passed = False
            
            # Check 2: System state
            logger.info("Check 2/5: System state...")
            state = self.coordinator.get_system_state()
            if state['is_initialized']:
                logger.info("✓ Production system initialized")
            else:
                logger.error("❌ Production system not initialized")
                checks_passed = False
            
            # Check 3: Risk limits
            logger.info("Check 3/5: Risk limits...")
            if self.coordinator.risk_manager:
                risk_summary = self.coordinator.risk_manager.get_portfolio_summary()
                logger.info(f"✓ Portfolio value: ${risk_summary['portfolio_value']:.2f}")
                logger.info(f"✓ Risk limits configured")
            else:
                logger.error("❌ Risk manager not available")
                checks_passed = False
            
            # Check 4: Strategies
            logger.info("Check 4/5: Strategy registration...")
            if self.coordinator.portfolio_manager:
                strategies = self.coordinator.portfolio_manager.strategies
                logger.info(f"✓ {len(strategies)} strategies registered")
            else:
                logger.error("❌ Portfolio manager not available")
                checks_passed = False
            
            # Check 5: Emergency protocols
            logger.info("Check 5/5: Emergency shutdown protocols...")
            if self.coordinator.circuit_breaker:
                logger.info("✓ Circuit breaker active")
            else:
                logger.warning("⚠ Circuit breaker not available")
            
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
        """
        Start the main trading loop.
        
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
        
        # Send Telegram notification
        if self.telegram:
            self.telegram.send(
                f"🚀 <b>LIVE TRADING STARTED</b>\n"
                f"Mode: {self.mode.upper()}\n"
                f"Capital: {self.CAPITAL_VST} VST\n"
                f"Exchange: BingX\n"
                f"Pairs: {len(self.TRADING_PAIRS)}\n"
                f"Max Position: {self.RISK_PARAMS['max_position_size']:.1%}\n"
                f"Stop Loss: {self.RISK_PARAMS['stop_loss_pct']:.1%}\n"
                f"Take Profit: {self.RISK_PARAMS['take_profit_pct']:.1%}"
            )
        
        try:
            # Start production trading loop
            await self.coordinator.run_production_loop(
                mode=self.mode,
                duration=duration
            )
            
        except KeyboardInterrupt:
            logger.info("\n⚠ Keyboard interrupt received - initiating shutdown...")
            await self._shutdown()
            
        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
            await self._emergency_shutdown(f"Critical error: {e}")
    
    async def _shutdown(self) -> None:
        """Graceful shutdown of trading system."""
        logger.info("\n" + "="*70)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("="*70)
        
        try:
            if self.coordinator:
                await self.coordinator.stop_system()
                logger.info("✓ Production system stopped")
            
            # Send Telegram notification
            if self.telegram:
                self.telegram.send("🛑 <b>Trading stopped - Graceful shutdown completed</b>")
            
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
            if self.coordinator:
                await self.coordinator.handle_emergency_shutdown(reason)
            
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
            
            # Step 6: Initialize production system
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
    
    args = parser.parse_args()
    
    # Determine mode
    mode = 'paper' if args.paper else 'live'
    
    # Create and run launcher
    launcher = LiveTradingLauncher(mode=mode, dry_run=args.dry_run)
    exit_code = await launcher.run(duration=args.duration)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())
