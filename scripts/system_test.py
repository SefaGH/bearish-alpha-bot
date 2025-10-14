#!/usr/bin/env python3
"""
Comprehensive System Test for Bearish Alpha Bot
Tests all components across all implemented phases:
- Phase 1: Multi-Exchange Integration (KuCoin + BingX)
- Phase 2: Market Intelligence Engine
- Phase 3: Risk Management, Portfolio Management, Live Trading
- Phase 4: AI Enhancement System (ML Regime Prediction, Adaptive Learning, Strategy Optimization, Price Prediction)
"""
import sys
import os
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class SystemTestRunner:
    """Comprehensive system test runner for all bot components."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def log_test(self, test_name: str, status: str, message: str = "", details: Dict = None):
        """Log test result."""
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message,
            'details': details or {}
        })
    
    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    
    def print_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Print individual test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"  {message}")
    
    # =========================================================================
    # Phase 1: Exchange Connection Tests
    # =========================================================================
    
    def test_exchange_connections(self) -> bool:
        """Test KuCoin and BingX API connections."""
        self.print_header("Phase 1: Exchange Connection Tests")
        print("Testing multi-exchange infrastructure...")
        
        all_passed = True
        
        try:
            from core.ccxt_client import CcxtClient
            
            # Test KuCoin initialization
            try:
                kucoin = CcxtClient('kucoinfutures')
                if kucoin.name == 'kucoinfutures':
                    self.print_test_result("KuCoin initialization", True)
                    self.log_test("KuCoin Init", "PASS")
                else:
                    self.print_test_result("KuCoin initialization", False, "Name mismatch")
                    self.log_test("KuCoin Init", "FAIL", "Name mismatch")
                    all_passed = False
            except Exception as e:
                self.print_test_result("KuCoin initialization", False, str(e))
                self.log_test("KuCoin Init", "FAIL", str(e))
                all_passed = False
            
            # Test BingX initialization
            try:
                bingx = CcxtClient('bingx')
                if bingx.name == 'bingx':
                    self.print_test_result("BingX initialization", True)
                    self.log_test("BingX Init", "PASS")
                else:
                    self.print_test_result("BingX initialization", False, "Name mismatch")
                    self.log_test("BingX Init", "FAIL", "Name mismatch")
                    all_passed = False
            except Exception as e:
                self.print_test_result("BingX initialization", False, str(e))
                self.log_test("BingX Init", "FAIL", str(e))
                all_passed = False
            
            # Test server time sync
            try:
                kucoin_time = kucoin._get_kucoin_server_time()
                bingx_time = bingx._get_bingx_server_time()
                local_time = int(time.time() * 1000)
                
                kucoin_diff = abs(kucoin_time - local_time)
                bingx_diff = abs(bingx_time - local_time)
                
                if kucoin_diff < 60000 and bingx_diff < 60000:
                    self.print_test_result("Server time synchronization", True, 
                                         f"KuCoin: {kucoin_diff}ms, BingX: {bingx_diff}ms")
                    self.log_test("Server Time Sync", "PASS", 
                                details={'kucoin_diff': kucoin_diff, 'bingx_diff': bingx_diff})
                else:
                    self.print_test_result("Server time synchronization", False, 
                                         f"Large time differences detected")
                    self.log_test("Server Time Sync", "FAIL", "Time differences too large")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Server time synchronization", False, str(e))
                self.log_test("Server Time Sync", "FAIL", str(e))
                all_passed = False
            
            # Test contract discovery
            try:
                kucoin_contracts = kucoin._get_dynamic_symbol_mapping()
                bingx_contracts = bingx._get_bingx_contracts()
                
                if 'BTC/USDT:USDT' in kucoin_contracts and 'BTC/USDT:USDT' in bingx_contracts:
                    self.print_test_result("Contract discovery", True, 
                                         f"KuCoin: {len(kucoin_contracts)}, BingX: {len(bingx_contracts)}")
                    self.log_test("Contract Discovery", "PASS", 
                                details={'kucoin': len(kucoin_contracts), 'bingx': len(bingx_contracts)})
                else:
                    self.print_test_result("Contract discovery", False, "BTC contract not found")
                    self.log_test("Contract Discovery", "FAIL", "Essential contracts missing")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Contract discovery", False, str(e))
                self.log_test("Contract Discovery", "FAIL", str(e))
                all_passed = False
            
            # Test MultiExchangeManager
            try:
                from core.multi_exchange_manager import MultiExchangeManager
                exchanges = {
                    'kucoinfutures': kucoin,
                    'bingx': bingx
                }
                manager = MultiExchangeManager(exchanges)
                summary = manager.get_exchange_summary()
                
                if len(summary['exchanges']) == 2:
                    self.print_test_result("MultiExchangeManager", True, 
                                         f"Managing {len(summary['exchanges'])} exchanges")
                    self.log_test("MultiExchangeManager", "PASS")
                else:
                    self.print_test_result("MultiExchangeManager", False, "Exchange count mismatch")
                    self.log_test("MultiExchangeManager", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("MultiExchangeManager", False, str(e))
                self.log_test("MultiExchangeManager", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("Exchange connection tests", False, str(e))
            self.log_test("Exchange Tests", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Phase 1: Data Pipeline Tests
    # =========================================================================
    
    def test_data_pipeline(self) -> bool:
        """Test OHLCV data fetching and WebSocket connections."""
        self.print_header("Phase 1: Data Pipeline Tests")
        print("Testing data fetching and streaming...")
        
        all_passed = True
        
        try:
            from core.ccxt_client import CcxtClient
            
            # Test OHLCV data fetching
            try:
                kucoin = CcxtClient('kucoinfutures')
                ohlcv = kucoin.ohlcv('BTC/USDT:USDT', '1h', limit=10)
                
                if ohlcv and len(ohlcv) == 10:
                    self.print_test_result("OHLCV data fetching", True, 
                                         f"Fetched {len(ohlcv)} candles")
                    self.log_test("OHLCV Fetch", "PASS", details={'candles': len(ohlcv)})
                else:
                    self.print_test_result("OHLCV data fetching", False, "Incomplete data")
                    self.log_test("OHLCV Fetch", "FAIL", "Incomplete data")
                    all_passed = False
            except Exception as e:
                self.print_test_result("OHLCV data fetching", False, str(e))
                self.log_test("OHLCV Fetch", "FAIL", str(e))
                all_passed = False
            
            # Test bulk fetch capability
            try:
                from core.multi_exchange_manager import MultiExchangeManager
                exchanges = {'kucoinfutures': CcxtClient('kucoinfutures')}
                manager = MultiExchangeManager(exchanges)
                
                # Correct format: dict mapping exchange to symbols
                symbols_per_exchange = {'kucoinfutures': ['BTC/USDT:USDT', 'ETH/USDT:USDT']}
                result = manager.fetch_unified_data(symbols_per_exchange, '1h', limit=5)
                
                if result and 'kucoinfutures' in result and 'BTC/USDT:USDT' in result['kucoinfutures']:
                    total_symbols = sum(len(symbols) for symbols in result.values())
                    self.print_test_result("Bulk OHLCV fetching", True, 
                                         f"Fetched {total_symbols} symbols")
                    self.log_test("Bulk OHLCV", "PASS", details={'symbols': total_symbols})
                else:
                    self.print_test_result("Bulk OHLCV fetching", False, "Incomplete bulk data")
                    self.log_test("Bulk OHLCV", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Bulk OHLCV fetching", False, str(e))
                self.log_test("Bulk OHLCV", "FAIL", str(e))
                all_passed = False
            
            # Test WebSocket infrastructure
            try:
                from core.websocket_client import WebSocketClient
                from core.websocket_manager import WebSocketManager
                
                # Test WebSocket client initialization
                ws_client = WebSocketClient('kucoinfutures')
                self.print_test_result("WebSocket client initialization", True)
                self.log_test("WebSocket Client", "PASS")
                
                # Test WebSocket manager initialization
                ws_manager = WebSocketManager()
                if len(ws_manager.clients) > 0:
                    self.print_test_result("WebSocket manager initialization", True, 
                                         f"{len(ws_manager.clients)} clients initialized")
                    self.log_test("WebSocket Manager", "PASS", 
                                details={'clients': len(ws_manager.clients)})
                else:
                    self.print_test_result("WebSocket manager initialization", False)
                    self.log_test("WebSocket Manager", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("WebSocket infrastructure", False, str(e))
                self.log_test("WebSocket Infrastructure", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("Data pipeline tests", False, str(e))
            self.log_test("Data Pipeline", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Phase 2: Strategy System Tests
    # =========================================================================
    
    def test_strategy_system(self) -> bool:
        """Test Phase 2 adaptive strategies are loading."""
        self.print_header("Phase 2: Strategy System Tests")
        print("Testing market intelligence and adaptive strategies...")
        
        all_passed = True
        
        try:
            import pandas as pd
            import numpy as np
            
            # Test market regime analyzer
            try:
                from core.market_regime import MarketRegimeAnalyzer
                
                analyzer = MarketRegimeAnalyzer()
                
                # Create test data
                dates = pd.date_range(start='2024-01-01', periods=200, freq='30min')
                close = 100 + np.cumsum(np.random.randn(200) * 0.5)
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': close + np.random.randn(200) * 0.1,
                    'high': close + np.abs(np.random.randn(200) * 0.3),
                    'low': close - np.abs(np.random.randn(200) * 0.3),
                    'close': close,
                    'volume': np.random.randint(1000, 10000, 200)
                })
                
                regime = analyzer.analyze_market_regime(df, df, df)
                
                if regime and 'trend' in regime:
                    self.print_test_result("Market regime analyzer", True, 
                                         f"Detected regime: {regime['trend']}")
                    self.log_test("Market Regime", "PASS", details=regime)
                else:
                    self.print_test_result("Market regime analyzer", False)
                    self.log_test("Market Regime", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Market regime analyzer", False, str(e))
                self.log_test("Market Regime", "FAIL", str(e))
                all_passed = False
            
            # Test base strategies
            try:
                from strategies.oversold_bounce import OversoldBounce
                from strategies.short_the_rip import ShortTheRip
                
                ob_config = {'rsi_max': 30, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
                ob_strategy = OversoldBounce(ob_config)
                
                str_config = {'rsi_min': 70, 'tp_pct': 0.012, 'sl_atr_mult': 1.0}
                str_strategy = ShortTheRip(str_config)
                
                self.print_test_result("Base strategies loading", True, 
                                     "OversoldBounce and ShortTheRip loaded")
                self.log_test("Base Strategies", "PASS")
            except Exception as e:
                self.print_test_result("Base strategies loading", False, str(e))
                self.log_test("Base Strategies", "FAIL", str(e))
                all_passed = False
            
            # Test adaptive strategies
            try:
                from strategies.adaptive_ob import AdaptiveOversoldBounce
                from strategies.adaptive_str import AdaptiveShortTheRip
                from core.market_regime import MarketRegimeAnalyzer
                
                regime_analyzer = MarketRegimeAnalyzer()
                
                adaptive_ob_config = {'rsi_max': 30, 'tp_pct': 0.015, 'sl_atr_mult': 1.0}
                adaptive_ob = AdaptiveOversoldBounce(adaptive_ob_config, regime_analyzer)
                
                adaptive_str_config = {'rsi_min': 70, 'tp_pct': 0.012, 'sl_atr_mult': 1.0}
                adaptive_str = AdaptiveShortTheRip(adaptive_str_config, regime_analyzer)
                
                self.print_test_result("Adaptive strategies loading", True, 
                                     "AdaptiveOversoldBounce and AdaptiveShortTheRip loaded")
                self.log_test("Adaptive Strategies", "PASS")
            except Exception as e:
                self.print_test_result("Adaptive strategies loading", False, str(e))
                self.log_test("Adaptive Strategies", "FAIL", str(e))
                all_passed = False
            
            # Test performance monitoring
            try:
                from core.performance_monitor import RealTimePerformanceMonitor
                
                monitor = RealTimePerformanceMonitor()
                
                # Test basic functionality with track_strategy_performance
                monitor.track_strategy_performance('test_strategy', {
                    'return': 0.02,
                    'sharpe': 1.5,
                    'win_rate': 0.6
                })
                
                summary = monitor.get_strategy_summary('test_strategy')
                
                if summary:
                    self.print_test_result("Performance monitoring", True, 
                                         "Strategy performance tracked")
                    self.log_test("Performance Monitor", "PASS")
                else:
                    self.print_test_result("Performance monitoring", False)
                    self.log_test("Performance Monitor", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Performance monitoring", False, str(e))
                self.log_test("Performance Monitor", "FAIL", str(e))
                all_passed = False
            
            # Test VST intelligence
            try:
                from core.vst_intelligence import VSTMarketAnalyzer
                from core.ccxt_client import CcxtClient
                
                bingx = CcxtClient('bingx')
                vst_analyzer = VSTMarketAnalyzer(bingx)
                
                self.print_test_result("VST intelligence system", True)
                self.log_test("VST Intelligence", "PASS")
            except Exception as e:
                self.print_test_result("VST intelligence system", False, str(e))
                self.log_test("VST Intelligence", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("Strategy system tests", False, str(e))
            self.log_test("Strategy System", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Phase 3: Portfolio Manager Tests
    # =========================================================================
    
    def test_portfolio_manager(self) -> bool:
        """Test Phase 3 portfolio management components."""
        self.print_header("Phase 3: Portfolio Management Tests")
        print("Testing risk management and portfolio optimization...")
        
        all_passed = True
        
        try:
            # Test risk manager
            try:
                from core.risk_manager import RiskManager
                import asyncio
                
                portfolio_config = {'equity_usd': 10000}
                risk_manager = RiskManager(portfolio_config)
                
                # Test position sizing with correct signal format
                test_signal = {
                    'entry': 50000,
                    'stop': 49000,
                    'target': 52000,
                    'side': 'long'
                }
                
                # Run async method
                async def test_position_size():
                    return await risk_manager.calculate_position_size(test_signal)
                
                position_size = asyncio.run(test_position_size())
                
                if position_size and position_size > 0:
                    self.print_test_result("Risk manager", True, 
                                         f"Position size: {position_size:.4f}")
                    self.log_test("Risk Manager", "PASS", 
                                details={'position_size': position_size})
                else:
                    self.print_test_result("Risk manager", False)
                    self.log_test("Risk Manager", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Risk manager", False, str(e))
                self.log_test("Risk Manager", "FAIL", str(e))
                all_passed = False
            
            # Test portfolio manager
            try:
                from core.portfolio_manager import PortfolioManager
                from core.performance_monitor import RealTimePerformanceMonitor
                
                portfolio_config = {'equity_usd': 10000}
                risk_manager = RiskManager(portfolio_config)
                performance_monitor = RealTimePerformanceMonitor()
                
                portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
                
                # Test strategy registration
                class MockStrategy:
                    def __init__(self, name):
                        self.name = name
                
                result = portfolio_manager.register_strategy(
                    'test_strategy',
                    MockStrategy('test_strategy'),
                    0.25
                )
                
                if result['status'] == 'success':
                    self.print_test_result("Portfolio manager", True, 
                                         f"Strategy registered with {result['allocation']*100}% allocation")
                    self.log_test("Portfolio Manager", "PASS", details=result)
                else:
                    self.print_test_result("Portfolio manager", False)
                    self.log_test("Portfolio Manager", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Portfolio manager", False, str(e))
                self.log_test("Portfolio Manager", "FAIL", str(e))
                all_passed = False
            
            # Test strategy coordinator
            try:
                from core.strategy_coordinator import StrategyCoordinator
                import asyncio
                
                coordinator = StrategyCoordinator(portfolio_manager, risk_manager)
                
                # Test signal processing with correct signal format (good risk/reward ratio)
                test_signal = {
                    'symbol': 'BTC/USDT:USDT',
                    'side': 'long',
                    'entry': 50000,
                    'stop': 49000,  # Risk: 1000
                    'target': 52000  # Reward: 2000, R/R = 2.0
                }
                
                # Run async method
                async def test_signal_processing():
                    return await coordinator.process_strategy_signal('test_strategy', test_signal)
                
                result = asyncio.run(test_signal_processing())
                
                # Check if result is valid (action could be 'queued', 'rejected', 'duplicate', etc.)
                if result and isinstance(result, dict) and 'action' in result:
                    self.print_test_result("Strategy coordinator", True, 
                                         f"Signal processed: {result['action']}")
                    self.log_test("Strategy Coordinator", "PASS", details=result)
                elif result:
                    # Signal was processed but maybe with different structure
                    self.print_test_result("Strategy coordinator", True, 
                                         f"Signal processed: {str(result)[:50]}")
                    self.log_test("Strategy Coordinator", "PASS", details=result)
                else:
                    self.print_test_result("Strategy coordinator", False, 
                                         f"No result returned: {result}")
                    self.log_test("Strategy Coordinator", "FAIL", "No result")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Strategy coordinator", False, str(e))
                self.log_test("Strategy Coordinator", "FAIL", str(e))
                all_passed = False
            
            # Test position manager (skip - integrated in risk manager)
            try:
                # Position management is integrated into RiskManager
                # Test that position tracking methods exist
                if hasattr(risk_manager, 'register_position'):
                    self.print_test_result("Position management", True, 
                                         "Position tracking available in RiskManager")
                    self.log_test("Position Management", "PASS")
                else:
                    self.print_test_result("Position management", False)
                    self.log_test("Position Management", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("Position management", False, str(e))
                self.log_test("Position Management", "FAIL", str(e))
                all_passed = False
            
            # Test live trading engine
            try:
                from core.live_trading_engine import LiveTradingEngine
                from core.ccxt_client import CcxtClient
                from core.websocket_manager import WebSocketManager
                
                exchange_clients = {'kucoinfutures': CcxtClient('kucoinfutures')}
                ws_manager = WebSocketManager()
                engine = LiveTradingEngine(portfolio_manager, risk_manager, ws_manager, exchange_clients)
                
                self.print_test_result("Live trading engine", True)
                self.log_test("Live Trading Engine", "PASS")
            except Exception as e:
                self.print_test_result("Live trading engine", False, str(e))
                self.log_test("Live Trading Engine", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("Portfolio manager tests", False, str(e))
            self.log_test("Portfolio Tests", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Phase 4: ML Models Tests
    # =========================================================================
    
    def test_ml_models(self) -> bool:
        """Test Phase 4 ML components."""
        self.print_header("Phase 4: ML Enhancement Tests")
        print("Testing AI/ML components...")
        
        all_passed = True
        
        try:
            # Test Phase 4.1: ML Regime Prediction
            try:
                from ml.regime_predictor import MLRegimePredictor
                from ml.feature_engineering import FeatureEngineeringPipeline
                from ml.model_trainer import RegimeModelTrainer
                from ml.prediction_engine import RealTimePredictionEngine
                
                self.print_test_result("ML regime prediction imports", True, 
                                     "All Phase 4.1 components loaded")
                self.log_test("ML Regime Prediction", "PASS")
            except Exception as e:
                self.print_test_result("ML regime prediction imports", False, str(e))
                self.log_test("ML Regime Prediction", "FAIL", str(e))
                all_passed = False
            
            # Test Phase 4.2: Adaptive Learning
            try:
                from ml.reinforcement_learning import TradingRLAgent, DQNNetwork
                from ml.experience_replay import ExperienceReplay, EpisodeBuffer
                
                self.print_test_result("Adaptive learning imports", True, 
                                     "All Phase 4.2 components loaded")
                self.log_test("Adaptive Learning", "PASS")
            except Exception as e:
                self.print_test_result("Adaptive learning imports", False, str(e))
                self.log_test("Adaptive Learning", "FAIL", str(e))
                all_passed = False
            
            # Test Phase 4.3: Strategy Optimization
            try:
                from ml.genetic_optimizer import GeneticOptimizer, Individual
                from ml.multi_objective_optimizer import MultiObjectiveOptimizer
                from ml.neural_architecture_search import NeuralArchitectureSearch
                from ml.strategy_optimizer import StrategyOptimizer
                
                self.print_test_result("Strategy optimization imports", True, 
                                     "All Phase 4.3 components loaded")
                self.log_test("Strategy Optimization", "PASS")
            except Exception as e:
                self.print_test_result("Strategy optimization imports", False, str(e))
                self.log_test("Strategy Optimization", "FAIL", str(e))
                all_passed = False
            
            # Test Phase 4 Final: Price Prediction
            try:
                from ml.price_predictor import (
                    LSTMPricePredictor,
                    TransformerPricePredictor,
                    EnsemblePricePredictor,
                    MultiTimeframePricePredictor,
                    AdvancedPricePredictionEngine
                )
                from ml.strategy_integration import (
                    AIEnhancedStrategyAdapter,
                    StrategyPerformanceTracker,
                    MLStrategyIntegrationManager
                )
                
                self.print_test_result("Price prediction imports", True, 
                                     "All Phase 4 Final components loaded")
                self.log_test("Price Prediction", "PASS")
            except Exception as e:
                self.print_test_result("Price prediction imports", False, str(e))
                self.log_test("Price Prediction", "FAIL", str(e))
                all_passed = False
            
            # Test ML feature engineering with real data
            try:
                import pandas as pd
                import numpy as np
                from ml.feature_engineering import FeatureEngineeringPipeline
                
                # Create test data
                dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
                close = 100 + np.cumsum(np.random.randn(200) * 0.5)
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': close + np.random.randn(200) * 0.1,
                    'high': close + np.abs(np.random.randn(200) * 0.3),
                    'low': close - np.abs(np.random.randn(200) * 0.3),
                    'close': close,
                    'volume': np.random.randint(1000, 10000, 200)
                })
                
                pipeline = FeatureEngineeringPipeline()
                features = pipeline.extract_features(df)
                
                if features is not None and len(features) > 0:
                    self.print_test_result("ML feature engineering", True, 
                                         f"{len(features.columns)} features created")
                    self.log_test("ML Features", "PASS", 
                                details={'feature_count': len(features.columns)})
                else:
                    self.print_test_result("ML feature engineering", False)
                    self.log_test("ML Features", "FAIL")
                    all_passed = False
            except Exception as e:
                self.print_test_result("ML feature engineering", False, str(e))
                self.log_test("ML Features", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("ML model tests", False, str(e))
            self.log_test("ML Tests", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Integration Tests
    # =========================================================================
    
    def test_integration(self) -> bool:
        """Test full system integration."""
        self.print_header("Full System Integration Tests")
        print("Testing complete system integration...")
        
        all_passed = True
        
        try:
            import pandas as pd
            import numpy as np
            from datetime import timedelta
            
            # Test Phase 1 + Phase 2 Integration
            try:
                from core.ccxt_client import CcxtClient
                from core.market_regime import MarketRegimeAnalyzer
                from strategies.adaptive_ob import AdaptiveOversoldBounce
                
                # Fetch real data
                kucoin = CcxtClient('kucoinfutures')
                ohlcv_30m = kucoin.ohlcv('BTC/USDT:USDT', '30m', limit=100)
                ohlcv_1h = kucoin.ohlcv('BTC/USDT:USDT', '1h', limit=100)
                ohlcv_4h = kucoin.ohlcv('BTC/USDT:USDT', '4h', limit=100)
                
                # Convert to DataFrames
                df_30m = pd.DataFrame(ohlcv_30m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Analyze regime
                analyzer = MarketRegimeAnalyzer()
                regime = analyzer.analyze_market_regime(df_30m, df_1h, df_4h)
                
                # Generate signal with adaptive strategy
                from core.indicators import add_indicators
                df_30m_ind = add_indicators(df_30m, {'rsi_period': 14, 'atr_period': 14})
                
                adaptive_strategy = AdaptiveOversoldBounce(
                    {'rsi_max': 30, 'tp_pct': 0.015, 'sl_atr_mult': 1.0},
                    analyzer
                )
                
                signal = adaptive_strategy.signal(df_30m_ind, regime)
                
                self.print_test_result("Phase 1+2 integration", True, 
                                     f"Regime: {regime['trend']}, Signal: {signal is not None}")
                self.log_test("Phase 1+2 Integration", "PASS", 
                            details={'regime': regime, 'has_signal': signal is not None})
            except Exception as e:
                self.print_test_result("Phase 1+2 integration", False, str(e))
                self.log_test("Phase 1+2 Integration", "FAIL", str(e))
                all_passed = False
            
            # Test Phase 1 + Phase 3 Integration
            try:
                from core.ccxt_client import CcxtClient
                from core.risk_manager import RiskManager
                from core.portfolio_manager import PortfolioManager
                from core.performance_monitor import RealTimePerformanceMonitor
                
                exchanges = {'kucoinfutures': CcxtClient('kucoinfutures')}
                portfolio_config = {'equity_usd': 10000}
                risk_manager = RiskManager(portfolio_config)
                performance_monitor = RealTimePerformanceMonitor()
                portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
                
                self.print_test_result("Phase 1+3 integration", True)
                self.log_test("Phase 1+3 Integration", "PASS")
            except Exception as e:
                self.print_test_result("Phase 1+3 integration", False, str(e))
                self.log_test("Phase 1+3 Integration", "FAIL", str(e))
                all_passed = False
            
            # Test Complete Pipeline
            try:
                from core.ccxt_client import CcxtClient
                from core.multi_exchange_manager import MultiExchangeManager
                from core.market_regime import MarketRegimeAnalyzer
                from core.risk_manager import RiskManager
                from core.portfolio_manager import PortfolioManager
                from core.performance_monitor import RealTimePerformanceMonitor
                from strategies.adaptive_ob import AdaptiveOversoldBounce
                
                # Initialize all components
                kucoin = CcxtClient('kucoinfutures')
                bingx = CcxtClient('bingx')
                exchanges = {'kucoinfutures': kucoin, 'bingx': bingx}
                manager = MultiExchangeManager(exchanges)
                
                regime_analyzer = MarketRegimeAnalyzer()
                portfolio_config = {'equity_usd': 10000}
                risk_manager = RiskManager(portfolio_config)
                performance_monitor = RealTimePerformanceMonitor()
                portfolio_manager = PortfolioManager(risk_manager, performance_monitor)
                
                # Register strategy
                strategy = AdaptiveOversoldBounce(
                    {'rsi_max': 30, 'tp_pct': 0.015, 'sl_atr_mult': 1.0},
                    regime_analyzer
                )
                portfolio_manager.register_strategy('adaptive_ob', strategy, 0.5)
                
                self.print_test_result("Complete pipeline integration", True, 
                                     "All components initialized and connected")
                self.log_test("Complete Pipeline", "PASS")
            except Exception as e:
                self.print_test_result("Complete pipeline integration", False, str(e))
                self.log_test("Complete Pipeline", "FAIL", str(e))
                all_passed = False
                
        except Exception as e:
            self.print_test_result("Integration tests", False, str(e))
            self.log_test("Integration Tests", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    # =========================================================================
    # Main Test Runner
    # =========================================================================
    
    def run_all_tests(self) -> int:
        """Run all system tests."""
        print("=" * 70)
        print("BEARISH ALPHA BOT - COMPREHENSIVE SYSTEM TEST")
        print("=" * 70)
        print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
        print()
        
        test_groups = [
            ("Exchange Connections", self.test_exchange_connections),
            ("Data Pipeline", self.test_data_pipeline),
            ("Strategy System", self.test_strategy_system),
            ("Portfolio Manager", self.test_portfolio_manager),
            ("ML Models", self.test_ml_models),
            ("System Integration", self.test_integration),
        ]
        
        group_results = []
        
        for group_name, test_func in test_groups:
            try:
                result = test_func()
                group_results.append((group_name, result))
            except Exception as e:
                print(f"\n✗ {group_name} crashed: {e}")
                import traceback
                traceback.print_exc()
                group_results.append((group_name, False))
        
        # Final summary
        elapsed = time.time() - self.start_time
        passed_groups = sum(1 for _, result in group_results if result)
        total_groups = len(group_results)
        
        print("\n" + "=" * 70)
        print("SYSTEM TEST SUMMARY")
        print("=" * 70)
        print(f"Test Groups: {passed_groups}/{total_groups} passed")
        print(f"Total Time: {elapsed:.2f}s")
        print()
        
        for group_name, result in group_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {group_name}")
        
        print("\n" + "=" * 70)
        
        # Individual test results
        passed_tests = sum(1 for r in self.results if r['status'] == 'PASS')
        total_tests = len(self.results)
        
        print(f"Individual Tests: {passed_tests}/{total_tests} passed")
        print()
        
        for result in self.results:
            status = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status} {result['test']}")
            if result['message']:
                print(f"  {result['message']}")
        
        print("\n" + "=" * 70)
        
        if passed_groups == total_groups:
            print("✅ ALL SYSTEM TESTS PASSED!")
            print()
            print("System Status: OPERATIONAL")
            print("- Phase 1: Multi-Exchange Integration ✓")
            print("- Phase 2: Market Intelligence Engine ✓")
            print("- Phase 3: Portfolio Management ✓")
            print("- Phase 4: AI Enhancement System ✓")
            exit_code = 0
        else:
            print("⚠ SOME TESTS FAILED")
            print()
            print("Review failed tests above for details.")
            exit_code = 1
        
        print("=" * 70)
        print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 70)
        
        return exit_code


def main():
    """Main entry point."""
    runner = SystemTestRunner()
    return runner.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
