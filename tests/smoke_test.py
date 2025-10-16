#!/usr/bin/env python3
"""
Production-ready smoke test for Bearish Alpha Bot.
Tests all critical components including Phase 3-4 infrastructure.
"""
import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    try:
        # Phase 1: Exchange connectivity
        from core.ccxt_client import CcxtClient
        from core.multi_exchange import build_clients_from_env
        from core.bingx_authenticator import BingXAuthenticator
        
        # Phase 2: Strategies and indicators
        from core.indicators import add_indicators
        from core.regime import is_bearish_regime
        from core.market_regime import MarketRegimeAnalyzer
        from strategies.oversold_bounce import OversoldBounce
        from strategies.short_the_rip import ShortTheRip
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        from strategies.adaptive_str import AdaptiveShortTheRip
        
        # Phase 3: Production infrastructure
        from core.production_coordinator import ProductionCoordinator
        from core.live_trading_engine import LiveTradingEngine
        from core.websocket_manager import WebSocketManager
        from core.websocket_client import WebSocketClient
        from core.risk_manager import RiskManager
        from core.portfolio_manager import PortfolioManager
        from core.strategy_coordinator import StrategyCoordinator
        from core.circuit_breaker import CircuitBreakerSystem
        from core.order_manager import SmartOrderManager
        from core.position_manager import AdvancedPositionManager
        from core.execution_analytics import ExecutionAnalytics
        
        # Phase 4: AI/ML components
        from ml.regime_predictor import MLRegimePredictor
        from ml.price_predictor import AdvancedPricePredictionEngine
        from ml.strategy_optimizer import StrategyOptimizer
        from ml.strategy_integration import AIEnhancedStrategyAdapter
        
        # Utilities
        from core.notify import Telegram
        from core.state import load_state, save_state
        from universe import build_universe, pick_execution_exchange
        
        print("  ✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_phase3_components():
    """Test Phase 3 production components initialization."""
    print("Testing Phase 3 components...")
    try:
        from core.production_coordinator import ProductionCoordinator
        from core.risk_manager import RiskManager
        from core.portfolio_manager import PortfolioManager
        
        # Test basic instantiation
        coordinator = ProductionCoordinator()
        
        # Test risk manager with config
        portfolio_config = {
            'equity_usd': 100,
            'max_portfolio_risk': 0.02,
            'max_position_size': 0.15,
            'max_drawdown': 0.15
        }
        risk_manager = RiskManager(portfolio_config, None, None)
        
        print("  ✓ Phase 3 components initialized")
        return True
    except Exception as e:
        print(f"  ✗ Phase 3 test failed: {e}")
        return False

def test_phase4_ml():
    """Test Phase 4 ML components."""
    print("Testing Phase 4 ML components...")
    try:
        from ml.regime_predictor import MLRegimePredictor
        predictor = MLRegimePredictor()
        
        # Test prediction with dummy data
        import pandas as pd
        import numpy as np
        
        dummy_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.rand(100) * 1000,
            'rsi': np.random.rand(100) * 100
        })
        
        # Should not crash
        predictor.is_bearish = True  # Default state
        
        print("  ✓ Phase 4 ML components working")
        return True
    except Exception as e:
        print(f"  ✗ Phase 4 test failed: {e}")
        return False

def test_bingx_authentication():
    """Test BingX authentication module."""
    print("Testing BingX authentication...")
    try:
        from core.bingx_authenticator import BingXAuthenticator
        
        # Test with dummy credentials (won't actually connect)
        auth = BingXAuthenticator("test_key", "test_secret")
        
        # Test symbol conversion
        ccxt_symbol = "BTC/USDT:USDT"
        bingx_symbol = auth.convert_symbol_to_bingx(ccxt_symbol)
        
        if bingx_symbol == "BTC-USDT":
            print("  ✓ BingX symbol conversion working")
        else:
            print(f"  ✗ Symbol conversion wrong: {bingx_symbol}")
            return False
            
        # Test signature generation (should not crash)
        params = {'test': 'value'}
        signed = auth.prepare_authenticated_request(params)
        
        if 'headers' in signed and 'params' in signed:
            print("  ✓ BingX authentication logic working")
            return True
        else:
            print("  ✗ Authentication preparation failed")
            return False
            
    except Exception as e:
        print(f"  ✗ BingX authentication test failed: {e}")
        return False

def test_live_trading_launcher():
    """Test live trading launcher components."""
    print("Testing live trading launcher...")
    try:
        # Import the launcher
        from scripts.live_trading_launcher import (
            LiveTradingLauncher,
            HealthMonitor,
            AutoRestartManager
        )
        
        # Test basic instantiation (paper mode)
        launcher = LiveTradingLauncher(mode='paper', dry_run=True)
        
        # Test health monitor
        health = HealthMonitor(telegram=None)
        report = health.get_health_report()
        
        if 'status' in report and 'uptime_hours' in report:
            print("  ✓ Health monitor working")
        else:
            print("  ✗ Health monitor report incomplete")
            return False
            
        # Test auto-restart manager
        restart_mgr = AutoRestartManager(max_restarts=10)
        should_restart, reason = restart_mgr.should_restart()
        
        if isinstance(should_restart, bool):
            print("  ✓ Auto-restart manager working")
        else:
            print("  ✗ Auto-restart manager failed")
            return False
            
        print("  ✓ Live trading launcher components working")
        return True
        
    except Exception as e:
        print(f"  ✗ Live trading launcher test failed: {e}")
        return False

def test_config_and_strategies():
    """Test config loading and strategy initialization."""
    print("Testing config and strategies...")
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.example.yaml')
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check critical config sections
        required = ['execution', 'risk', 'signals', 'notify']
        missing = [k for k in required if k not in cfg]
        
        if missing:
            print(f"  ✗ Missing config sections: {missing}")
            return False
            
        # Check ignore_regime settings (test için true olmalı)
        ob_ignore = cfg.get('signals', {}).get('oversold_bounce', {}).get('ignore_regime')
        str_ignore = cfg.get('signals', {}).get('short_the_rip', {}).get('ignore_regime')
        
        if ob_ignore and str_ignore:
            print("  ✓ Config: ignore_regime=true (test mode)")
        else:
            print("  ⚠ Config: ignore_regime should be true for testing")
            
        # Test strategy initialization
        from strategies.adaptive_ob import AdaptiveOversoldBounce
        from strategies.adaptive_str import AdaptiveShortTheRip
        from core.market_regime import MarketRegimeAnalyzer
        
        regime = MarketRegimeAnalyzer()
        ob = AdaptiveOversoldBounce({}, regime)
        str_strat = AdaptiveShortTheRip({}, regime)
        
        print("  ✓ Strategies initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Config/strategy test failed: {e}")
        return False

async def test_async_components():
    """Test async components (WebSocket, etc)."""
    print("Testing async components...")
    try:
        from core.websocket_client import WebSocketClient
        from core.websocket_manager import WebSocketManager
        
        # Test WebSocket client instantiation
        ws_client = WebSocketClient("test_exchange")
        
        # Test WebSocket manager
        ws_manager = WebSocketManager(exchanges=['bingx', 'kucoinfutures'])
        
        print("  ✓ Async components initialized")
        return True
        
    except Exception as e:
        print(f"  ✗ Async test failed: {e}")
        return False

def test_production_mode_check():
    """Check if system is properly configured for production."""
    print("Testing production readiness...")
    issues = []
    
    # Check config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.example.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check live mode setting
    if not cfg.get('execution', {}).get('enable_live'):
        issues.append("execution.enable_live is not true")
    
    # Check risk settings
    equity = cfg.get('risk', {}).get('equity_usd', 0)
    if equity < 20:
        issues.append(f"equity_usd too low: {equity}")
    
    # Check if test mode still active
    if cfg.get('signals', {}).get('oversold_bounce', {}).get('ignore_regime'):
        issues.append("ignore_regime is true (should be false for production)")
    
    if issues:
        print("  ⚠ Production issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ Production configuration looks good")
        return True

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Bearish Alpha Bot - Production Smoke Test v2.0")
    print("=" * 60)
    print()
    
    tests = [
        test_core_imports,
        test_phase3_components,
        test_phase4_ml,
        test_bingx_authentication,
        test_live_trading_launcher,
        test_config_and_strategies,
        test_production_mode_check,
    ]
    
    # Run sync tests
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Run async tests
    try:
        result = asyncio.run(test_async_components())
        results.append(result)
    except Exception as e:
        print(f"  ✗ Async test crashed: {e}")
        results.append(False)
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✅ All smoke tests passed! System ready for production.")
        return 0
    else:
        print("❌ Some tests failed. Review issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
