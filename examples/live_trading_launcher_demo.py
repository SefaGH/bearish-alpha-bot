#!/usr/bin/env python3
"""
Live Trading Launcher Demo

Demonstrates how to use the live trading launcher programmatically
and shows the complete initialization flow.

This is a demonstration script that shows the capabilities of the
live trading launcher without requiring actual API credentials.
"""

import sys
import os
import asyncio

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from live_trading_launcher import LiveTradingLauncher


async def demo_launcher_capabilities():
    """Demonstrate the launcher's capabilities."""
    
    print("="*70)
    print("LIVE TRADING LAUNCHER DEMO")
    print("="*70)
    print()
    
    # Create launcher instance
    print("1. Creating Launcher Instance...")
    launcher = LiveTradingLauncher(mode='paper', dry_run=True)
    print(f"   ✓ Launcher created in {launcher.mode.upper()} mode")
    print()
    
    # Show configuration
    print("2. Trading Configuration:")
    print(f"   Capital: {launcher.CAPITAL_USDT} USDT")
    print(f"   Exchange: BingX")
    print(f"   Trading Pairs: {len(launcher.TRADING_PAIRS)}")
    for i, pair in enumerate(launcher.TRADING_PAIRS, 1):
        print(f"      {i}. {pair}")
    print()
    
    # Show risk parameters
    print("3. Risk Parameters:")
    for param, value in launcher.RISK_PARAMS.items():
        if isinstance(value, float) and value < 1:
            print(f"   {param}: {value:.1%}")
        else:
            print(f"   {param}: {value}")
    print()
    
    # Show initialization phases
    print("4. Initialization Phases (8 steps):")
    phases = [
        "Environment Configuration",
        "Exchange Connection",
        "Risk Management",
        "AI Components (Phase 4)",
        "Trading Strategies",
        "Production System",
        "Strategy Registration",
        "Pre-Flight Checks"
    ]
    for i, phase in enumerate(phases, 1):
        print(f"   [{i}/8] {phase}")
    print()
    
    # Show AI components
    print("5. Phase 4 AI Integration:")
    ai_components = [
        "ML Regime Prediction - Market regime forecasting",
        "Adaptive Learning - Real-time strategy adaptation",
        "Strategy Optimization - Multi-objective optimization",
        "Price Prediction - LSTM/Transformer forecasting"
    ]
    for component in ai_components:
        print(f"   • {component}")
    print()
    
    # Show safety features
    print("6. Safety Features:")
    safety_features = [
        "Pre-flight system checks (5 validations)",
        "Emergency shutdown protocols",
        "Circuit breaker integration",
        "Risk limit enforcement",
        "Position size controls",
        "Drawdown protection",
        "Real-time monitoring"
    ]
    for feature in safety_features:
        print(f"   • {feature}")
    print()
    
    # Show command examples
    print("7. Usage Examples:")
    examples = [
        ("Dry run (checks only)", "python scripts/live_trading_launcher.py --dry-run"),
        ("Paper trading", "python scripts/live_trading_launcher.py --paper"),
        ("Timed session (1 hour)", "python scripts/live_trading_launcher.py --paper --duration 3600"),
        ("Live trading", "python scripts/live_trading_launcher.py"),
    ]
    for description, command in examples:
        print(f"   {description}:")
        print(f"      {command}")
        print()
    
    print("="*70)
    print("Demo complete! See README_LIVE_TRADING_LAUNCHER.md for full docs")
    print("="*70)


async def demo_initialization_flow():
    """
    Demonstrate the initialization flow without actual credentials.
    Shows what would happen during a real launch.
    """
    
    print("\n" + "="*70)
    print("INITIALIZATION FLOW DEMONSTRATION")
    print("="*70)
    print()
    
    print("This demonstrates what happens during launcher initialization:")
    print()
    
    steps = [
        {
            "phase": "1/8 - Environment Configuration",
            "actions": [
                "Load BINGX_KEY and BINGX_SECRET",
                "Check for Telegram credentials (optional)",
                "Validate required variables present"
            ]
        },
        {
            "phase": "2/8 - Exchange Connection",
            "actions": [
                "Initialize BingX CcxtClient",
                "Test API connection",
                "Validate all 8 trading pairs available"
            ]
        },
        {
            "phase": "3/8 - Risk Management",
            "actions": [
                "Load RiskConfiguration",
                "Set max position size (15%)",
                "Configure stop loss (5%) and take profit (10%)",
                "Set max drawdown (15%)"
            ]
        },
        {
            "phase": "4/8 - AI Components",
            "actions": [
                "Initialize MLRegimePredictor",
                "Initialize StrategyOptimizer",
                "Initialize AdvancedPricePredictionEngine",
                "Create AIEnhancedStrategyAdapter"
            ]
        },
        {
            "phase": "5/8 - Trading Strategies",
            "actions": [
                "Initialize AdaptiveOversoldBounce",
                "Initialize AdaptiveShortTheRip",
                "Configure strategy parameters"
            ]
        },
        {
            "phase": "6/8 - Production System",
            "actions": [
                "Create ProductionCoordinator",
                "Initialize portfolio manager (100 USDT)",
                "Initialize risk manager",
                "Initialize live trading engine",
                "Establish websocket connections",
                "Activate circuit breaker"
            ]
        },
        {
            "phase": "7/8 - Strategy Registration",
            "actions": [
                "Register strategies with portfolio manager",
                "Allocate capital equally across strategies",
                "Validate strategy configurations"
            ]
        },
        {
            "phase": "8/8 - Pre-Flight Checks",
            "actions": [
                "Check 1/5: Exchange connectivity",
                "Check 2/5: System state validation",
                "Check 3/5: Risk limits verification",
                "Check 4/5: Strategy registration",
                "Check 5/5: Emergency protocols"
            ]
        }
    ]
    
    for step in steps:
        print(f"📋 {step['phase']}")
        for action in step['actions']:
            print(f"   → {action}")
        print()
    
    print("="*70)
    print("After successful initialization, trading loop begins")
    print("="*70)


async def main():
    """Run all demonstrations."""
    
    # Demo 1: Capabilities
    await demo_launcher_capabilities()
    
    # Small pause
    await asyncio.sleep(1)
    
    # Demo 2: Initialization flow
    await demo_initialization_flow()
    
    print("\n💡 To test with actual credentials, set environment variables:")
    print("   export BINGX_KEY=your_key")
    print("   export BINGX_SECRET=your_secret")
    print()
    print("Then run:")
    print("   python scripts/live_trading_launcher.py --dry-run")


if __name__ == '__main__':
    asyncio.run(main())
