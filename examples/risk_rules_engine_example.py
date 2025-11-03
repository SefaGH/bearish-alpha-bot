"""
Example: Using the Phase 3 Risk Rules Engine

This example demonstrates how to use the modular risk rules engine
to validate trading signals with customizable risk management rules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import asyncio
from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager
from core.risk_rules import (
    CapitalLimitRule,
    PositionSizeRule,
    PortfolioHeatRule,
    MaxDrawdownRule,
    RiskRewardRatioRule,
    StrategyPerformanceRule
)


class MockPortfolioManager:
    """Mock portfolio manager for demonstration."""
    
    def __init__(self, equity=10000, exposure=0, drawdown=0.0):
        self.equity = equity
        self.exposure = exposure
        self.drawdown = drawdown
        self.positions = {}
    
    def get_current_equity(self):
        return self.equity
    
    def get_total_exposure(self):
        return self.exposure
    
    def get_current_drawdown(self):
        return self.drawdown
    
    def get_open_positions(self):
        return self.positions


def example_1_default_rules():
    """Example 1: Using default risk rules configuration."""
    print("="*60)
    print("Example 1: Default Risk Rules Configuration")
    print("="*60)
    
    # Create risk configuration
    risk_config = RiskConfiguration()
    
    # Create risk manager with default rules
    risk_manager = RiskManager(
        portfolio_value=10000,
        risk_config=risk_config
    )
    
    # Create portfolio manager
    portfolio = MockPortfolioManager(equity=10000, exposure=5000)
    
    # Create a trading signal
    signal = {
        'symbol': 'BTC/USDT',
        'entry': 50000,
        'stop': 49000,
        'target': 52000,
        'position_size': 0.02,  # 0.02 BTC = $1000 position
        'side': 'long',
        'strategy': 'momentum'
    }
    
    # Validate signal
    print(f"\nValidating signal for {signal['symbol']}...")
    print(f"Entry: ${signal['entry']:,.0f}")
    print(f"Stop: ${signal['stop']:,.0f}")
    print(f"Target: ${signal['target']:,.0f}")
    print(f"Position Size: {signal['position_size']} BTC (${signal['position_size'] * signal['entry']:,.0f})")
    
    # Run validation
    is_valid, reason, metrics = asyncio.run(
        risk_manager.validate_new_position(signal, portfolio_manager=portfolio)
    )
    
    print(f"\n{'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    print(f"Reason: {reason}")
    print(f"\nRisk Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_2_custom_rules():
    """Example 2: Creating custom risk rules configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Risk Rules Configuration")
    print("="*60)
    
    # Create custom risk configuration
    custom_config = {
        'max_position_size': 0.05,  # More conservative: 5% max
        'max_portfolio_risk': 0.01,  # More conservative: 1% max risk
        'max_drawdown': 0.10,  # Stricter: 10% max drawdown
    }
    
    risk_config = RiskConfiguration(custom_limits=custom_config)
    
    # Create custom rule set
    custom_rules = [
        CapitalLimitRule(),
        PositionSizeRule(max_position_size=0.05),  # 5% instead of default 10%
        PortfolioHeatRule(max_portfolio_heat=0.08, max_portfolio_risk=0.01),  # Stricter
        MaxDrawdownRule(max_drawdown=0.10),  # 10% instead of default 15%
        RiskRewardRatioRule(min_risk_reward=2.0),  # Higher minimum: 2:1 instead of 1.5:1
    ]
    
    # Create risk manager with custom rules
    risk_manager = RiskManager(
        portfolio_value=10000,
        risk_config=risk_config,
        rules=custom_rules
    )
    
    portfolio = MockPortfolioManager(equity=10000, exposure=3000)
    
    # Same signal as before
    signal = {
        'symbol': 'BTC/USDT',
        'entry': 50000,
        'stop': 49000,
        'target': 52000,
        'position_size': 0.02,  # This might be rejected now due to stricter rules
        'side': 'long'
    }
    
    print(f"\nValidating signal with CUSTOM (stricter) rules...")
    print(f"Custom Rules:")
    print(f"  - Max Position Size: 5% (vs 10% default)")
    print(f"  - Max Portfolio Risk: 1% (vs 2% default)")
    print(f"  - Max Drawdown: 10% (vs 15% default)")
    print(f"  - Min R/R Ratio: 2.0 (vs 1.5 default)")
    
    is_valid, reason, metrics = asyncio.run(
        risk_manager.validate_new_position(signal, portfolio_manager=portfolio)
    )
    
    print(f"\n{'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    print(f"Reason: {reason}")


def example_3_enabling_disabling_rules():
    """Example 3: Dynamically enabling/disabling rules."""
    print("\n" + "="*60)
    print("Example 3: Dynamically Enabling/Disabling Rules")
    print("="*60)
    
    risk_config = RiskConfiguration()
    
    # Create custom rule set
    rules = [
        CapitalLimitRule(),
        PositionSizeRule(max_position_size=0.10),
        MaxDrawdownRule(max_drawdown=0.15),
        RiskRewardRatioRule(min_risk_reward=1.5),
    ]
    
    risk_manager = RiskManager(
        portfolio_value=10000,
        risk_config=risk_config,
        rules=rules
    )
    
    portfolio = MockPortfolioManager(equity=10000, exposure=0)
    
    # Signal with poor risk/reward ratio
    signal = {
        'symbol': 'ETH/USDT',
        'entry': 3000,
        'stop': 2900,
        'target': 3050,  # R/R = 50/100 = 0.5 (below 1.5 minimum)
        'position_size': 0.5,  # 0.5 ETH = $1500 position
        'side': 'long'
    }
    
    print("\nTest 1: With R/R rule enabled")
    print(f"Signal R/R ratio: {(3050-3000)/(3000-2900):.2f} (below 1.5 minimum)")
    
    is_valid, reason, metrics = asyncio.run(
        risk_manager.validate_new_position(signal, portfolio_manager=portfolio)
    )
    
    print(f"Result: {'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    print(f"Reason: {reason}")
    
    # Disable R/R rule
    print("\nTest 2: Disabling R/R rule...")
    for rule in risk_manager.rules:
        if isinstance(rule, RiskRewardRatioRule):
            rule.disable()
            print(f"Disabled: {rule.rule_name}")
    
    is_valid, reason, metrics = asyncio.run(
        risk_manager.validate_new_position(signal, portfolio_manager=portfolio)
    )
    
    print(f"Result: {'✅ APPROVED' if is_valid else '❌ REJECTED'}")
    print(f"Reason: {reason}")


def example_4_testing_multiple_signals():
    """Example 4: Testing multiple signals through rules engine."""
    print("\n" + "="*60)
    print("Example 4: Testing Multiple Signals")
    print("="*60)
    
    risk_config = RiskConfiguration()
    risk_manager = RiskManager(
        portfolio_value=10000,
        risk_config=risk_config
    )
    
    portfolio = MockPortfolioManager(equity=10000, exposure=0)
    
    # Multiple signals with different characteristics
    signals = [
        {
            'name': 'Good Signal',
            'symbol': 'BTC/USDT',
            'entry': 50000,
            'stop': 49000,
            'target': 52000,  # R/R = 2.0
            'position_size': 0.02,
            'side': 'long'
        },
        {
            'name': 'Too Large Position',
            'symbol': 'ETH/USDT',
            'entry': 3000,
            'stop': 2900,
            'target': 3200,
            'position_size': 0.5,  # $1500 (15% of portfolio - too large)
            'side': 'long'
        },
        {
            'name': 'Poor R/R',
            'symbol': 'SOL/USDT',
            'entry': 100,
            'stop': 98,
            'target': 101,  # R/R = 1/2 = 0.5
            'position_size': 10,
            'side': 'long'
        },
        {
            'name': 'High Risk',
            'symbol': 'AVAX/USDT',
            'entry': 30,
            'stop': 20,  # $10 risk distance
            'target': 50,
            'position_size': 30,  # Risk: 10 * 30 = $300 (3% of portfolio - too high)
            'side': 'long'
        }
    ]
    
    print("\nTesting signals through rules engine:\n")
    
    approved = 0
    rejected = 0
    
    for signal in signals:
        name = signal.pop('name')
        is_valid, reason, metrics = asyncio.run(
            risk_manager.validate_new_position(signal, portfolio_manager=portfolio)
        )
        
        status = '✅' if is_valid else '❌'
        approved += 1 if is_valid else 0
        rejected += 0 if is_valid else 1
        
        print(f"{status} {name} ({signal['symbol']})")
        print(f"   {reason}")
    
    print(f"\nSummary:")
    print(f"  Approved: {approved}/{len(signals)}")
    print(f"  Rejected: {rejected}/{len(signals)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Risk Rules Engine Examples")
    print("Phase 3: Modular, Extensible Risk Management")
    print("="*60)
    
    # Run all examples
    example_1_default_rules()
    example_2_custom_rules()
    example_3_enabling_disabling_rules()
    example_4_testing_multiple_signals()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
