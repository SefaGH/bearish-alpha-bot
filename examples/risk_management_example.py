#!/usr/bin/env python3
"""
Example: Phase 3.2 Risk Management Engine
Demonstrates comprehensive risk management with real-time monitoring.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager
from core.position_sizing import AdvancedPositionSizing
from core.realtime_risk import RealTimeRiskMonitor
from core.correlation_monitor import CorrelationMonitor
from core.circuit_breaker import CircuitBreakerSystem
from core.websocket_manager import WebSocketManager
from core.performance_monitor import RealTimePerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_risk_configuration():
    """Demonstrate risk configuration management."""
    logger.info("\n" + "="*70)
    logger.info("1. Risk Configuration Management")
    logger.info("="*70)
    
    # Create default configuration
    config = RiskConfiguration()
    logger.info("✓ Default risk configuration created")
    
    # Display default limits
    risk_limits = config.get_risk_limits()
    logger.info(f"  Max portfolio risk per trade: {risk_limits.max_portfolio_risk:.1%}")
    logger.info(f"  Max position size: {risk_limits.max_position_size:.1%}")
    logger.info(f"  Max drawdown: {risk_limits.max_drawdown:.1%}")
    logger.info(f"  Max correlation: {risk_limits.max_correlation:.1%}")
    
    # Display circuit breaker limits
    breaker_limits = config.get_circuit_breaker_limits()
    logger.info(f"  Daily loss limit: {breaker_limits.daily_loss_limit:.1%}")
    logger.info(f"  Position loss limit: {breaker_limits.position_loss_limit:.1%}")
    logger.info(f"  Volatility spike threshold: {breaker_limits.volatility_spike_threshold}σ")
    
    # Custom configuration
    custom_limits = {
        'max_portfolio_risk': 0.015,  # 1.5% instead of 2%
        'max_position_size': 0.08,    # 8% instead of 10%
    }
    custom_config = RiskConfiguration(custom_limits)
    logger.info(f"✓ Custom configuration: {custom_limits}")


async def demonstrate_risk_manager():
    """Demonstrate risk manager functionality."""
    logger.info("\n" + "="*70)
    logger.info("2. Risk Manager - Position Validation & Monitoring")
    logger.info("="*70)
    
    # Initialize risk manager
    portfolio_config = {
        'equity_usd': 10000,
        'max_portfolio_risk': 0.02,
        'max_position_size': 0.10,
        'max_drawdown': 0.15
    }
    
    risk_manager = RiskManager(portfolio_config)
    logger.info(f"✓ Risk manager initialized with ${risk_manager.portfolio_value:,.2f} portfolio")
    
    # Example signal
    signal = {
        'symbol': 'BTC/USDT:USDT',
        'entry': 50000,
        'stop': 49000,
        'target': 52000,
        'side': 'long',
        'strategy': 'oversold_bounce'
    }
    
    # Calculate position size
    position_size = await risk_manager.calculate_position_size(signal)
    logger.info(f"✓ Calculated position size: {position_size:.4f} BTC")
    logger.info(f"  Position value: ${position_size * signal['entry']:,.2f}")
    
    # Validate position
    signal['position_size'] = position_size
    is_valid, reason, metrics = await risk_manager.validate_new_position(signal, {})
    
    if is_valid:
        logger.info(f"✓ Position validation: PASSED")
        logger.info(f"  Risk amount: ${metrics['risk_amount']:.2f}")
        logger.info(f"  Risk/Reward ratio: {metrics['risk_reward_ratio']:.2f}")
        logger.info(f"  Portfolio heat: {metrics.get('portfolio_heat', 0):.2%}")
        
        # Register position
        risk_manager.register_position('pos_1', {
            'symbol': signal['symbol'],
            'entry_price': signal['entry'],
            'stop_loss': signal['stop'],
            'size': position_size,
            'side': signal['side'],
            'risk_amount': metrics['risk_amount']
        })
        logger.info(f"✓ Position registered: pos_1")
    else:
        logger.warning(f"✗ Position validation FAILED: {reason}")
    
    # Get portfolio summary
    summary = risk_manager.get_portfolio_summary()
    logger.info(f"\nPortfolio Summary:")
    logger.info(f"  Value: ${summary['portfolio_value']:,.2f}")
    logger.info(f"  Active positions: {summary['active_positions']}")
    logger.info(f"  Total risk: ${summary['total_risk']:.2f}")
    logger.info(f"  Portfolio heat: {summary['portfolio_heat']:.2%}")


async def demonstrate_position_sizing():
    """Demonstrate advanced position sizing algorithms."""
    logger.info("\n" + "="*70)
    logger.info("3. Advanced Position Sizing Algorithms")
    logger.info("="*70)
    
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    sizing = AdvancedPositionSizing(risk_manager)
    
    signal = {
        'entry': 50000,
        'stop': 49000,
        'target': 52000,
        'side': 'long',
        'atr': 500
    }
    
    # Kelly Criterion
    logger.info("\n[Kelly Criterion]")
    performance_history = {
        'win_rate': 0.6,
        'avg_win': 100,
        'avg_loss': 50
    }
    kelly_size = await sizing.calculate_optimal_size(
        signal,
        method='kelly',
        performance_history=performance_history
    )
    logger.info(f"  Position size: {kelly_size:.4f}")
    logger.info(f"  Based on: 60% win rate, 2:1 win/loss ratio")
    
    # Fixed Risk
    logger.info("\n[Fixed Risk]")
    fixed_size = await sizing.calculate_optimal_size(
        signal,
        method='fixed_risk',
        risk_per_trade=200
    )
    logger.info(f"  Position size: {fixed_size:.4f}")
    logger.info(f"  Based on: $200 fixed risk per trade")
    
    # Volatility Adjusted
    logger.info("\n[Volatility Adjusted]")
    vol_size = await sizing.calculate_optimal_size(
        signal,
        method='volatility_adjusted',
        target_risk=200,
        market_volatility=500,
        avg_volatility=400
    )
    logger.info(f"  Position size: {vol_size:.4f}")
    logger.info(f"  Based on: ATR {signal['atr']}, adjusted for current volatility")
    
    # Regime Based
    logger.info("\n[Regime Based]")
    market_regime = {
        'trend': 'bullish',
        'risk_multiplier': 1.2,
        'volatility': 'normal'
    }
    regime_size = await sizing.calculate_optimal_size(
        signal,
        method='regime_based',
        market_regime=market_regime,
        base_risk=200
    )
    logger.info(f"  Position size: {regime_size:.4f}")
    logger.info(f"  Based on: Bullish trend, normal volatility, 1.2x multiplier")


async def demonstrate_realtime_monitoring():
    """Demonstrate real-time risk monitoring."""
    logger.info("\n" + "="*70)
    logger.info("4. Real-Time Risk Monitoring")
    logger.info("="*70)
    
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    monitor = RealTimeRiskMonitor(risk_manager, None)
    
    # Register a position
    risk_manager.register_position('pos_1', {
        'symbol': 'BTC/USDT:USDT',
        'entry_price': 50000,
        'stop_loss': 49000,
        'size': 0.1,
        'side': 'long',
        'risk_amount': 100
    })
    logger.info("✓ Position registered for monitoring")
    
    # Simulate price updates
    logger.info("\nSimulating price movements:")
    
    # Price moves up (profit)
    await monitor.on_price_update('BTC/USDT:USDT', {'last': 50500})
    logger.info(f"  Price: $50,500 - Position in profit")
    
    # Price moves down (still above stop)
    await monitor.on_price_update('BTC/USDT:USDT', {'last': 49500})
    logger.info(f"  Price: $49,500 - Position near stop loss")
    
    # Price breaches stop loss
    await monitor.on_price_update('BTC/USDT:USDT', {'last': 48500})
    logger.info(f"  Price: $48,500 - STOP LOSS TRIGGERED!")
    
    # Check for alerts
    alerts = await monitor.get_risk_alerts(count=10)
    logger.info(f"\n✓ Generated {len(alerts)} risk alerts:")
    for alert in alerts:
        logger.info(f"  - {alert['type']}: {alert['message']}")
    
    # Calculate VaR
    logger.info("\nValue at Risk (VaR) Calculation:")
    
    # Add price history
    prices = [50000, 50500, 49500, 51000, 49800, 52000, 48500, 51500]
    for price in prices:
        monitor.update_price_history('BTC/USDT:USDT', price)
    
    var_metrics = monitor.calculate_portfolio_var(confidence=0.05)
    logger.info(f"  Historical VaR (95% confidence): ${var_metrics['historical_var']:.2f}")
    logger.info(f"  Parametric VaR: ${var_metrics['parametric_var']:.2f}")
    logger.info(f"  Expected Shortfall: ${var_metrics['expected_shortfall']:.2f}")


async def demonstrate_correlation_monitoring():
    """Demonstrate correlation and diversification monitoring."""
    logger.info("\n" + "="*70)
    logger.info("5. Correlation & Diversification Monitoring")
    logger.info("="*70)
    
    monitor = CorrelationMonitor()
    
    # Simulate correlated price movements
    logger.info("\nSimulating price history for correlation analysis...")
    
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    for i in range(50):
        # BTC and ETH highly correlated, SOL less correlated
        btc_price = 50000 + i * 100
        eth_price = 3000 + i * 6
        sol_price = 100 + (i % 10)
        
        monitor.update_price_history('BTC/USDT:USDT', btc_price)
        monitor.update_price_history('ETH/USDT:USDT', eth_price)
        monitor.update_price_history('SOL/USDT:USDT', sol_price)
    
    # Update correlation matrix
    await monitor.update_correlation_matrix(symbols)
    logger.info("✓ Correlation matrix calculated")
    
    # Display correlations
    logger.info("\nCorrelations:")
    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 < sym2:  # Avoid duplicates
                corr = monitor.get_correlation(sym1, sym2)
                if corr is not None:
                    logger.info(f"  {sym1} <-> {sym2}: {corr:.3f}")
    
    # Calculate diversification
    positions = {
        'pos_1': {'symbol': 'BTC/USDT:USDT', 'size': 0.1, 'entry_price': 50000},
        'pos_2': {'symbol': 'ETH/USDT:USDT', 'size': 1.0, 'entry_price': 3000},
        'pos_3': {'symbol': 'SOL/USDT:USDT', 'size': 50.0, 'entry_price': 100}
    }
    
    metrics = monitor.calculate_portfolio_diversification(positions)
    logger.info(f"\nPortfolio Diversification:")
    logger.info(f"  Number of positions: {metrics['num_positions']}")
    logger.info(f"  Effective positions: {metrics['effective_positions']:.2f}")
    logger.info(f"  Concentration risk: {metrics['concentration_risk']:.2%}")
    logger.info(f"  Diversification ratio: {metrics['diversification_ratio']:.2f}")
    
    # Check correlation alerts
    alerts = monitor.get_correlation_alerts()
    if alerts:
        logger.info(f"\n⚠ Correlation alerts:")
        for alert in alerts:
            logger.info(f"  - {alert['type']}: {alert['message']}")
    else:
        logger.info("\n✓ No correlation alerts - portfolio well diversified")


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker system."""
    logger.info("\n" + "="*70)
    logger.info("6. Circuit Breaker & Emergency Stop System")
    logger.info("="*70)
    
    portfolio_config = {'equity_usd': 10000}
    risk_manager = RiskManager(portfolio_config)
    breaker = CircuitBreakerSystem(risk_manager)
    
    # Configure circuit breakers
    breaker.set_circuit_breakers(
        daily_loss_limit=0.05,      # 5% daily loss
        position_loss_limit=0.03,   # 3% position loss
        volatility_spike_threshold=3.0
    )
    logger.info("✓ Circuit breakers configured")
    logger.info(f"  Daily loss limit: 5%")
    logger.info(f"  Position loss limit: 3%")
    logger.info(f"  Volatility spike threshold: 3.0σ")
    
    # Simulate positions
    risk_manager.register_position('pos_1', {
        'symbol': 'BTC/USDT:USDT',
        'entry_price': 50000,
        'current_price': 48000,  # -4% loss
        'size': 0.1,
        'side': 'long',
        'unrealized_pnl': -200
    })
    
    risk_manager.register_position('pos_2', {
        'symbol': 'ETH/USDT:USDT',
        'entry_price': 3000,
        'current_price': 2800,  # -6.7% loss
        'size': 1.0,
        'side': 'long',
        'unrealized_pnl': -200
    })
    
    logger.info(f"\n✓ Registered 2 positions with losses")
    
    # Simulate portfolio drawdown
    risk_manager.portfolio_value = 9400  # $600 loss (6% drawdown)
    risk_manager.peak_portfolio_value = 10000
    risk_manager.current_drawdown = 0.06
    
    logger.info(f"\nCurrent portfolio state:")
    logger.info(f"  Portfolio value: ${risk_manager.portfolio_value:,.2f}")
    logger.info(f"  Drawdown: {risk_manager.current_drawdown:.2%}")
    logger.info(f"  Active positions: {len(risk_manager.active_positions)}")
    
    # Trigger circuit breaker
    logger.info(f"\n⚠ Daily loss limit exceeded - triggering circuit breaker...")
    await breaker.trigger_circuit_breaker('daily_loss', severity='critical')
    
    # Check status
    status = breaker.get_breaker_status()
    logger.info(f"\nCircuit Breaker Status:")
    logger.info(f"  Daily loss breaker: {'TRIGGERED' if status['circuit_breakers']['daily_loss']['triggered'] else 'OK'}")
    logger.info(f"  Total triggers: {status['active_triggers']}")
    
    # Show positions after emergency protocol
    logger.info(f"\nPositions after emergency protocol:")
    logger.info(f"  Active positions: {len(risk_manager.active_positions)}")
    logger.info(f"  Portfolio value: ${risk_manager.portfolio_value:,.2f}")


async def main():
    """Run all risk management demonstrations."""
    logger.info("="*70)
    logger.info("Phase 3.2: Risk Management Engine Demonstration")
    logger.info("="*70)
    logger.info("Comprehensive risk management with real-time monitoring")
    logger.info("")
    
    try:
        # Run demonstrations
        await demonstrate_risk_configuration()
        await demonstrate_risk_manager()
        await demonstrate_position_sizing()
        await demonstrate_realtime_monitoring()
        await demonstrate_correlation_monitoring()
        await demonstrate_circuit_breaker()
        
        logger.info("\n" + "="*70)
        logger.info("Risk Management Engine Demonstration Complete")
        logger.info("="*70)
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Configurable risk limits and circuit breakers")
        logger.info("✓ Position validation with multiple risk checks")
        logger.info("✓ Advanced position sizing (Kelly, Fixed, Volatility, Regime)")
        logger.info("✓ Real-time risk monitoring with alerts")
        logger.info("✓ VaR calculation (Historical, Parametric, Expected Shortfall)")
        logger.info("✓ Correlation analysis and diversification metrics")
        logger.info("✓ Emergency stop and circuit breaker protocols")
        logger.info("\nReady for integration with live trading system!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
