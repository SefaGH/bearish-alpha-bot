#!/usr/bin/env python3
"""
Example: Complete Phase 3.4 Live Trading System
Demonstrates the full production-ready live trading engine with all Phase 3 integration.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.production_coordinator import ProductionCoordinator
from core.ccxt_client import CcxtClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStrategy:
    """Mock trading strategy for demonstration."""
    
    def __init__(self, name):
        self.name = name
    
    async def generate_signal(self):
        """Generate a mock trading signal."""
        return {
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': self.name,
            'exchange': 'kucoinfutures',
            'urgency': 'normal'
        }


async def example_1_basic_live_trading():
    """Example 1: Basic live trading setup and initialization."""
    logger.info("="*70)
    logger.info("Example 1: Basic Live Trading Setup")
    logger.info("="*70)
    
    try:
        # Create mock exchange clients
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        
        # Portfolio configuration
        portfolio_config = {
            'equity_usd': 10000  # $10,000 starting capital
        }
        
        # Initialize production coordinator
        coordinator = ProductionCoordinator()
        
        logger.info("\nInitializing production system...")
        init_result = await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        if init_result['success']:
            logger.info("✓ Production system initialized successfully")
            logger.info(f"Components: {init_result['components']}")
        else:
            logger.error(f"✗ Initialization failed: {init_result.get('reason')}")
            return
        
        # Get system state
        state = coordinator.get_system_state()
        logger.info(f"\nSystem state: {state['is_initialized']}")
        
        logger.info("\nExample 1 complete - System ready for trading")
        
    except Exception as e:
        logger.error(f"Error in example 1: {e}")


async def example_2_register_strategies():
    """Example 2: Register trading strategies with portfolio manager."""
    logger.info("\n" + "="*70)
    logger.info("Example 2: Register Trading Strategies")
    logger.info("="*70)
    
    try:
        # Setup
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        portfolio_config = {'equity_usd': 10000}
        
        coordinator = ProductionCoordinator()
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        # Register multiple strategies
        strategies = [
            ('momentum_strategy', MockStrategy('momentum'), 0.30),
            ('mean_reversion', MockStrategy('mean_reversion'), 0.30),
            ('breakout_strategy', MockStrategy('breakout'), 0.40)
        ]
        
        logger.info("\nRegistering strategies:")
        for name, instance, allocation in strategies:
            result = coordinator.register_strategy(name, instance, allocation)
            if result.get('status') == 'success':
                logger.info(f"  ✓ {name}: {allocation*100}% allocation")
            else:
                logger.error(f"  ✗ {name}: {result.get('reason', 'Unknown error')}")
        
        # Check portfolio state
        portfolio_state = coordinator.portfolio_manager.portfolio_state
        logger.info(f"\nPortfolio state:")
        logger.info(f"  Total value: ${portfolio_state['total_value']:.2f}")
        logger.info(f"  Allocated capital: ${portfolio_state['allocated_capital']:.2f}")
        logger.info(f"  Available capital: ${portfolio_state['available_capital']:.2f}")
        
        logger.info("\nExample 2 complete")
        
    except Exception as e:
        logger.error(f"Error in example 2: {e}")


async def example_3_signal_execution():
    """Example 3: Execute trading signals through the engine."""
    logger.info("\n" + "="*70)
    logger.info("Example 3: Signal Execution")
    logger.info("="*70)
    
    try:
        # Setup
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        portfolio_config = {'equity_usd': 10000}
        
        coordinator = ProductionCoordinator()
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        # Start trading engine in paper mode
        logger.info("\nStarting trading engine (paper mode)...")
        await coordinator.trading_engine.start_live_trading(mode='paper')
        
        # Create and submit a trading signal
        signal = {
            'signal_id': 'test_signal_1',
            'symbol': 'BTC/USDT:USDT',
            'side': 'buy',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test_strategy',
            'exchange': 'kucoinfutures',
            'position_size': 0.01
        }
        
        logger.info(f"\nSubmitting signal for {signal['symbol']}...")
        result = await coordinator.trading_engine.execute_signal(signal)
        
        if result['success']:
            logger.info(f"✓ Signal executed successfully")
            logger.info(f"  Position ID: {result['position_id']}")
            logger.info(f"  Order ID: {result['order_id']}")
        else:
            logger.error(f"✗ Signal execution failed: {result.get('reason')}")
            logger.error(f"  Failed at stage: {result.get('stage')}")
        
        # Check engine status
        engine_status = coordinator.trading_engine.get_engine_status()
        logger.info(f"\nEngine status:")
        logger.info(f"  State: {engine_status['state']}")
        logger.info(f"  Active positions: {engine_status['active_positions']}")
        logger.info(f"  Total trades: {engine_status['total_trades']}")
        
        # Stop engine
        await coordinator.trading_engine.stop_live_trading()
        logger.info("\nExample 3 complete")
        
    except Exception as e:
        logger.error(f"Error in example 3: {e}")


async def example_4_position_management():
    """Example 4: Monitor and manage positions."""
    logger.info("\n" + "="*70)
    logger.info("Example 4: Position Management")
    logger.info("="*70)
    
    try:
        # Setup
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        portfolio_config = {'equity_usd': 10000}
        
        coordinator = ProductionCoordinator()
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        await coordinator.trading_engine.start_live_trading(mode='paper')
        
        # Open a position
        signal = {
            'symbol': 'BTC/USDT:USDT',
            'side': 'long',
            'entry': 50000.0,
            'stop': 49000.0,
            'target': 52000.0,
            'strategy': 'test',
            'exchange': 'kucoinfutures',
            'position_size': 0.01
        }
        
        exec_result = await coordinator.trading_engine.execute_signal(signal)
        position_id = exec_result['position_id']
        
        logger.info(f"\nPosition opened: {position_id}")
        
        # Monitor P&L with price changes
        prices = [50500.0, 51000.0, 51500.0, 50800.0]
        
        for price in prices:
            pnl_result = await coordinator.trading_engine.position_manager.monitor_position_pnl(
                position_id, current_price=price
            )
            
            if pnl_result['success']:
                logger.info(f"\nPrice: ${price:.2f}")
                logger.info(f"  Unrealized P&L: ${pnl_result['unrealized_pnl']:.2f}")
                logger.info(f"  P&L %: {pnl_result['pnl_pct']:.2f}%")
                
                if pnl_result.get('exit_signal'):
                    logger.info(f"  Exit signal: {pnl_result['exit_signal']['reason']}")
        
        # Get position metrics
        metrics_result = coordinator.trading_engine.position_manager.calculate_position_metrics(position_id)
        if metrics_result['success']:
            metrics = metrics_result['metrics']
            logger.info(f"\nPosition metrics:")
            logger.info(f"  Status: {metrics['status']}")
            logger.info(f"  Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
            logger.info(f"  Max Favorable Excursion: ${metrics['max_favorable_excursion']:.2f}")
        
        # Close position
        close_result = await coordinator.trading_engine.position_manager.close_position(
            position_id, exit_price=51000.0, exit_reason='manual'
        )
        
        if close_result['success']:
            logger.info(f"\nPosition closed:")
            logger.info(f"  Realized P&L: ${close_result['realized_pnl']:.2f}")
            logger.info(f"  Return: {close_result['return_pct']:.2f}%")
        
        await coordinator.trading_engine.stop_live_trading()
        logger.info("\nExample 4 complete")
        
    except Exception as e:
        logger.error(f"Error in example 4: {e}")


async def example_5_execution_analytics():
    """Example 5: Analyze execution quality."""
    logger.info("\n" + "="*70)
    logger.info("Example 5: Execution Analytics")
    logger.info("="*70)
    
    try:
        # Setup
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        portfolio_config = {'equity_usd': 10000}
        
        coordinator = ProductionCoordinator()
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        await coordinator.trading_engine.start_live_trading(mode='paper')
        
        # Execute multiple signals
        for i in range(3):
            signal = {
                'symbol': 'BTC/USDT:USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'entry': 50000.0 + (i * 100),
                'stop': 49000.0 + (i * 100),
                'target': 52000.0 + (i * 100),
                'strategy': 'test',
                'exchange': 'kucoinfutures',
                'position_size': 0.01
            }
            
            await coordinator.trading_engine.execute_signal(signal)
            await asyncio.sleep(0.1)
        
        logger.info("\nExecuted 3 test signals")
        
        # Generate execution report
        analytics = coordinator.trading_engine.execution_analytics
        report_result = analytics.generate_execution_report('1d')
        
        if report_result['success']:
            report = report_result['report']
            logger.info("\nExecution Report:")
            logger.info(f"  Period: {report['report_period']}")
            
            if 'performance_metrics' in report:
                perf = report['performance_metrics']
                logger.info(f"\n  Performance Metrics:")
                logger.info(f"    Total orders: {perf['total_orders']}")
                logger.info(f"    Success rate: {perf['success_rate']*100:.1f}%")
                logger.info(f"    Avg execution time: {perf['avg_execution_time']:.2f}s")
            
            if 'position_performance' in report:
                pos_perf = report['position_performance']
                logger.info(f"\n  Position Performance:")
                logger.info(f"    Active: {pos_perf['active_positions']}")
                logger.info(f"    Closed: {pos_perf['closed_positions']}")
                logger.info(f"    Total P&L: ${pos_perf['total_pnl']:.2f}")
            
            if 'recommendations' in report:
                logger.info(f"\n  Recommendations:")
                for rec in report['recommendations']:
                    logger.info(f"    • {rec}")
        
        await coordinator.trading_engine.stop_live_trading()
        logger.info("\nExample 5 complete")
        
    except Exception as e:
        logger.error(f"Error in example 5: {e}")


async def example_6_production_loop():
    """Example 6: Run production trading loop (short duration)."""
    logger.info("\n" + "="*70)
    logger.info("Example 6: Production Trading Loop")
    logger.info("="*70)
    
    try:
        # Setup
        exchange_clients = {
            'kucoinfutures': CcxtClient('kucoinfutures')
        }
        portfolio_config = {'equity_usd': 10000}
        
        coordinator = ProductionCoordinator()
        await coordinator.initialize_production_system(
            exchange_clients=exchange_clients,
            portfolio_config=portfolio_config
        )
        
        # Register a strategy
        coordinator.register_strategy('test_strategy', MockStrategy('test'), 0.25)
        
        logger.info("\nStarting production loop for 10 seconds...")
        logger.info("(In production, duration would be None for continuous operation)")
        
        # Run for 10 seconds
        await coordinator.run_production_loop(mode='paper', duration=10)
        
        # Get final system state
        state = coordinator.get_system_state()
        logger.info(f"\nFinal system state:")
        logger.info(f"  Is running: {state['is_running']}")
        logger.info(f"  Emergency stop: {state['emergency_stop']}")
        
        if 'trading_engine' in state:
            logger.info(f"\n  Trading engine:")
            logger.info(f"    State: {state['trading_engine']['state']}")
            logger.info(f"    Total trades: {state['trading_engine']['total_trades']}")
        
        logger.info("\nExample 6 complete")
        
    except Exception as e:
        logger.error(f"Error in example 6: {e}")


async def main():
    """Run all examples."""
    logger.info("="*70)
    logger.info("Phase 3.4 Live Trading Engine - Complete Examples")
    logger.info("="*70)
    
    examples = [
        example_1_basic_live_trading,
        example_2_register_strategies,
        example_3_signal_execution,
        example_4_position_management,
        example_5_execution_analytics,
        example_6_production_loop,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            await example_func()
            if i < len(examples):
                await asyncio.sleep(2)  # Pause between examples
        except Exception as e:
            logger.error(f"Error running example {i}: {e}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("All examples completed!")
    logger.info("="*70)


if __name__ == '__main__':
    asyncio.run(main())
