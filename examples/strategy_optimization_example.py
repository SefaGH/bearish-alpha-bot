#!/usr/bin/env python3
"""
Example: Phase 4.3 AI-Powered Strategy Optimization

Demonstrates genetic algorithms, multi-objective optimization,
and neural architecture search for trading strategy improvement.
"""

import asyncio
import logging
import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.strategy_optimizer import StrategyOptimizer, OptimizationResult
from ml.genetic_optimizer import GeneticOptimizer
from ml.multi_objective_optimizer import MultiObjectiveOptimizer
from ml.neural_architecture_search import NeuralArchitectureSearch
from config.optimization_config import OptimizationConfiguration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_backtest_results(params: dict) -> dict:
    """
    Simulate backtest results for strategy parameters.
    
    In real usage, this would run actual backtests.
    """
    # Simulate performance based on parameters
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    stop_loss_pct = params.get('stop_loss_pct', 0.02)
    take_profit_pct = params.get('take_profit_pct', 0.05)
    
    # Simulate metrics (in reality, these come from actual backtesting)
    # Better RSI periods and tighter risk parameters generally perform better
    base_return = 0.1
    rsi_factor = 1.0 + (14 - abs(rsi_period - 14)) * 0.01
    risk_factor = 1.0 + (take_profit_pct / stop_loss_pct) * 0.05
    
    total_return = base_return * rsi_factor * risk_factor + np.random.normal(0, 0.02)
    sharpe_ratio = 1.0 + rsi_factor * 0.3 + np.random.normal(0, 0.1)
    max_drawdown = 0.15 + (stop_loss_pct * 2) + np.random.uniform(0, 0.05)
    win_rate = 0.5 + (rsi_oversold - 25) * 0.01 + np.random.normal(0, 0.05)
    profit_factor = 1.5 + rsi_factor * 0.2 + np.random.normal(0, 0.1)
    
    return {
        'total_return': max(0.0, total_return),
        'sharpe_ratio': max(0.0, sharpe_ratio),
        'max_drawdown': min(1.0, max(0.0, max_drawdown)),
        'win_rate': min(1.0, max(0.0, win_rate)),
        'profit_factor': max(1.0, profit_factor)
    }


async def demonstrate_genetic_optimization():
    """Demonstrate genetic algorithm for strategy optimization."""
    logger.info("\n" + "="*70)
    logger.info("1. Genetic Algorithm Optimization")
    logger.info("="*70)
    
    # Configuration
    config = OptimizationConfiguration.get_default_config()
    config.genetic_algorithm.population_size = 20
    config.genetic_algorithm.num_generations = 15
    config.genetic_algorithm.elite_size = 3
    
    # Define parameter space
    parameter_space = {
        'rsi_period': (7, 21, 'int'),
        'rsi_oversold': (20, 35, 'int'),
        'rsi_overbought': (65, 80, 'int'),
        'stop_loss_pct': (0.01, 0.05, 'float'),
        'take_profit_pct': (0.02, 0.1, 'float'),
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(config)
    
    # Define fitness function
    def fitness_function(params):
        results = simulate_backtest_results(params)
        # Optimize for Sharpe ratio with return consideration
        fitness = results['sharpe_ratio'] * 0.6 + results['total_return'] * 0.4
        return fitness
    
    logger.info("\n[Starting Genetic Algorithm]")
    logger.info(f"Population size: {config.genetic_algorithm.population_size}")
    logger.info(f"Generations: {config.genetic_algorithm.num_generations}")
    logger.info(f"Parameter space: {len(parameter_space)} parameters")
    
    # Run optimization
    result = optimizer.optimize_strategy_parameters(
        parameter_space,
        fitness_function,
        method='genetic',
        verbose=True
    )
    
    logger.info("\n[Optimization Results]")
    logger.info(f"Best fitness: {result.best_score:.4f}")
    logger.info(f"Optimization time: {result.optimization_time:.2f}s")
    logger.info(f"Generations completed: {result.generations_completed}")
    logger.info("\nBest parameters:")
    for param, value in result.best_parameters.items():
        logger.info(f"  {param}: {value}")
    
    # Show improvement over time
    if result.fitness_history:
        logger.info("\n[Fitness Progression]")
        for i in [0, len(result.fitness_history)//2, -1]:
            logger.info(f"Generation {i}: {result.fitness_history[i]:.4f}")
        improvement = result.fitness_history[-1] - result.fitness_history[0]
        logger.info(f"Total improvement: {improvement:.4f} ({improvement/result.fitness_history[0]*100:.1f}%)")


async def demonstrate_multi_objective_optimization():
    """Demonstrate multi-objective optimization for balanced performance."""
    logger.info("\n" + "="*70)
    logger.info("2. Multi-Objective Optimization (NSGA-II)")
    logger.info("="*70)
    
    # Configuration
    config = OptimizationConfiguration.get_default_config()
    config.genetic_algorithm.population_size = 30
    config.genetic_algorithm.num_generations = 20
    config.multi_objective.objectives = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # Parameter space
    parameter_space = {
        'rsi_period': (10, 20, 'int'),
        'rsi_oversold': (25, 35, 'int'),
        'stop_loss_pct': (0.015, 0.04, 'float'),
        'take_profit_pct': (0.03, 0.08, 'float'),
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(config)
    
    # Create objective functions
    def mock_backtest(params):
        return simulate_backtest_results(params)
    
    objective_functions = optimizer.create_multi_objective_functions(
        mock_backtest,
        base_config={}
    )
    
    logger.info("\n[Starting Multi-Objective Optimization]")
    logger.info(f"Objectives: {config.multi_objective.objectives}")
    logger.info(f"Population size: {config.genetic_algorithm.population_size}")
    
    # Run optimization
    result = optimizer.optimize_strategy_parameters(
        parameter_space,
        objective_functions,
        method='multi_objective',
        verbose=True
    )
    
    logger.info("\n[Optimization Results]")
    logger.info(f"Pareto front size: {len(result.pareto_front) if result.pareto_front else 0}")
    logger.info(f"Best compromise fitness: {result.best_score:.4f}")
    logger.info(f"Optimization time: {result.optimization_time:.2f}s")
    
    logger.info("\nBest compromise parameters:")
    for param, value in result.best_parameters.items():
        logger.info(f"  {param}: {value}")
    
    # Show Pareto front solutions
    if result.pareto_front:
        logger.info("\n[Pareto Front Solutions (Top 3)]")
        for i, solution in enumerate(result.pareto_front[:3], 1):
            logger.info(f"\nSolution {i}:")
            logger.info(f"  Parameters: {solution['parameters']}")
            logger.info(f"  Objectives: {solution['objectives']}")


async def demonstrate_neural_architecture_search():
    """Demonstrate neural architecture search for model optimization."""
    logger.info("\n" + "="*70)
    logger.info("3. Neural Architecture Search")
    logger.info("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    n_classes = 3
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_val = np.random.randn(100, n_features)
    y_val = np.random.randint(0, n_classes, 100)
    
    # Configuration
    config = OptimizationConfiguration.get_default_config()
    config.neural_architecture_search.num_architectures = 12
    config.neural_architecture_search.evaluation_epochs = 5
    
    # Create NAS
    nas = NeuralArchitectureSearch(
        config.neural_architecture_search,
        input_size=n_features,
        output_size=n_classes
    )
    
    logger.info("\n[Starting Neural Architecture Search]")
    logger.info(f"Input size: {n_features}")
    logger.info(f"Output size: {n_classes}")
    logger.info(f"Training samples: {n_samples}")
    logger.info(f"Architectures to evaluate: {config.neural_architecture_search.num_architectures}")
    
    # Run search
    best_architecture = nas.search(
        X_train, y_train, X_val, y_val,
        population_size=8,
        num_generations=5,
        verbose=True
    )
    
    # Get summary
    summary = nas.get_search_summary()
    
    logger.info("\n[NAS Results]")
    logger.info(f"Best validation accuracy: {summary['best_score']:.4f}")
    logger.info(f"Best architecture:")
    logger.info(f"  Layers: {best_architecture.num_layers}")
    logger.info(f"  Layer sizes: {best_architecture.layer_sizes}")
    logger.info(f"  Activations: {best_architecture.activations}")
    logger.info(f"  Dropout rates: {best_architecture.dropout_rates}")
    logger.info(f"  Batch normalization: {best_architecture.use_batch_norm}")
    logger.info(f"  Parameter count: {summary['parameter_count']:,}")


async def demonstrate_optimization_comparison():
    """Compare different optimization methods."""
    logger.info("\n" + "="*70)
    logger.info("4. Optimization Methods Comparison")
    logger.info("="*70)
    
    config = OptimizationConfiguration.get_default_config()
    config.genetic_algorithm.population_size = 15
    config.genetic_algorithm.num_generations = 10
    
    parameter_space = {
        'rsi_period': (10, 20, 'int'),
        'stop_loss_pct': (0.02, 0.05, 'float'),
    }
    
    optimizer = StrategyOptimizer(config)
    
    def fitness_fn(params):
        results = simulate_backtest_results(params)
        return results['sharpe_ratio']
    
    # Run genetic optimization
    logger.info("\n[Running Genetic Algorithm]")
    result_ga = optimizer.optimize_strategy_parameters(
        parameter_space,
        fitness_fn,
        method='genetic',
        verbose=False
    )
    
    logger.info(f"✓ Genetic Algorithm: Score={result_ga.best_score:.4f}, Time={result_ga.optimization_time:.2f}s")
    
    # Compare methods
    logger.info("\n[Method Comparison]")
    comparison = optimizer.compare_optimization_methods()
    
    for method, stats in comparison.items():
        if stats['num_runs'] > 0:
            logger.info(f"\n{method.upper()}:")
            logger.info(f"  Runs: {stats['num_runs']}")
            logger.info(f"  Best score: {stats['best_score']:.4f}")
            logger.info(f"  Avg score: {stats['avg_score']:.4f}")
            logger.info(f"  Avg time: {stats['avg_time']:.2f}s")


async def main():
    """Main demonstration function."""
    logger.info("="*70)
    logger.info("Phase 4.3: AI-Powered Strategy Optimization Demo")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run demonstrations
        await demonstrate_genetic_optimization()
        await demonstrate_multi_objective_optimization()
        await demonstrate_neural_architecture_search()
        await demonstrate_optimization_comparison()
        
        logger.info("\n" + "="*70)
        logger.info("✓ All demonstrations completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
