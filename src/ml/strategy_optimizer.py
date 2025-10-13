"""
Strategy Optimization Coordinator.

Unified interface for strategy parameter optimization using genetic algorithms,
multi-objective optimization, and neural architecture search.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
import time

from .genetic_optimizer import GeneticOptimizer, Individual
from .multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveIndividual
from .neural_architecture_search import NeuralArchitectureSearch, NetworkArchitecture

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from strategy optimization."""
    
    method: str  # 'genetic', 'multi_objective', 'nas'
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_time: float
    generations_completed: int
    
    # For multi-objective
    pareto_front: Optional[List[Dict[str, Any]]] = None
    
    # For NAS
    best_architecture: Optional[Dict[str, Any]] = None
    
    # Metrics
    fitness_history: Optional[List[float]] = None
    diversity_history: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'method': self.method,
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_time': self.optimization_time,
            'generations_completed': self.generations_completed
        }
        
        if self.pareto_front:
            result['pareto_front'] = self.pareto_front
        
        if self.best_architecture:
            result['best_architecture'] = self.best_architecture
        
        if self.fitness_history:
            result['fitness_history'] = self.fitness_history
        
        if self.diversity_history:
            result['diversity_history'] = self.diversity_history
        
        return result


class StrategyOptimizer:
    """
    Unified strategy optimization coordinator.
    
    Provides a single interface for:
    - Single-objective genetic algorithm optimization
    - Multi-objective Pareto optimization
    - Neural architecture search for ML models
    """
    
    def __init__(self, config):
        """
        Initialize strategy optimizer.
        
        Args:
            config: OptimizationConfig instance
        """
        self.config = config
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("Initialized StrategyOptimizer")
    
    def optimize_strategy_parameters(self, 
                                    parameter_space: Dict[str, Tuple[Any, Any, str]],
                                    fitness_function: Callable[[Dict[str, Any]], float],
                                    method: str = 'genetic',
                                    verbose: bool = True) -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            parameter_space: Dictionary of parameter bounds
            fitness_function: Function to evaluate parameter sets
            method: Optimization method ('genetic' or 'multi_objective')
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult with best parameters
        """
        start_time = time.time()
        
        if method == 'genetic':
            result = self._optimize_genetic(
                parameter_space, fitness_function, verbose
            )
        elif method == 'multi_objective':
            result = self._optimize_multi_objective(
                parameter_space, fitness_function, verbose
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - start_time
        result.optimization_time = optimization_time
        
        # Store in history
        self.optimization_history.append(result)
        
        logger.info(f"Optimization complete in {optimization_time:.2f}s")
        return result
    
    def _optimize_genetic(self,
                         parameter_space: Dict[str, Tuple[Any, Any, str]],
                         fitness_function: Callable[[Dict[str, Any]], float],
                         verbose: bool) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        logger.info("Running genetic algorithm optimization")
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            self.config.genetic_algorithm,
            parameter_space
        )
        
        # Run evolution
        best_individual = optimizer.evolve(fitness_function, verbose=verbose)
        
        # Get summary
        summary = optimizer.get_optimization_summary()
        
        result = OptimizationResult(
            method='genetic',
            best_parameters=best_individual.genome,
            best_score=best_individual.fitness,
            optimization_time=0.0,  # Will be set by caller
            generations_completed=optimizer.generation,
            fitness_history=summary['fitness_history'],
            diversity_history=summary['diversity_history']
        )
        
        return result
    
    def _optimize_multi_objective(self,
                                  parameter_space: Dict[str, Tuple[Any, Any, str]],
                                  objective_functions: Dict[str, Callable],
                                  verbose: bool) -> OptimizationResult:
        """Run multi-objective optimization."""
        logger.info("Running multi-objective optimization (NSGA-II)")
        
        # Get objectives from config
        objectives = self.config.multi_objective.objectives
        maximize_objectives = self.config.multi_objective.maximize_objectives
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            self.config.genetic_algorithm,  # Uses genetic algorithm config
            parameter_space,
            objectives,
            maximize_objectives
        )
        
        # Run evolution
        pareto_front = optimizer.evolve_multi_objective(
            objective_functions,
            verbose=verbose
        )
        
        # Get best compromise solution
        best_compromise = optimizer.get_best_compromise_solution()
        
        # Get summary
        summary = optimizer.get_optimization_summary()
        
        result = OptimizationResult(
            method='multi_objective',
            best_parameters=best_compromise.genome if best_compromise else {},
            best_score=best_compromise.fitness if best_compromise else 0.0,
            optimization_time=0.0,  # Will be set by caller
            generations_completed=optimizer.generation,
            pareto_front=summary.get('pareto_solutions', []),
            fitness_history=summary.get('fitness_history', [])
        )
        
        return result
    
    def optimize_neural_architecture(self,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray,
                                    input_size: int,
                                    output_size: int,
                                    verbose: bool = True) -> OptimizationResult:
        """
        Optimize neural network architecture.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_size: Input dimension
            output_size: Output dimension
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult with best architecture
        """
        logger.info("Running neural architecture search")
        
        start_time = time.time()
        
        # Create NAS
        nas = NeuralArchitectureSearch(
            self.config.neural_architecture_search,
            input_size,
            output_size
        )
        
        # Run search
        best_architecture = nas.search(
            X_train, y_train, X_val, y_val,
            population_size=self.config.neural_architecture_search.num_architectures // 3,
            num_generations=10,
            verbose=verbose
        )
        
        # Get summary
        summary = nas.get_search_summary()
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            method='nas',
            best_parameters={},  # Architecture is stored separately
            best_score=summary['best_score'],
            optimization_time=optimization_time,
            generations_completed=summary['total_generations'],
            best_architecture=summary['best_architecture']
        )
        
        # Store in history
        self.optimization_history.append(result)
        
        logger.info(f"NAS complete in {optimization_time:.2f}s")
        return result
    
    def create_backtest_fitness_function(self,
                                        backtest_runner: Callable,
                                        base_config: Dict[str, Any],
                                        objective: str = 'sharpe_ratio') -> Callable:
        """
        Create fitness function based on backtest results.
        
        Args:
            backtest_runner: Function that runs backtest with config
            base_config: Base configuration to modify
            objective: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Fitness function that takes parameters and returns score
        """
        def fitness_function(parameters: Dict[str, Any]) -> float:
            """Evaluate parameters by running backtest."""
            try:
                # Merge parameters with base config
                config = base_config.copy()
                config.update(parameters)
                
                # Run backtest
                results = backtest_runner(config)
                
                # Extract objective
                if objective == 'sharpe_ratio':
                    score = results.get('sharpe_ratio', 0.0)
                elif objective == 'total_return':
                    score = results.get('total_return', 0.0)
                elif objective == 'profit_factor':
                    score = results.get('profit_factor', 1.0)
                elif objective == 'win_rate':
                    score = results.get('win_rate', 0.0)
                elif objective == 'max_drawdown':
                    # Minimize drawdown (return negative)
                    score = -results.get('max_drawdown', 1.0)
                else:
                    # Default: weighted combination
                    sharpe = results.get('sharpe_ratio', 0.0)
                    returns = results.get('total_return', 0.0)
                    drawdown = results.get('max_drawdown', 1.0)
                    score = sharpe * 0.5 + returns * 0.3 - drawdown * 0.2
                
                return score
                
            except Exception as e:
                logger.error(f"Error in fitness evaluation: {e}")
                return 0.0
        
        return fitness_function
    
    def create_multi_objective_functions(self,
                                        backtest_runner: Callable,
                                        base_config: Dict[str, Any]
                                        ) -> Dict[str, Callable]:
        """
        Create multiple objective functions for multi-objective optimization.
        
        Args:
            backtest_runner: Function that runs backtest with config
            base_config: Base configuration to modify
            
        Returns:
            Dictionary of objective functions
        """
        def create_objective_fn(metric_name: str) -> Callable:
            """Create objective function for a specific metric."""
            def objective_fn(parameters: Dict[str, Any]) -> float:
                try:
                    config = base_config.copy()
                    config.update(parameters)
                    results = backtest_runner(config)
                    return results.get(metric_name, 0.0)
                except Exception as e:
                    logger.error(f"Error evaluating {metric_name}: {e}")
                    return 0.0
            return objective_fn
        
        objectives = {}
        for metric in self.config.multi_objective.objectives:
            objectives[metric] = create_objective_fn(metric)
        
        return objectives
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all optimizations."""
        return [result.to_dict() for result in self.optimization_history]
    
    def get_best_parameters_by_method(self, method: str) -> Optional[Dict[str, Any]]:
        """
        Get best parameters found by a specific method.
        
        Args:
            method: Optimization method ('genetic', 'multi_objective', 'nas')
            
        Returns:
            Best parameters or None if not found
        """
        method_results = [r for r in self.optimization_history if r.method == method]
        
        if not method_results:
            return None
        
        best_result = max(method_results, key=lambda r: r.best_score)
        return best_result.best_parameters
    
    def compare_optimization_methods(self) -> Dict[str, Any]:
        """
        Compare performance of different optimization methods.
        
        Returns:
            Comparison statistics
        """
        comparison = {}
        
        for method in ['genetic', 'multi_objective', 'nas']:
            method_results = [r for r in self.optimization_history if r.method == method]
            
            if method_results:
                scores = [r.best_score for r in method_results]
                times = [r.optimization_time for r in method_results]
                
                comparison[method] = {
                    'num_runs': len(method_results),
                    'best_score': max(scores),
                    'avg_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'avg_time': np.mean(times),
                    'total_time': sum(times)
                }
            else:
                comparison[method] = {
                    'num_runs': 0,
                    'best_score': 0.0,
                    'avg_score': 0.0,
                    'std_score': 0.0,
                    'avg_time': 0.0,
                    'total_time': 0.0
                }
        
        return comparison
