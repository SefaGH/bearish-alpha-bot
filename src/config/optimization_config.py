"""
Optimization Configuration for Phase 4.3: AI-Powered Strategy Optimization.

Defines configuration for genetic algorithms, neural architecture search,
and multi-objective optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm optimization."""
    
    # Population parameters
    population_size: int = 50
    num_generations: int = 100
    elite_size: int = 5  # Number of best individuals to preserve
    
    # Selection parameters
    selection_method: str = 'tournament'  # 'tournament', 'roulette', 'rank'
    tournament_size: int = 3
    
    # Crossover parameters
    crossover_method: str = 'uniform'  # 'uniform', 'single_point', 'multi_point'
    crossover_rate: float = 0.8
    
    # Mutation parameters
    mutation_method: str = 'gaussian'  # 'gaussian', 'uniform', 'adaptive'
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2  # Standard deviation for gaussian mutation
    
    # Diversity preservation
    diversity_pressure: float = 0.1  # Penalty for similar individuals
    adaptive_mutation: bool = True  # Increase mutation when diversity is low
    
    # Convergence criteria
    early_stopping: bool = True
    convergence_threshold: float = 1e-4  # Stop if best fitness doesn't improve
    patience: int = 20  # Generations to wait before stopping
    
    # Logging
    log_frequency: int = 1  # Log every N generations


@dataclass
class NeuralArchitectureSearchConfig:
    """Configuration for neural architecture search."""
    
    # Search space
    min_layers: int = 2
    max_layers: int = 8
    layer_size_options: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    activation_options: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid'])
    dropout_options: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Search parameters
    search_method: str = 'genetic'  # 'genetic', 'random', 'bayesian'
    num_architectures: int = 30  # Number of architectures to evaluate
    max_search_time: int = 3600  # Maximum search time in seconds
    
    # Performance estimation
    quick_evaluation: bool = True  # Use subset of data for faster evaluation
    evaluation_epochs: int = 10  # Epochs for architecture evaluation
    validation_split: float = 0.2
    
    # Architecture constraints
    max_parameters: int = 1_000_000  # Maximum model parameters
    min_performance: float = 0.5  # Minimum acceptable accuracy


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    
    # Objectives to optimize
    objectives: List[str] = field(default_factory=lambda: [
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate'
    ])
    
    # Objective directions (True = maximize, False = minimize)
    maximize_objectives: Dict[str, bool] = field(default_factory=lambda: {
        'total_return': True,
        'sharpe_ratio': True,
        'max_drawdown': False,  # Minimize drawdown
        'win_rate': True,
        'profit_factor': True,
        'volatility': False  # Minimize volatility
    })
    
    # Objective weights (for weighted sum method)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'total_return': 0.3,
        'sharpe_ratio': 0.3,
        'max_drawdown': 0.2,
        'win_rate': 0.2
    })
    
    # NSGA-II parameters
    use_nsga2: bool = True
    crowding_distance_weight: float = 0.5
    pareto_front_size: int = 20  # Number of solutions to keep on Pareto front
    
    # Constraints
    min_acceptable_return: float = 0.0  # Minimum return threshold
    max_acceptable_drawdown: float = 0.5  # Maximum 50% drawdown
    min_win_rate: float = 0.4  # Minimum 40% win rate


@dataclass
class StrategyParameterSpace:
    """Definition of parameter search space for strategy optimization."""
    
    # Parameter bounds (name: (min, max, type))
    parameter_bounds: Dict[str, Tuple[Any, Any, str]] = field(default_factory=lambda: {
        # RSI parameters
        'rsi_period': (7, 21, 'int'),
        'rsi_oversold': (15, 35, 'int'),
        'rsi_overbought': (65, 85, 'int'),
        
        # EMA parameters
        'ema_fast': (5, 30, 'int'),
        'ema_slow': (20, 100, 'int'),
        'ema_trend': (50, 250, 'int'),
        
        # Position sizing
        'position_size': (0.01, 0.1, 'float'),
        'max_position_pct': (0.05, 0.25, 'float'),
        
        # Risk management
        'stop_loss_pct': (0.01, 0.1, 'float'),
        'take_profit_pct': (0.02, 0.2, 'float'),
        'trailing_stop_pct': (0.005, 0.05, 'float'),
        
        # Entry/Exit thresholds
        'entry_threshold': (0.5, 2.0, 'float'),
        'exit_threshold': (0.5, 2.0, 'float'),
        
        # Volume filters
        'min_volume': (100000, 10000000, 'float'),
        'volume_ma_period': (10, 50, 'int'),
    })
    
    # Discrete parameter choices
    parameter_choices: Dict[str, List[Any]] = field(default_factory=lambda: {
        'timeframe': ['1m', '5m', '15m', '1h', '4h'],
        'regime_filter': [True, False],
        'use_trailing_stop': [True, False],
        'entry_type': ['limit', 'market'],
    })


@dataclass
class OptimizationConfig:
    """Master configuration for all optimization methods."""
    
    genetic_algorithm: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    neural_architecture_search: NeuralArchitectureSearchConfig = field(default_factory=NeuralArchitectureSearchConfig)
    multi_objective: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    parameter_space: StrategyParameterSpace = field(default_factory=StrategyParameterSpace)
    
    # General optimization settings
    n_parallel_evaluations: int = 4  # Number of parallel fitness evaluations
    cache_evaluations: bool = True  # Cache fitness evaluations
    save_checkpoints: bool = True  # Save optimization progress
    checkpoint_frequency: int = 10  # Save every N generations
    
    # Logging
    verbose: bool = True
    log_frequency: int = 1  # Log every N generations
    plot_progress: bool = False  # Generate progress plots (requires matplotlib)


class OptimizationConfiguration:
    """Factory class for optimization configurations."""
    
    @classmethod
    def get_default_config(cls) -> OptimizationConfig:
        """Get default optimization configuration."""
        return OptimizationConfig()
    
    @classmethod
    def get_genetic_config(cls) -> GeneticAlgorithmConfig:
        """Get genetic algorithm configuration."""
        return GeneticAlgorithmConfig()
    
    @classmethod
    def get_nas_config(cls) -> NeuralArchitectureSearchConfig:
        """Get neural architecture search configuration."""
        return NeuralArchitectureSearchConfig()
    
    @classmethod
    def get_multi_objective_config(cls) -> MultiObjectiveConfig:
        """Get multi-objective optimization configuration."""
        return MultiObjectiveConfig()
    
    @classmethod
    def get_parameter_space(cls) -> StrategyParameterSpace:
        """Get strategy parameter search space."""
        return StrategyParameterSpace()
    
    @classmethod
    def create_custom_config(cls, **kwargs) -> OptimizationConfig:
        """
        Create custom optimization configuration.
        
        Args:
            **kwargs: Configuration parameters to override
            
        Returns:
            Custom optimization configuration
        """
        config = cls.get_default_config()
        
        # Update configuration with provided parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
