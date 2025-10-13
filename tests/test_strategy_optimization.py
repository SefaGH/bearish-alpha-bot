"""
Tests for Phase 4.3: AI-Powered Strategy Optimization

Tests genetic algorithms, multi-objective optimization, neural architecture search,
and the unified strategy optimizer.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from src.ml.genetic_optimizer import GeneticOptimizer, Individual
from src.ml.multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveIndividual
from src.ml.neural_architecture_search import NeuralArchitectureSearch, NetworkArchitecture
from src.ml.strategy_optimizer import StrategyOptimizer, OptimizationResult
from src.config.optimization_config import (
    GeneticAlgorithmConfig,
    MultiObjectiveConfig,
    NeuralArchitectureSearchConfig,
    OptimizationConfig,
    StrategyParameterSpace
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test fixtures
@pytest.fixture
def simple_parameter_space():
    """Simple parameter space for testing."""
    return {
        'param1': (0, 10, 'int'),
        'param2': (0.0, 1.0, 'float'),
        'param3': (True, False, 'bool'),
    }


@pytest.fixture
def genetic_config():
    """Create genetic algorithm configuration."""
    config = GeneticAlgorithmConfig()
    config.population_size = 10
    config.num_generations = 5
    config.elite_size = 2
    config.verbose = False
    config.log_frequency = 1
    return config


@pytest.fixture
def multi_objective_config():
    """Create multi-objective configuration."""
    return MultiObjectiveConfig()


@pytest.fixture
def nas_config():
    """Create NAS configuration."""
    config = NeuralArchitectureSearchConfig()
    config.num_architectures = 6
    config.max_search_time = 60
    return config


@pytest.fixture
def optimization_config():
    """Create full optimization configuration."""
    return OptimizationConfig()


# Test GeneticOptimizer
class TestGeneticOptimizer:
    """Test genetic algorithm optimizer."""
    
    def test_initialization(self, genetic_config, simple_parameter_space):
        """Test optimizer initialization."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        
        assert optimizer.config == genetic_config
        assert optimizer.parameter_space == simple_parameter_space
        assert optimizer.generation == 0
        assert len(optimizer.population) == 0
    
    def test_random_genome_generation(self, genetic_config, simple_parameter_space):
        """Test random genome generation."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        genome = optimizer._random_genome()
        
        assert 'param1' in genome
        assert 'param2' in genome
        assert 'param3' in genome
        assert isinstance(genome['param1'], int)
        assert isinstance(genome['param2'], float)
        assert isinstance(genome['param3'], bool)
        assert 0 <= genome['param1'] <= 10
        assert 0.0 <= genome['param2'] <= 1.0
    
    def test_population_initialization(self, genetic_config, simple_parameter_space):
        """Test population initialization."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        population = optimizer.initialize_population()
        
        assert len(population) == genetic_config.population_size
        assert all(isinstance(ind, Individual) for ind in population)
        assert all(ind.fitness == 0.0 for ind in population)
    
    def test_fitness_evaluation(self, genetic_config, simple_parameter_space):
        """Test fitness evaluation."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        individual = Individual(genome={'param1': 5, 'param2': 0.5, 'param3': True})
        
        def fitness_fn(genome):
            return genome['param1'] + genome['param2']
        
        fitness = optimizer.evaluate_fitness(individual, fitness_fn)
        
        assert fitness == 5.5
        assert individual.fitness == 5.5
    
    def test_tournament_selection(self, genetic_config, simple_parameter_space):
        """Test tournament selection."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        
        # Create population with different fitness
        population = []
        for i in range(5):
            ind = Individual(genome={'param1': i, 'param2': 0.5, 'param3': True})
            ind.fitness = float(i)
            population.append(ind)
        
        optimizer.population = population
        
        # Select parents
        parents = optimizer._tournament_selection(population, n_parents=2)
        
        assert len(parents) == 2
        assert all(isinstance(p, Individual) for p in parents)
    
    def test_uniform_crossover(self, genetic_config, simple_parameter_space):
        """Test uniform crossover."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        
        parent1 = Individual(genome={'param1': 1, 'param2': 0.1, 'param3': True})
        parent2 = Individual(genome={'param1': 10, 'param2': 0.9, 'param3': False})
        
        offspring1, offspring2 = optimizer._uniform_crossover(parent1, parent2)
        
        assert isinstance(offspring1, Individual)
        assert isinstance(offspring2, Individual)
        assert 'param1' in offspring1.genome
        assert 'param2' in offspring1.genome
        assert 'param3' in offspring1.genome
    
    def test_gaussian_mutation(self, genetic_config, simple_parameter_space):
        """Test gaussian mutation."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        
        individual = Individual(genome={'param1': 5, 'param2': 0.5, 'param3': True})
        mutated = optimizer.mutate(individual)
        
        assert isinstance(mutated, Individual)
        assert 'param1' in mutated.genome
        # Values should be within bounds
        assert 0 <= mutated.genome['param1'] <= 10
        assert 0.0 <= mutated.genome['param2'] <= 1.0
    
    def test_evolution(self, genetic_config, simple_parameter_space):
        """Test full evolution process."""
        optimizer = GeneticOptimizer(genetic_config, simple_parameter_space)
        
        # Simple fitness function: maximize param1 + param2
        def fitness_fn(genome):
            return genome['param1'] + genome['param2']
        
        best = optimizer.evolve(fitness_fn, verbose=False)
        
        assert isinstance(best, Individual)
        assert best.fitness > 0
        assert len(optimizer.fitness_history) == genetic_config.num_generations + 1
        # Best should be improving over time
        assert optimizer.fitness_history[-1] >= optimizer.fitness_history[0]


# Test MultiObjectiveOptimizer
class TestMultiObjectiveOptimizer:
    """Test multi-objective optimizer."""
    
    def test_initialization(self, genetic_config, simple_parameter_space):
        """Test multi-objective optimizer initialization."""
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        assert optimizer.objectives == objectives
        assert optimizer.maximize_objectives == maximize_objectives
        assert len(optimizer.pareto_front) == 0
    
    def test_dominance_check(self, genetic_config, simple_parameter_space):
        """Test Pareto dominance checking."""
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        ind1 = MultiObjectiveIndividual(
            genome={'param1': 1, 'param2': 0.5, 'param3': True},
            objective_scores={'obj1': 0.8, 'obj2': 0.6}
        )
        
        ind2 = MultiObjectiveIndividual(
            genome={'param1': 2, 'param2': 0.3, 'param3': False},
            objective_scores={'obj1': 0.5, 'obj2': 0.5}
        )
        
        # ind1 dominates ind2 (better in both objectives)
        assert ind1.dominates_other(ind2, maximize_objectives)
        assert not ind2.dominates_other(ind1, maximize_objectives)
    
    def test_non_dominated_sorting(self, genetic_config, simple_parameter_space):
        """Test non-dominated sorting."""
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        # Create population with known dominance relationships
        population = [
            MultiObjectiveIndividual(
                genome={'param1': i, 'param2': 0.5, 'param3': True},
                objective_scores={'obj1': i * 0.1, 'obj2': (10 - i) * 0.1}
            )
            for i in range(5)
        ]
        
        fronts = optimizer.fast_non_dominated_sort(population)
        
        assert len(fronts) > 0
        assert all(isinstance(front, list) for front in fronts)
        # All individuals should be assigned to a front
        total_individuals = sum(len(front) for front in fronts)
        assert total_individuals == len(population)
    
    def test_crowding_distance(self, genetic_config, simple_parameter_space):
        """Test crowding distance calculation."""
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        front = [
            MultiObjectiveIndividual(
                genome={'param1': i, 'param2': 0.5, 'param3': True},
                objective_scores={'obj1': i * 0.2, 'obj2': i * 0.1}
            )
            for i in range(5)
        ]
        
        optimizer.calculate_crowding_distance(front)
        
        # Boundary individuals should have infinite distance
        assert front[0].crowding_distance == float('inf') or front[-1].crowding_distance == float('inf')
        # Middle individuals should have finite distances
        assert all(ind.crowding_distance >= 0 for ind in front)
    
    def test_multi_objective_evolution(self, genetic_config, simple_parameter_space):
        """Test multi-objective evolution."""
        genetic_config.num_generations = 3  # Short test
        
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        # Create objective functions
        objective_functions = {
            'obj1': lambda genome: genome['param1'] + genome['param2'],
            'obj2': lambda genome: genome['param1'] * genome['param2']
        }
        
        pareto_front = optimizer.evolve_multi_objective(
            objective_functions, verbose=False
        )
        
        assert len(pareto_front) > 0
        assert all(isinstance(ind, MultiObjectiveIndividual) for ind in pareto_front)
        assert all(ind.rank == 0 for ind in pareto_front)
    
    def test_best_compromise_solution(self, genetic_config, simple_parameter_space):
        """Test finding best compromise solution."""
        objectives = ['obj1', 'obj2']
        maximize_objectives = {'obj1': True, 'obj2': True}
        
        optimizer = MultiObjectiveOptimizer(
            genetic_config, simple_parameter_space,
            objectives, maximize_objectives
        )
        
        # Create Pareto front
        optimizer.pareto_front = [
            MultiObjectiveIndividual(
                genome={'param1': i, 'param2': 0.5, 'param3': True},
                objective_scores={'obj1': i * 0.2, 'obj2': (5 - i) * 0.2}
            )
            for i in range(5)
        ]
        
        best = optimizer.get_best_compromise_solution()
        
        assert best is not None
        assert isinstance(best, MultiObjectiveIndividual)


# Test NeuralArchitectureSearch
class TestNeuralArchitectureSearch:
    """Test neural architecture search."""
    
    def test_initialization(self, nas_config):
        """Test NAS initialization."""
        nas = NeuralArchitectureSearch(nas_config, input_size=50, output_size=3)
        
        assert nas.input_size == 50
        assert nas.output_size == 3
        assert nas.best_architecture is None
    
    def test_random_architecture_generation(self, nas_config):
        """Test random architecture generation."""
        nas = NeuralArchitectureSearch(nas_config, input_size=50, output_size=3)
        
        arch = nas.random_architecture()
        
        assert isinstance(arch, NetworkArchitecture)
        assert nas_config.min_layers <= arch.num_layers <= nas_config.max_layers
        assert len(arch.layer_sizes) == arch.num_layers
        assert len(arch.activations) == arch.num_layers
        assert len(arch.dropout_rates) == arch.num_layers
    
    def test_architecture_parameter_count(self, nas_config):
        """Test architecture parameter counting."""
        arch = NetworkArchitecture(
            num_layers=2,
            layer_sizes=[64, 32],
            activations=['relu', 'relu'],
            dropout_rates=[0.1, 0.2]
        )
        
        param_count = arch.get_parameter_count(input_size=50, output_size=3)
        
        # 50*64 + 64 + 64*32 + 32 + 32*3 + 3
        expected = 50*64 + 64 + 64*32 + 32 + 32*3 + 3
        assert param_count == expected
    
    def test_architecture_serialization(self, nas_config):
        """Test architecture to/from dict."""
        arch = NetworkArchitecture(
            num_layers=2,
            layer_sizes=[64, 32],
            activations=['relu', 'tanh'],
            dropout_rates=[0.1, 0.2],
            use_batch_norm=True
        )
        
        arch_dict = arch.to_dict()
        arch_restored = NetworkArchitecture.from_dict(arch_dict)
        
        assert arch_restored.num_layers == arch.num_layers
        assert arch_restored.layer_sizes == arch.layer_sizes
        assert arch_restored.activations == arch.activations
        assert arch_restored.use_batch_norm == arch.use_batch_norm
    
    def test_architecture_mutation(self, nas_config):
        """Test architecture mutation."""
        nas = NeuralArchitectureSearch(nas_config, input_size=50, output_size=3)
        
        arch = NetworkArchitecture(
            num_layers=3,
            layer_sizes=[64, 32, 16],
            activations=['relu', 'relu', 'relu'],
            dropout_rates=[0.1, 0.1, 0.1]
        )
        
        mutated = nas.mutate_architecture(arch, mutation_rate=0.5)
        
        assert isinstance(mutated, NetworkArchitecture)
        assert mutated.num_layers >= nas_config.min_layers
        assert mutated.num_layers <= nas_config.max_layers
    
    def test_architecture_crossover(self, nas_config):
        """Test architecture crossover."""
        nas = NeuralArchitectureSearch(nas_config, input_size=50, output_size=3)
        
        parent1 = NetworkArchitecture(
            num_layers=2,
            layer_sizes=[64, 32],
            activations=['relu', 'relu'],
            dropout_rates=[0.1, 0.1]
        )
        
        parent2 = NetworkArchitecture(
            num_layers=3,
            layer_sizes=[128, 64, 32],
            activations=['tanh', 'tanh', 'tanh'],
            dropout_rates=[0.2, 0.2, 0.2]
        )
        
        offspring = nas.crossover_architectures(parent1, parent2)
        
        assert isinstance(offspring, NetworkArchitecture)
        assert offspring.num_layers >= nas_config.min_layers


# Test StrategyOptimizer
class TestStrategyOptimizer:
    """Test unified strategy optimizer."""
    
    def test_initialization(self, optimization_config):
        """Test strategy optimizer initialization."""
        optimizer = StrategyOptimizer(optimization_config)
        
        assert optimizer.config == optimization_config
        assert len(optimizer.optimization_history) == 0
    
    def test_genetic_optimization(self, optimization_config, simple_parameter_space):
        """Test genetic algorithm optimization."""
        optimization_config.genetic_algorithm.population_size = 10
        optimization_config.genetic_algorithm.num_generations = 3
        
        optimizer = StrategyOptimizer(optimization_config)
        
        def fitness_fn(params):
            return params['param1'] + params['param2']
        
        result = optimizer.optimize_strategy_parameters(
            simple_parameter_space,
            fitness_fn,
            method='genetic',
            verbose=False
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.method == 'genetic'
        assert result.best_score > 0
        assert result.optimization_time > 0
        assert len(result.best_parameters) > 0
    
    def test_backtest_fitness_function_creation(self, optimization_config):
        """Test backtest fitness function creation."""
        optimizer = StrategyOptimizer(optimization_config)
        
        def mock_backtest(config):
            return {
                'sharpe_ratio': 1.5,
                'total_return': 0.25,
                'max_drawdown': 0.15
            }
        
        fitness_fn = optimizer.create_backtest_fitness_function(
            mock_backtest,
            base_config={'base_param': 1},
            objective='sharpe_ratio'
        )
        
        score = fitness_fn({'test_param': 5})
        assert score == 1.5
    
    def test_multi_objective_functions_creation(self, optimization_config):
        """Test multi-objective functions creation."""
        optimizer = StrategyOptimizer(optimization_config)
        
        def mock_backtest(config):
            return {
                'total_return': 0.25,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.15,
                'win_rate': 0.6
            }
        
        objective_fns = optimizer.create_multi_objective_functions(
            mock_backtest,
            base_config={}
        )
        
        assert 'total_return' in objective_fns
        assert 'sharpe_ratio' in objective_fns
        
        # Test function execution
        score = objective_fns['sharpe_ratio']({'test': 1})
        assert score == 1.5
    
    def test_optimization_history(self, optimization_config, simple_parameter_space):
        """Test optimization history tracking."""
        optimization_config.genetic_algorithm.population_size = 5
        optimization_config.genetic_algorithm.num_generations = 2
        
        optimizer = StrategyOptimizer(optimization_config)
        
        def fitness_fn(params):
            return params['param1']
        
        # Run optimization
        result1 = optimizer.optimize_strategy_parameters(
            simple_parameter_space,
            fitness_fn,
            method='genetic',
            verbose=False
        )
        
        assert len(optimizer.optimization_history) == 1
        assert optimizer.optimization_history[0] == result1
        
        # Get history
        history = optimizer.get_optimization_history()
        assert len(history) == 1
        assert isinstance(history[0], dict)


# Test Integration
class TestOptimizationIntegration:
    """Test integration between optimization components."""
    
    def test_full_optimization_workflow(self, optimization_config, simple_parameter_space):
        """Test complete optimization workflow."""
        optimization_config.genetic_algorithm.population_size = 8
        optimization_config.genetic_algorithm.num_generations = 3
        
        optimizer = StrategyOptimizer(optimization_config)
        
        # Define a simple strategy evaluation
        def evaluate_strategy(params):
            # Simulate strategy performance
            performance = params['param1'] * 0.5 + params['param2'] * 2.0
            if params['param3']:
                performance *= 1.1
            return performance
        
        # Run optimization
        result = optimizer.optimize_strategy_parameters(
            simple_parameter_space,
            evaluate_strategy,
            method='genetic',
            verbose=False
        )
        
        assert result.best_score > 0
        assert 'param1' in result.best_parameters
        assert 'param2' in result.best_parameters
        assert 'param3' in result.best_parameters
        
        # Verify best parameters make sense
        assert result.best_parameters['param1'] >= 0
        assert 0.0 <= result.best_parameters['param2'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
