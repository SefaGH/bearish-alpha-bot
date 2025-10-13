"""
Genetic Algorithm Engine for Strategy Parameter Evolution.

Implements genetic algorithms for evolving trading strategy parameters
through selection, crossover, mutation, and fitness evaluation.
"""

import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    genome: Dict[str, Any]  # Parameter values
    fitness: float = 0.0  # Fitness score
    age: int = 0  # Generations survived
    objectives: Dict[str, float] = None  # Multi-objective scores
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = {}
    
    def clone(self) -> 'Individual':
        """Create a deep copy of the individual."""
        return Individual(
            genome=deepcopy(self.genome),
            fitness=self.fitness,
            age=self.age,
            objectives=deepcopy(self.objectives)
        )


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for trading strategy parameters.
    
    Implements evolutionary optimization with:
    - Tournament and roulette wheel selection
    - Uniform, single-point, and multi-point crossover
    - Gaussian and uniform mutation
    - Elitism and diversity preservation
    """
    
    def __init__(self, config, parameter_space: Dict[str, Tuple[Any, Any, str]]):
        """
        Initialize genetic optimizer.
        
        Args:
            config: GeneticAlgorithmConfig instance
            parameter_space: Dictionary of parameter bounds (name: (min, max, type))
        """
        self.config = config
        self.parameter_space = parameter_space
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        logger.info(f"Initialized GeneticOptimizer with population_size={config.population_size}")
    
    def initialize_population(self) -> List[Individual]:
        """
        Create initial random population.
        
        Returns:
            List of random individuals
        """
        population = []
        
        for _ in range(self.config.population_size):
            genome = self._random_genome()
            individual = Individual(genome=genome)
            population.append(individual)
        
        logger.info(f"Initialized population of {len(population)} individuals")
        return population
    
    def _random_genome(self) -> Dict[str, Any]:
        """Generate random genome within parameter bounds."""
        genome = {}
        
        for param_name, (min_val, max_val, param_type) in self.parameter_space.items():
            if param_type == 'int':
                genome[param_name] = random.randint(int(min_val), int(max_val))
            elif param_type == 'float':
                genome[param_name] = random.uniform(float(min_val), float(max_val))
            elif param_type == 'bool':
                genome[param_name] = random.choice([True, False])
            else:
                # For discrete choices
                genome[param_name] = random.choice([min_val, max_val])
        
        return genome
    
    def evaluate_fitness(self, individual: Individual, 
                        fitness_function: Callable[[Dict[str, Any]], float]) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            fitness_function: Function that takes genome and returns fitness score
            
        Returns:
            Fitness score
        """
        try:
            fitness = fitness_function(individual.genome)
            individual.fitness = fitness
            return fitness
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            individual.fitness = 0.0
            return 0.0
    
    def select_parents(self, population: List[Individual], 
                      n_parents: int = 2) -> List[Individual]:
        """
        Select parents for reproduction.
        
        Args:
            population: Current population
            n_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if self.config.selection_method == 'tournament':
            return self._tournament_selection(population, n_parents)
        elif self.config.selection_method == 'roulette':
            return self._roulette_wheel_selection(population, n_parents)
        elif self.config.selection_method == 'rank':
            return self._rank_selection(population, n_parents)
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def _tournament_selection(self, population: List[Individual], 
                             n_parents: int) -> List[Individual]:
        """Tournament selection: randomly select k individuals and choose the best."""
        parents = []
        
        for _ in range(n_parents):
            tournament = random.sample(population, self.config.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_wheel_selection(self, population: List[Individual],
                                  n_parents: int) -> List[Individual]:
        """Roulette wheel selection: probability proportional to fitness."""
        # Ensure all fitness values are positive
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            adjusted_fitness = [ind.fitness - min_fitness + 1e-6 for ind in population]
        else:
            adjusted_fitness = [ind.fitness + 1e-6 for ind in population]
        
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        parents = np.random.choice(
            population, 
            size=n_parents, 
            replace=False,
            p=probabilities
        ).tolist()
        
        return parents
    
    def _rank_selection(self, population: List[Individual],
                       n_parents: int) -> List[Individual]:
        """Rank selection: probability based on rank rather than fitness."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        parents = np.random.choice(
            sorted_pop,
            size=n_parents,
            replace=False,
            p=probabilities
        ).tolist()
        
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if random.random() > self.config.crossover_rate:
            # No crossover, return clones
            return parent1.clone(), parent2.clone()
        
        if self.config.crossover_method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == 'multi_point':
            return self._multi_point_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.config.crossover_method}")
    
    def _uniform_crossover(self, parent1: Individual, 
                          parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover: randomly inherit each gene from either parent."""
        genome1 = {}
        genome2 = {}
        
        for param_name in parent1.genome.keys():
            if random.random() < 0.5:
                genome1[param_name] = parent1.genome[param_name]
                genome2[param_name] = parent2.genome[param_name]
            else:
                genome1[param_name] = parent2.genome[param_name]
                genome2[param_name] = parent1.genome[param_name]
        
        offspring1 = Individual(genome=genome1)
        offspring2 = Individual(genome=genome2)
        
        return offspring1, offspring2
    
    def _single_point_crossover(self, parent1: Individual,
                               parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover: split at one random point."""
        genes = list(parent1.genome.keys())
        crossover_point = random.randint(1, len(genes) - 1)
        
        genome1 = {}
        genome2 = {}
        
        for i, param_name in enumerate(genes):
            if i < crossover_point:
                genome1[param_name] = parent1.genome[param_name]
                genome2[param_name] = parent2.genome[param_name]
            else:
                genome1[param_name] = parent2.genome[param_name]
                genome2[param_name] = parent1.genome[param_name]
        
        offspring1 = Individual(genome=genome1)
        offspring2 = Individual(genome=genome2)
        
        return offspring1, offspring2
    
    def _multi_point_crossover(self, parent1: Individual,
                              parent2: Individual) -> Tuple[Individual, Individual]:
        """Multi-point crossover: split at multiple random points."""
        genes = list(parent1.genome.keys())
        n_points = min(3, len(genes) - 1)  # Maximum 3 crossover points
        crossover_points = sorted(random.sample(range(1, len(genes)), n_points))
        
        genome1 = {}
        genome2 = {}
        
        use_parent1 = True
        point_idx = 0
        
        for i, param_name in enumerate(genes):
            if point_idx < len(crossover_points) and i >= crossover_points[point_idx]:
                use_parent1 = not use_parent1
                point_idx += 1
            
            if use_parent1:
                genome1[param_name] = parent1.genome[param_name]
                genome2[param_name] = parent2.genome[param_name]
            else:
                genome1[param_name] = parent2.genome[param_name]
                genome2[param_name] = parent1.genome[param_name]
        
        offspring1 = Individual(genome=genome1)
        offspring2 = Individual(genome=genome2)
        
        return offspring1, offspring2
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.clone()
        
        for param_name, (min_val, max_val, param_type) in self.parameter_space.items():
            if random.random() < self.config.mutation_rate:
                if self.config.mutation_method == 'gaussian':
                    mutated.genome[param_name] = self._gaussian_mutation(
                        mutated.genome[param_name], min_val, max_val, param_type
                    )
                elif self.config.mutation_method == 'uniform':
                    mutated.genome[param_name] = self._uniform_mutation(
                        min_val, max_val, param_type
                    )
                elif self.config.mutation_method == 'adaptive':
                    mutated.genome[param_name] = self._adaptive_mutation(
                        mutated.genome[param_name], min_val, max_val, param_type
                    )
        
        return mutated
    
    def _gaussian_mutation(self, current_value: Any, min_val: Any, 
                          max_val: Any, param_type: str) -> Any:
        """Gaussian mutation: add random noise from normal distribution."""
        if param_type == 'int':
            range_val = max_val - min_val
            noise = int(np.random.normal(0, range_val * self.config.mutation_strength))
            new_value = int(current_value) + noise
            return max(min_val, min(max_val, new_value))
        
        elif param_type == 'float':
            range_val = max_val - min_val
            noise = np.random.normal(0, range_val * self.config.mutation_strength)
            new_value = float(current_value) + noise
            return max(min_val, min(max_val, new_value))
        
        elif param_type == 'bool':
            # Flip with probability
            if random.random() < self.config.mutation_strength:
                return not current_value
            return current_value
        
        else:
            # For discrete choices, randomly change
            if random.random() < self.config.mutation_strength:
                return random.choice([min_val, max_val])
            return current_value
    
    def _uniform_mutation(self, min_val: Any, max_val: Any, param_type: str) -> Any:
        """Uniform mutation: replace with random value."""
        if param_type == 'int':
            return random.randint(int(min_val), int(max_val))
        elif param_type == 'float':
            return random.uniform(float(min_val), float(max_val))
        elif param_type == 'bool':
            return random.choice([True, False])
        else:
            return random.choice([min_val, max_val])
    
    def _adaptive_mutation(self, current_value: Any, min_val: Any,
                          max_val: Any, param_type: str) -> Any:
        """Adaptive mutation: adjust mutation strength based on diversity."""
        # Increase mutation strength if diversity is low
        diversity = self._calculate_diversity()
        adaptive_strength = self.config.mutation_strength * (2.0 - diversity)
        
        if param_type in ['int', 'float']:
            range_val = max_val - min_val
            noise = np.random.normal(0, range_val * adaptive_strength)
            
            if param_type == 'int':
                new_value = int(current_value) + int(noise)
                return max(min_val, min(max_val, new_value))
            else:
                new_value = float(current_value) + noise
                return max(min_val, min(max_val, new_value))
        
        return self._gaussian_mutation(current_value, min_val, max_val, param_type)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity (0 = no diversity, 1 = maximum diversity)."""
        if len(self.population) < 2:
            return 1.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._genome_distance(
                    self.population[i].genome,
                    self.population[j].genome
                )
                total_distance += distance
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_distance = total_distance / count
        # Normalize to [0, 1]
        max_possible_distance = len(self.parameter_space)
        diversity = min(1.0, avg_distance / max_possible_distance)
        
        return diversity
    
    def _genome_distance(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> float:
        """Calculate normalized distance between two genomes."""
        distance = 0.0
        
        for param_name, (min_val, max_val, param_type) in self.parameter_space.items():
            val1 = genome1.get(param_name, min_val)
            val2 = genome2.get(param_name, min_val)
            
            if param_type in ['int', 'float']:
                # Normalize to [0, 1]
                range_val = max_val - min_val
                if range_val > 0:
                    normalized_diff = abs(val1 - val2) / range_val
                    distance += normalized_diff
                else:
                    distance += 0 if val1 == val2 else 1
            else:
                # Binary or categorical
                distance += 0 if val1 == val2 else 1
        
        return distance
    
    def evolve(self, fitness_function: Callable[[Dict[str, Any]], float],
               verbose: bool = True) -> Individual:
        """
        Run genetic algorithm evolution.
        
        Args:
            fitness_function: Function to evaluate individual fitness
            verbose: Whether to print progress
            
        Returns:
            Best individual found
        """
        # Initialize population
        self.population = self.initialize_population()
        
        # Evaluate initial population
        for individual in self.population:
            self.evaluate_fitness(individual, fitness_function)
        
        self.best_individual = max(self.population, key=lambda ind: ind.fitness)
        self.fitness_history.append(self.best_individual.fitness)
        
        if verbose:
            logger.info(f"Generation 0: Best fitness = {self.best_individual.fitness:.6f}")
        
        # Evolution loop
        no_improvement_count = 0
        
        for generation in range(1, self.config.num_generations + 1):
            self.generation = generation
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:self.config.elite_size]
            new_population.extend([ind.clone() for ind in elite])
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Select parents
                parents = self.select_parents(self.population, n_parents=2)
                
                # Crossover
                offspring1, offspring2 = self.crossover(parents[0], parents[1])
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                # Evaluate offspring
                self.evaluate_fitness(offspring1, fitness_function)
                self.evaluate_fitness(offspring2, fitness_function)
                
                new_population.append(offspring1)
                if len(new_population) < self.config.population_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population = new_population[:self.config.population_size]
            
            # Update best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Track metrics
            self.fitness_history.append(self.best_individual.fitness)
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # Logging
            if verbose and generation % self.config.log_frequency == 0:
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                logger.info(
                    f"Generation {generation}: "
                    f"Best={self.best_individual.fitness:.6f}, "
                    f"Avg={avg_fitness:.6f}, "
                    f"Diversity={diversity:.3f}"
                )
            
            # Early stopping
            if self.config.early_stopping and no_improvement_count >= self.config.patience:
                logger.info(f"Early stopping at generation {generation}: No improvement for {self.config.patience} generations")
                break
        
        logger.info(f"Evolution complete. Best fitness: {self.best_individual.fitness:.6f}")
        return self.best_individual
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            'best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'best_parameters': self.best_individual.genome if self.best_individual else {},
            'total_generations': self.generation,
            'fitness_improvement': self.fitness_history[-1] - self.fitness_history[0] if self.fitness_history else 0.0,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history
        }
