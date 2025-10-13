"""
Multi-Objective Optimization Framework.

Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
optimizing multiple trading objectives simultaneously (returns, Sharpe,
drawdown, win rate, etc.).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from .genetic_optimizer import Individual, GeneticOptimizer

logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveIndividual(Individual):
    """Individual with multiple objective scores for Pareto optimization."""
    
    objective_scores: Dict[str, float] = field(default_factory=dict)
    rank: int = 0  # Pareto rank (0 = best)
    crowding_distance: float = 0.0  # Diversity measure
    dominates: List[int] = field(default_factory=list)  # Indices of dominated individuals
    dominated_by: int = 0  # Count of individuals that dominate this one
    
    def dominates_other(self, other: 'MultiObjectiveIndividual',
                       maximize_objectives: Dict[str, bool]) -> bool:
        """
        Check if this individual dominates another in Pareto sense.
        
        An individual dominates another if it's at least as good in all
        objectives and strictly better in at least one objective.
        
        Args:
            other: Other individual to compare
            maximize_objectives: Dict indicating which objectives to maximize
            
        Returns:
            True if this individual dominates the other
        """
        at_least_as_good = True
        strictly_better = False
        
        for obj_name, score in self.objective_scores.items():
            other_score = other.objective_scores.get(obj_name, 0.0)
            maximize = maximize_objectives.get(obj_name, True)
            
            if maximize:
                # For maximization objectives
                if score < other_score:
                    at_least_as_good = False
                    break
                if score > other_score:
                    strictly_better = True
            else:
                # For minimization objectives
                if score > other_score:
                    at_least_as_good = False
                    break
                if score < other_score:
                    strictly_better = True
        
        return at_least_as_good and strictly_better


class MultiObjectiveOptimizer(GeneticOptimizer):
    """
    Multi-objective genetic algorithm using NSGA-II.
    
    Optimizes multiple objectives simultaneously and finds the Pareto front
    of non-dominated solutions.
    """
    
    def __init__(self, config, parameter_space: Dict[str, Tuple[Any, Any, str]],
                 objectives: List[str], maximize_objectives: Dict[str, bool]):
        """
        Initialize multi-objective optimizer.
        
        Args:
            config: MultiObjectiveConfig instance
            parameter_space: Dictionary of parameter bounds
            objectives: List of objective names to optimize
            maximize_objectives: Dict indicating which objectives to maximize
        """
        super().__init__(config, parameter_space)
        
        self.objectives = objectives
        self.maximize_objectives = maximize_objectives
        self.pareto_front: List[MultiObjectiveIndividual] = []
        self.pareto_history: List[List[Dict[str, Any]]] = []
        
        logger.info(f"Initialized MultiObjectiveOptimizer with {len(objectives)} objectives")
    
    def evaluate_objectives(self, individual: MultiObjectiveIndividual,
                          objective_functions: Dict[str, Callable]) -> Dict[str, float]:
        """
        Evaluate all objectives for an individual.
        
        Args:
            individual: Individual to evaluate
            objective_functions: Dict mapping objective names to evaluation functions
            
        Returns:
            Dictionary of objective scores
        """
        objective_scores = {}
        
        for obj_name in self.objectives:
            if obj_name in objective_functions:
                try:
                    score = objective_functions[obj_name](individual.genome)
                    objective_scores[obj_name] = score
                except Exception as e:
                    logger.error(f"Error evaluating objective {obj_name}: {e}")
                    objective_scores[obj_name] = 0.0
            else:
                logger.warning(f"No evaluation function for objective {obj_name}")
                objective_scores[obj_name] = 0.0
        
        individual.objective_scores = objective_scores
        
        # Calculate aggregate fitness for compatibility with parent class
        individual.fitness = self._aggregate_fitness(objective_scores)
        
        return objective_scores
    
    def _aggregate_fitness(self, objective_scores: Dict[str, float]) -> float:
        """
        Aggregate multiple objectives into single fitness score.
        Uses weighted sum for compatibility with parent class methods.
        
        Args:
            objective_scores: Dictionary of objective scores
            
        Returns:
            Aggregated fitness score
        """
        if hasattr(self.config, 'objective_weights'):
            weights = self.config.objective_weights
        else:
            # Equal weights
            weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}
        
        fitness = 0.0
        for obj_name, score in objective_scores.items():
            weight = weights.get(obj_name, 1.0 / len(self.objectives))
            maximize = self.maximize_objectives.get(obj_name, True)
            
            if maximize:
                fitness += weight * score
            else:
                # For minimization, negate the score
                fitness -= weight * score
        
        return fitness
    
    def fast_non_dominated_sort(self, population: List[MultiObjectiveIndividual]
                               ) -> List[List[MultiObjectiveIndividual]]:
        """
        Perform fast non-dominated sorting (NSGA-II).
        
        Args:
            population: Population to sort
            
        Returns:
            List of fronts, where each front is a list of individuals
        """
        # Reset domination information
        for ind in population:
            ind.dominates = []
            ind.dominated_by = 0
            ind.rank = 0
        
        # Calculate domination
        for i, ind_i in enumerate(population):
            for j, ind_j in enumerate(population):
                if i != j:
                    if ind_i.dominates_other(ind_j, self.maximize_objectives):
                        ind_i.dominates.append(j)
                    elif ind_j.dominates_other(ind_i, self.maximize_objectives):
                        ind_i.dominated_by += 1
        
        # Create fronts
        fronts = [[]]
        
        # First front: non-dominated individuals
        for i, ind in enumerate(population):
            if ind.dominated_by == 0:
                ind.rank = 0
                fronts[0].append(ind)
        
        # Subsequent fronts
        current_front = 0
        while current_front < len(fronts) and len(fronts[current_front]) > 0:
            next_front = []
            
            for ind in fronts[current_front]:
                for j in ind.dominates:
                    dominated_ind = population[j]
                    dominated_ind.dominated_by -= 1
                    
                    if dominated_ind.dominated_by == 0:
                        dominated_ind.rank = current_front + 1
                        next_front.append(dominated_ind)
            
            if next_front:
                fronts.append(next_front)
            
            current_front += 1
        
        # Remove empty last front if any
        while fronts and fronts[-1] == []:
            fronts.pop()
        
        logger.debug(f"Non-dominated sorting: {len(fronts)} fronts, "
                    f"first front size: {len(fronts[0]) if fronts else 0}")
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[MultiObjectiveIndividual]):
        """
        Calculate crowding distance for individuals in a front.
        
        Crowding distance measures density of solutions around an individual.
        Higher values indicate more isolated solutions (better diversity).
        
        Args:
            front: List of individuals in the same Pareto front
        """
        if len(front) <= 2:
            # Boundary solutions get infinite distance
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for ind in front:
            ind.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for obj_name in self.objectives:
            # Sort by this objective
            front.sort(key=lambda ind: ind.objective_scores.get(obj_name, 0.0))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = front[0].objective_scores.get(obj_name, 0.0)
            obj_max = front[-1].objective_scores.get(obj_name, 0.0)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate distances for middle solutions
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    prev_score = front[i - 1].objective_scores.get(obj_name, 0.0)
                    next_score = front[i + 1].objective_scores.get(obj_name, 0.0)
                    front[i].crowding_distance += (next_score - prev_score) / obj_range
    
    def crowding_distance_assignment(self, population: List[MultiObjectiveIndividual]):
        """
        Assign crowding distances to entire population using fronts.
        
        Args:
            population: Population to process
        """
        fronts = self.fast_non_dominated_sort(population)
        
        for front in fronts:
            self.calculate_crowding_distance(front)
    
    def select_next_generation(self, population: List[MultiObjectiveIndividual],
                              offspring: List[MultiObjectiveIndividual]
                              ) -> List[MultiObjectiveIndividual]:
        """
        Select next generation using non-dominated sorting and crowding distance.
        
        Args:
            population: Current population
            offspring: Offspring generated
            
        Returns:
            Next generation population
        """
        # Combine population and offspring
        combined = population + offspring
        
        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined)
        
        # Calculate crowding distances
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select individuals for next generation
        next_generation = []
        
        for front in fronts:
            if len(next_generation) + len(front) <= self.config.population_size:
                # Add entire front
                next_generation.extend(front)
            else:
                # Sort by crowding distance (descending) and add until population is full
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                remaining = self.config.population_size - len(next_generation)
                next_generation.extend(front[:remaining])
                break
        
        return next_generation
    
    def get_pareto_front(self, population: List[MultiObjectiveIndividual]
                        ) -> List[MultiObjectiveIndividual]:
        """
        Extract the Pareto front from population.
        
        Args:
            population: Population to extract from
            
        Returns:
            List of individuals on the Pareto front (rank 0)
        """
        fronts = self.fast_non_dominated_sort(population)
        
        if fronts:
            pareto_front = fronts[0]
            self.calculate_crowding_distance(pareto_front)
            return pareto_front
        
        return []
    
    def evolve_multi_objective(self, objective_functions: Dict[str, Callable],
                              verbose: bool = True) -> List[MultiObjectiveIndividual]:
        """
        Run multi-objective evolutionary algorithm (NSGA-II).
        
        Args:
            objective_functions: Dict of objective evaluation functions
            verbose: Whether to print progress
            
        Returns:
            Pareto front of best solutions
        """
        # Initialize population with MultiObjectiveIndividual
        self.population = []
        for _ in range(self.config.population_size):
            genome = self._random_genome()
            individual = MultiObjectiveIndividual(genome=genome)
            self.population.append(individual)
        
        # Evaluate initial population
        for individual in self.population:
            self.evaluate_objectives(individual, objective_functions)
        
        # Get initial Pareto front
        self.pareto_front = self.get_pareto_front(self.population)
        self._store_pareto_history()
        
        if verbose:
            logger.info(f"Generation 0: Pareto front size = {len(self.pareto_front)}")
        
        # Evolution loop
        for generation in range(1, self.config.num_generations + 1):
            self.generation = generation
            
            # Generate offspring
            offspring = []
            
            while len(offspring) < self.config.population_size:
                # Select parents (tournament selection based on rank and crowding)
                parents = self._tournament_selection_nsga2(self.population, n_parents=2)
                
                # Crossover
                if len(parents) >= 2:
                    child1, child2 = self.crossover(parents[0], parents[1])
                else:
                    child1 = parents[0].clone()
                    child2 = self._random_genome()
                    child2 = MultiObjectiveIndividual(genome=child2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Ensure they are MultiObjectiveIndividual
                if not isinstance(child1, MultiObjectiveIndividual):
                    child1 = MultiObjectiveIndividual(genome=child1.genome)
                if not isinstance(child2, MultiObjectiveIndividual):
                    child2 = MultiObjectiveIndividual(genome=child2.genome)
                
                # Evaluate offspring
                self.evaluate_objectives(child1, objective_functions)
                self.evaluate_objectives(child2, objective_functions)
                
                offspring.append(child1)
                if len(offspring) < self.config.population_size:
                    offspring.append(child2)
            
            # Select next generation
            self.population = self.select_next_generation(self.population, offspring)
            
            # Update Pareto front
            self.pareto_front = self.get_pareto_front(self.population)
            self._store_pareto_history()
            
            # Logging
            if verbose and generation % self.config.log_frequency == 0:
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size={len(self.pareto_front)}, "
                    f"Avg fitness={avg_fitness:.6f}"
                )
        
        logger.info(f"Multi-objective evolution complete. Final Pareto front size: {len(self.pareto_front)}")
        return self.pareto_front
    
    def _tournament_selection_nsga2(self, population: List[MultiObjectiveIndividual],
                                   n_parents: int) -> List[MultiObjectiveIndividual]:
        """
        Tournament selection for NSGA-II based on rank and crowding distance.
        
        Args:
            population: Current population
            n_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        parents = []
        
        for _ in range(n_parents):
            # Random tournament
            tournament = random.sample(population, min(self.config.tournament_size, len(population)))
            
            # Select best based on rank, then crowding distance
            winner = min(tournament, key=lambda ind: (ind.rank, -ind.crowding_distance))
            parents.append(winner)
        
        return parents
    
    def _store_pareto_history(self):
        """Store current Pareto front for history tracking."""
        front_data = []
        
        for ind in self.pareto_front:
            front_data.append({
                'genome': deepcopy(ind.genome),
                'objectives': deepcopy(ind.objective_scores),
                'fitness': ind.fitness,
                'crowding_distance': ind.crowding_distance
            })
        
        self.pareto_history.append(front_data)
    
    def get_best_compromise_solution(self) -> Optional[MultiObjectiveIndividual]:
        """
        Get best compromise solution from Pareto front.
        Uses distance to ideal point method.
        
        Returns:
            Individual closest to ideal point
        """
        if not self.pareto_front:
            return None
        
        # Find ideal point (best value for each objective)
        ideal_point = {}
        
        for obj_name in self.objectives:
            scores = [ind.objective_scores.get(obj_name, 0.0) for ind in self.pareto_front]
            maximize = self.maximize_objectives.get(obj_name, True)
            ideal_point[obj_name] = max(scores) if maximize else min(scores)
        
        # Find solution closest to ideal point
        best_individual = None
        best_distance = float('inf')
        
        for ind in self.pareto_front:
            # Calculate normalized distance to ideal point
            distance = 0.0
            
            for obj_name in self.objectives:
                score = ind.objective_scores.get(obj_name, 0.0)
                ideal = ideal_point[obj_name]
                
                # Normalize by range
                scores = [i.objective_scores.get(obj_name, 0.0) for i in self.pareto_front]
                obj_range = max(scores) - min(scores)
                
                if obj_range > 0:
                    normalized_diff = abs(score - ideal) / obj_range
                    distance += normalized_diff ** 2
            
            distance = np.sqrt(distance)
            
            if distance < best_distance:
                best_distance = distance
                best_individual = ind
        
        return best_individual
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of multi-objective optimization results."""
        base_summary = super().get_optimization_summary()
        
        # Add multi-objective specific metrics
        base_summary['pareto_front_size'] = len(self.pareto_front)
        base_summary['pareto_solutions'] = []
        
        for ind in self.pareto_front:
            base_summary['pareto_solutions'].append({
                'parameters': deepcopy(ind.genome),
                'objectives': deepcopy(ind.objective_scores),
                'crowding_distance': ind.crowding_distance
            })
        
        # Best compromise solution
        compromise = self.get_best_compromise_solution()
        if compromise:
            base_summary['best_compromise'] = {
                'parameters': deepcopy(compromise.genome),
                'objectives': deepcopy(compromise.objective_scores)
            }
        
        return base_summary


import random  # Import was missing
