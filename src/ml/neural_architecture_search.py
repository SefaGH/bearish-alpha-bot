"""
Neural Architecture Search (NAS) for Optimal Model Design.

Implements genetic algorithm-based search for optimal neural network
architectures for trading regime prediction and strategy optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. NAS will use mock implementations.")


@dataclass
class NetworkArchitecture:
    """Represents a neural network architecture."""
    
    num_layers: int
    layer_sizes: List[int]
    activations: List[str]
    dropout_rates: List[float]
    use_batch_norm: bool = False
    use_skip_connections: bool = False
    
    def __post_init__(self):
        """Validate architecture."""
        assert len(self.layer_sizes) == self.num_layers
        assert len(self.activations) == self.num_layers
        assert len(self.dropout_rates) == self.num_layers
    
    def get_parameter_count(self, input_size: int, output_size: int) -> int:
        """Estimate total number of parameters."""
        params = 0
        prev_size = input_size
        
        for layer_size in self.layer_sizes:
            params += prev_size * layer_size + layer_size  # weights + biases
            prev_size = layer_size
        
        # Output layer
        params += prev_size * output_size + output_size
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'num_layers': self.num_layers,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'dropout_rates': self.dropout_rates,
            'use_batch_norm': self.use_batch_norm,
            'use_skip_connections': self.use_skip_connections
        }
    
    @classmethod
    def from_dict(cls, arch_dict: Dict[str, Any]) -> 'NetworkArchitecture':
        """Create from dictionary representation."""
        return cls(
            num_layers=arch_dict['num_layers'],
            layer_sizes=arch_dict['layer_sizes'],
            activations=arch_dict['activations'],
            dropout_rates=arch_dict['dropout_rates'],
            use_batch_norm=arch_dict.get('use_batch_norm', False),
            use_skip_connections=arch_dict.get('use_skip_connections', False)
        )


if TORCH_AVAILABLE:
    class DynamicNetwork(nn.Module):
        """Dynamically constructed neural network from architecture specification."""
        
        def __init__(self, architecture: NetworkArchitecture, 
                     input_size: int, output_size: int):
            """
            Initialize dynamic network.
            
            Args:
                architecture: Network architecture specification
                input_size: Input dimension
                output_size: Output dimension
            """
            super().__init__()
            
            self.architecture = architecture
            self.input_size = input_size
            self.output_size = output_size
            
            # Build layers
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList() if architecture.use_batch_norm else None
            self.dropouts = nn.ModuleList()
            
            prev_size = input_size
            
            for i in range(architecture.num_layers):
                # Linear layer
                layer = nn.Linear(prev_size, architecture.layer_sizes[i])
                self.layers.append(layer)
                
                # Batch normalization
                if architecture.use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(architecture.layer_sizes[i]))
                
                # Dropout
                if architecture.dropout_rates[i] > 0:
                    self.dropouts.append(nn.Dropout(architecture.dropout_rates[i]))
                else:
                    self.dropouts.append(nn.Identity())
                
                prev_size = architecture.layer_sizes[i]
            
            # Output layer
            self.output_layer = nn.Linear(prev_size, output_size)
            
            # Activation functions
            self.activations = self._create_activations(architecture.activations)
        
        def _create_activations(self, activation_names: List[str]) -> nn.ModuleList:
            """Create activation function modules."""
            activations = nn.ModuleList()
            
            for name in activation_names:
                if name == 'relu':
                    activations.append(nn.ReLU())
                elif name == 'tanh':
                    activations.append(nn.Tanh())
                elif name == 'sigmoid':
                    activations.append(nn.Sigmoid())
                elif name == 'leaky_relu':
                    activations.append(nn.LeakyReLU())
                elif name == 'elu':
                    activations.append(nn.ELU())
                else:
                    activations.append(nn.ReLU())  # Default
            
            return activations
        
        def forward(self, x):
            """Forward pass through network."""
            for i in range(self.architecture.num_layers):
                # Linear transformation
                x = self.layers[i](x)
                
                # Batch normalization
                if self.architecture.use_batch_norm:
                    x = self.batch_norms[i](x)
                
                # Activation
                x = self.activations[i](x)
                
                # Dropout
                x = self.dropouts[i](x)
            
            # Output layer
            x = self.output_layer(x)
            
            return x


class NeuralArchitectureSearch:
    """
    Neural Architecture Search using genetic algorithms.
    
    Searches for optimal network architectures by evolving populations
    of architectures and evaluating their performance.
    """
    
    def __init__(self, config, input_size: int, output_size: int):
        """
        Initialize NAS.
        
        Args:
            config: NeuralArchitectureSearchConfig instance
            input_size: Input dimension
            output_size: Output dimension (number of classes)
        """
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        self.population: List[NetworkArchitecture] = []
        self.fitness_scores: List[float] = []
        self.best_architecture: Optional[NetworkArchitecture] = None
        self.best_score: float = 0.0
        
        self.search_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized NAS with input_size={input_size}, output_size={output_size}")
    
    def random_architecture(self) -> NetworkArchitecture:
        """Generate a random valid architecture."""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        
        layer_sizes = [
            random.choice(self.config.layer_size_options)
            for _ in range(num_layers)
        ]
        
        activations = [
            random.choice(self.config.activation_options)
            for _ in range(num_layers)
        ]
        
        dropout_rates = [
            random.choice(self.config.dropout_options)
            for _ in range(num_layers)
        ]
        
        use_batch_norm = random.choice([True, False])
        use_skip_connections = random.choice([True, False]) if num_layers >= 3 else False
        
        return NetworkArchitecture(
            num_layers=num_layers,
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            use_batch_norm=use_batch_norm,
            use_skip_connections=use_skip_connections
        )
    
    def initialize_population(self, population_size: int) -> List[NetworkArchitecture]:
        """Initialize population of random architectures."""
        population = []
        
        for _ in range(population_size):
            arch = self.random_architecture()
            
            # Check parameter constraint
            param_count = arch.get_parameter_count(self.input_size, self.output_size)
            if param_count <= self.config.max_parameters:
                population.append(arch)
            else:
                # Retry with smaller architecture
                arch = self._generate_smaller_architecture()
                population.append(arch)
        
        logger.info(f"Initialized population of {len(population)} architectures")
        return population
    
    def _generate_smaller_architecture(self) -> NetworkArchitecture:
        """Generate architecture with fewer parameters."""
        num_layers = random.randint(self.config.min_layers, 
                                   min(4, self.config.max_layers))
        
        # Use smaller layer sizes
        smaller_options = [s for s in self.config.layer_size_options if s <= 128]
        if not smaller_options:
            smaller_options = [64, 128]
        
        layer_sizes = [random.choice(smaller_options) for _ in range(num_layers)]
        activations = ['relu'] * num_layers  # Simple activation
        dropout_rates = [0.1] * num_layers  # Light dropout
        
        return NetworkArchitecture(
            num_layers=num_layers,
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            use_batch_norm=False,
            use_skip_connections=False
        )
    
    def evaluate_architecture(self, architecture: NetworkArchitecture,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Evaluate architecture performance.
        
        Args:
            architecture: Architecture to evaluate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Validation accuracy score
        """
        if not TORCH_AVAILABLE:
            # Mock evaluation
            return random.uniform(0.5, 0.8)
        
        try:
            # Create model
            model = DynamicNetwork(architecture, self.input_size, self.output_size)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val)
            
            # Quick training
            model.train()
            for epoch in range(self.config.evaluation_epochs):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val_t).float().mean().item()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return 0.0
    
    def mutate_architecture(self, architecture: NetworkArchitecture,
                          mutation_rate: float = 0.2) -> NetworkArchitecture:
        """
        Mutate an architecture.
        
        Args:
            architecture: Architecture to mutate
            mutation_rate: Probability of mutating each component
            
        Returns:
            Mutated architecture
        """
        # Create a copy
        mutated = NetworkArchitecture(
            num_layers=architecture.num_layers,
            layer_sizes=architecture.layer_sizes.copy(),
            activations=architecture.activations.copy(),
            dropout_rates=architecture.dropout_rates.copy(),
            use_batch_norm=architecture.use_batch_norm,
            use_skip_connections=architecture.use_skip_connections
        )
        
        # Mutate number of layers
        if random.random() < mutation_rate:
            change = random.choice([-1, 1])
            new_layers = mutated.num_layers + change
            
            if self.config.min_layers <= new_layers <= self.config.max_layers:
                if change > 0:
                    # Add layer
                    mutated.layer_sizes.append(random.choice(self.config.layer_size_options))
                    mutated.activations.append(random.choice(self.config.activation_options))
                    mutated.dropout_rates.append(random.choice(self.config.dropout_options))
                else:
                    # Remove layer
                    if len(mutated.layer_sizes) > 1:
                        idx = random.randint(0, len(mutated.layer_sizes) - 1)
                        mutated.layer_sizes.pop(idx)
                        mutated.activations.pop(idx)
                        mutated.dropout_rates.pop(idx)
                
                mutated.num_layers = len(mutated.layer_sizes)
        
        # Mutate layer sizes
        for i in range(mutated.num_layers):
            if random.random() < mutation_rate:
                mutated.layer_sizes[i] = random.choice(self.config.layer_size_options)
        
        # Mutate activations
        for i in range(mutated.num_layers):
            if random.random() < mutation_rate:
                mutated.activations[i] = random.choice(self.config.activation_options)
        
        # Mutate dropout rates
        for i in range(mutated.num_layers):
            if random.random() < mutation_rate:
                mutated.dropout_rates[i] = random.choice(self.config.dropout_options)
        
        # Mutate batch norm
        if random.random() < mutation_rate:
            mutated.use_batch_norm = not mutated.use_batch_norm
        
        # Mutate skip connections
        if random.random() < mutation_rate:
            mutated.use_skip_connections = not mutated.use_skip_connections
        
        return mutated
    
    def crossover_architectures(self, parent1: NetworkArchitecture,
                               parent2: NetworkArchitecture) -> NetworkArchitecture:
        """
        Create offspring by crossing over two parent architectures.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring architecture
        """
        # Inherit number of layers from one parent
        if random.random() < 0.5:
            num_layers = parent1.num_layers
            base_parent = parent1
        else:
            num_layers = parent2.num_layers
            base_parent = parent2
        
        # Mix layer properties
        layer_sizes = []
        activations = []
        dropout_rates = []
        
        for i in range(num_layers):
            # Choose from either parent (if layer exists)
            if i < parent1.num_layers and i < parent2.num_layers:
                if random.random() < 0.5:
                    layer_sizes.append(parent1.layer_sizes[i])
                    activations.append(parent1.activations[i])
                    dropout_rates.append(parent1.dropout_rates[i])
                else:
                    layer_sizes.append(parent2.layer_sizes[i])
                    activations.append(parent2.activations[i])
                    dropout_rates.append(parent2.dropout_rates[i])
            elif i < base_parent.num_layers:
                layer_sizes.append(base_parent.layer_sizes[i])
                activations.append(base_parent.activations[i])
                dropout_rates.append(base_parent.dropout_rates[i])
            else:
                # Random if beyond both parents
                layer_sizes.append(random.choice(self.config.layer_size_options))
                activations.append(random.choice(self.config.activation_options))
                dropout_rates.append(random.choice(self.config.dropout_options))
        
        # Inherit other properties
        use_batch_norm = random.choice([parent1.use_batch_norm, parent2.use_batch_norm])
        use_skip_connections = random.choice([parent1.use_skip_connections, parent2.use_skip_connections])
        
        return NetworkArchitecture(
            num_layers=num_layers,
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rates=dropout_rates,
            use_batch_norm=use_batch_norm,
            use_skip_connections=use_skip_connections
        )
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              population_size: int = 20,
              num_generations: int = 10,
              verbose: bool = True) -> NetworkArchitecture:
        """
        Run neural architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            population_size: Size of architecture population
            num_generations: Number of generations to evolve
            verbose: Whether to print progress
            
        Returns:
            Best architecture found
        """
        # Initialize population
        self.population = self.initialize_population(population_size)
        
        # Evaluate initial population
        self.fitness_scores = []
        for arch in self.population:
            score = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            self.fitness_scores.append(score)
        
        # Track best
        best_idx = np.argmax(self.fitness_scores)
        self.best_architecture = self.population[best_idx]
        self.best_score = self.fitness_scores[best_idx]
        
        if verbose:
            logger.info(f"Generation 0: Best score = {self.best_score:.4f}")
        
        # Evolution
        for generation in range(1, num_generations + 1):
            # Create new population
            new_population = []
            new_fitness = []
            
            # Elitism: keep top 20%
            elite_size = max(1, population_size // 5)
            elite_indices = np.argsort(self.fitness_scores)[-elite_size:]
            
            for idx in elite_indices:
                new_population.append(self.population[idx])
                new_fitness.append(self.fitness_scores[idx])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(len(self.population)), tournament_size)
                tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
                parent_indices = [tournament_indices[i] for i in np.argsort(tournament_fitness)[-2:]]
                
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Crossover
                offspring = self.crossover_architectures(parent1, parent2)
                
                # Mutation
                offspring = self.mutate_architecture(offspring)
                
                # Check constraints
                param_count = offspring.get_parameter_count(self.input_size, self.output_size)
                if param_count > self.config.max_parameters:
                    offspring = self._generate_smaller_architecture()
                
                # Evaluate
                score = self.evaluate_architecture(offspring, X_train, y_train, X_val, y_val)
                
                new_population.append(offspring)
                new_fitness.append(score)
            
            # Update population
            self.population = new_population[:population_size]
            self.fitness_scores = new_fitness[:population_size]
            
            # Update best
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.best_score:
                self.best_architecture = self.population[best_idx]
                self.best_score = self.fitness_scores[best_idx]
            
            # Store history
            self.search_history.append({
                'generation': generation,
                'best_score': self.best_score,
                'avg_score': np.mean(self.fitness_scores),
                'best_architecture': self.best_architecture.to_dict()
            })
            
            if verbose:
                logger.info(
                    f"Generation {generation}: "
                    f"Best={self.best_score:.4f}, "
                    f"Avg={np.mean(self.fitness_scores):.4f}"
                )
        
        logger.info(f"NAS complete. Best architecture score: {self.best_score:.4f}")
        return self.best_architecture
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of architecture search results."""
        return {
            'best_score': self.best_score,
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'parameter_count': self.best_architecture.get_parameter_count(
                self.input_size, self.output_size
            ) if self.best_architecture else 0,
            'search_history': self.search_history,
            'total_generations': len(self.search_history)
        }
