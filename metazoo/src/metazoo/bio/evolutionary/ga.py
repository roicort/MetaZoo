# Genetic Algorithms

from typing import Optional, Callable, Sequence, Tuple
import numpy as np
from rich.progress import Progress

from .utils import Population

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        crossover_function: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        mutation_function: Callable[[np.ndarray], np.ndarray],
        selection_function: Callable[[np.ndarray], np.ndarray],
        encoding: str,
        population_size: int,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        binary_precision: int = 3, # Number of bits per variable for binary encoding
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        genome_length: Optional[int] = None,
    ):
        self.population_size = population_size
        self.genome_length = genome_length
        self.fitness_function = fitness_function
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        self.selection_function = selection_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = bounds
        self.dim = len(self.bounds)
        self.encoding = encoding
        self.binary_precision = binary_precision

        # Get best genome_length
        if self.genome_length is None:
            if self.encoding == 'real':
                self.genome_length = self.dim
            elif self.encoding == 'binary':
                self.genome_length = self.binary_precision * self.dim

        # Validate
        if self.encoding == 'binary' and self.genome_length % self.dim != 0:
            raise ValueError(f"genome_length ({self.genome_length}) must be a multiple of dim ({self.dim}) for binary encoding.")

        # Initialize population
        self.population = Population(
            population_size=self.population_size,
            genome_length=self.genome_length,
            bounds=self.bounds,
            encoding=self.encoding
        ).initialize()

        self.best_individual = None
        self.best_fitness = -np.inf

    def eval(self):
        if self.encoding == 'binary':
            fitness = np.array([self.fitness_function(self.decode(individual)) for individual in self.population])
        else:
            fitness = np.array([self.fitness_function(individual) for individual in self.population])
        self.best_fitness = fitness.max()
        self.best_individual = self.population[fitness.argmax()]
        return fitness

    def evolve(self):
        fitness = self.eval()
        selected_indices = self.selection_function(self.population, fitness)
        selected_parents = self.population[selected_indices]
        next_generation = self.create_descendants(selected_parents)
        self.population = next_generation

    def create_descendants(self, parents: np.ndarray) -> np.ndarray:
        next_generation = []
        for _ in range(self.population_size // 2):
            # Select two parents
            idx1, idx2 = np.random.choice(len(parents), size=2, replace=False)
            parent1 = parents[idx1]
            parent2 = parents[idx2]
            child1, child2 = self.crossover_function(parent1, parent2)
            # Apply mutation
            child1 = self.mutation_function(child1)
            child2 = self.mutation_function(child2)
            next_generation.extend([child1, child2])

        return np.array(next_generation)

    def decode(self, individual: np.ndarray) -> np.ndarray:
        """
        Decodes a binary individual into its real-valued representation.
        """
        if self.encoding == 'binary':
            bits_per_var = self.binary_precision
            decoded = []
            for i, (a, b) in enumerate(self.bounds):
                bits = individual[i*bits_per_var:(i+1)*bits_per_var]
                value = int("".join(str(int(bit)) for bit in bits), 2)
                max_value = 2**bits_per_var - 1
                real_value = a + (b - a) * value / max_value
                decoded.append(real_value)
            return np.array(decoded)
        else:
            raise NotImplementedError("Decoding not implemented for this encoding.")

    def run(self, generations: int):
        with Progress() as progress:
            task = progress.add_task("Evolving...", total=generations)
            for _ in range(generations):
                self.evolve()
                progress.advance(task)