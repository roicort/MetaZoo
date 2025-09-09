# Genetic Algorithms

from typing import Optional, Callable, Sequence, Tuple
import numpy as np
from rich.progress import Progress
from rich.table import Table
from rich.console import Console
from plotly import express as px

from .utils import Population


class GeneticAlgorithm:
    """A Simple Genetic Algorithm (GA) implementation."""
    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        crossover_function: Callable[
            [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        mutation_function: Callable[[np.ndarray], np.ndarray],
        selection_function: Callable[[np.ndarray], np.ndarray],
        encoding: str,
        population_size: int,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        precision: int = 3,  # Number of bits per variable for binary encoding
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        minimize: bool = True,
        elitism: float = None,  # Percentage of best individuals to preserve
    ):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        self.selection_function = selection_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.encoding = encoding
        self.minimize = minimize
        self.genome_length = None
        self.elitism = elitism

        self.epsilon = 10 ** (
            -precision
        )  # ε = Obtaianable accuracy for binary encoding

        if bounds is None:
            raise ValueError(
                "Bounds must be provided as (low, high) or a sequence of (low, high)."
            )
        else:
            self.bounds = bounds
        self.dim = np.sum([1 for _ in bounds])  # Dimension inferred from bounds

        # Bits per variable for binary encoding
        # This is calculated based on the precision and the range of each variable
        # Using the formula: n = ceil(log2((XU - XL) / ε))
        # where XU and XL are the upper and lower bounds of the variable, and ε is the desired precision
        # We can understand ε as the smallest difference we want to be able to represent between two values of the variable
        self.bits_per_var = max(
            int(np.ceil(np.log2((xu - xl) / self.epsilon))) for xl, xu in self.bounds
        )
        # We use max to ensure we have enough bits for the most constrained variable

        # Get best genome_length
        if self.genome_length is None:
            if self.encoding == "real":
                self.genome_length = self.dim
            elif self.encoding == "binary":
                # For binary encoding, genome length is bits per variable times number of variables
                self.genome_length = self.bits_per_var * self.dim

        # Initialize population
        self.population = Population(
            population_size=self.population_size,
            genome_length=self.genome_length,
            bounds=self.bounds,
            encoding=self.encoding,
        ).initialize()

        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.best_history = []

    def summary(self):
        """
        Print all relevant information about the GA instance using rich.
        """
        table = Table(title="Genetic Algorithm Summary")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="bold magenta")
        table.add_row("Population Size", str(self.population_size))
        table.add_row("Genome Length", str(self.genome_length))
        table.add_row("Mutation Rate", str(self.mutation_rate))
        table.add_row("Crossover Rate", str(self.crossover_rate))
        table.add_row("Encoding", str(self.encoding))
        table.add_row("Selection Function", self.selection_function.__name__)
        table.add_row("Crossover Function", self.crossover_function.__name__)
        table.add_row("Mutation Function", self.mutation_function.__name__)
        table.add_row("Fitness Function", self.fitness_function.__name__)
        table.add_row("Dimension", str(self.dim))
        table.add_row(
            "Elitism", str(self.elitism) if self.elitism is not None else "None"
        )
        if self.encoding == "binary":
            table.add_row("Epsilon", str(self.epsilon))
            table.add_row("Bits Per Var", str(self.bits_per_var))
            table.add_row("Genome Length", str(self.genome_length))
        table.add_row("Bounds", str(self.bounds))
        table.add_row("Minimize", str(self.minimize))
        console = Console()
        console.print(table)

    def eval(self):
        # Raw Fitness.
        if self.encoding == "binary":
            raw_fitness = np.array(
                [
                    self.fitness_function(self.decode(individual))
                    for individual in self.population
                ]
            )
        else:
            raw_fitness = np.array(
                [self.fitness_function(individual) for individual in self.population]
            )

        if self.minimize:
            fitness = np.nan_to_num(raw_fitness, nan=1e10, posinf=1e10, neginf=1e10)
            self.best_fitness = float(fitness.min())
            best_idx = int(fitness.argmin())
            fitness_transformed = np.max(fitness) - fitness
        else:
            fitness = np.nan_to_num(raw_fitness, nan=-1e10, posinf=-1e10, neginf=-1e10)
            self.best_fitness = float(fitness.max())
            best_idx = int(fitness.argmax())
            fitness_transformed = fitness

        bestcandidate = self.population[best_idx]
        self.best_individual = (
            bestcandidate if self.encoding == "real" else self.decode(bestcandidate)
        )

        return fitness, fitness_transformed

    def evolve(self):
        fitness, fitness_transformed = self.eval()
        self.fitness_history.append(fitness.mean())
        self.best_history.append(self.best_fitness)
        selected_indices = self.selection_function(self.population, fitness_transformed)
        selected_parents = self.population[selected_indices]
        next_generation = self.create_descendants(selected_parents)

        if self.elitism is not None:
            # Elitism
            # Its a stategy to preserve the best individuals from one generation to the next.
            # This is done to ensure that the best solutions found so far are not lost due to the stochastic nature of genetic algorithms.
            # Here, we simply copy the best individual from the current population to the next generation.
            # This garantees that the best solution found so far is always preserved.
            # Think that in nature, the best individuals are more likely to survive and reproduce, passing their genes to the next generation.
            # Preserve N% of the best individuals
            N = max(1, int(self.elitism * self.population_size))
            if self.minimize:
                elite_indices = np.argsort(fitness)[
                    :N
                ]  # Indices of the N best individuals (minimization)
            else:
                elite_indices = np.argsort(fitness)[
                    -N:
                ]  # Indices of the N best individuals (maximization)
            elites = self.population[elite_indices]
            next_generation[:N] = (
                elites  # Replace the first N individuals with the elites
            )

        self.population = next_generation

    def create_descendants(self, parents: np.ndarray) -> np.ndarray:
        # Validate
        if len(parents) < 2:
            raise ValueError("Not enough parents to create descendants.")
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
        if self.encoding == "binary":
            bits_per_var = self.bits_per_var
            decoded = []
            for i, (a, b) in enumerate(self.bounds):
                bits = individual[i * bits_per_var : (i + 1) * bits_per_var]
                value = int("".join(str(int(bit)) for bit in bits), 2)
                max_value = 2**bits_per_var - 1
                real_value = a + (b - a) * value / max_value
                decoded.append(real_value)
            return np.array(decoded)
        else:
            raise NotImplementedError("Decoding not implemented for this encoding.")

    def run(
        self, generations: int, history: bool = False, verbose: bool = True
    ) -> list[np.ndarray]:
        pop_history = []
        if verbose:
            with Progress() as progress:
                task = progress.add_task("Evolving...", total=generations)
                for _ in range(generations):
                    self.evolve()
                    pop_history.append(self.population.copy())
                    progress.advance(task)
        else:
            for _ in range(generations):
                self.evolve()
                pop_history.append(self.population.copy())
        if history:
            if self.encoding == "binary":
                pop_history = [[self.decode(ind) for ind in gen] for gen in pop_history]
            return pop_history

    def fitness_plot(self) -> None:
        fig = px.line(
            y=np.array(self.fitness_history),
            labels={"x": "Generation", "y": "Fitness"},
            title="Fitness History",
        )
        fig.show()
        fig = px.line(
            y=np.array(self.best_history),
            labels={"x": "Generation", "y": "Best Fitness"},
            title="Best Fitness History",
        )
        fig.show()
