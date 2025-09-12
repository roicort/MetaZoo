# Genetic Algorithms

from typing import Optional, Callable, Sequence, Tuple
import numpy as np
from rich.progress import Progress
from rich.table import Table
from rich.console import Console
from plotly import express as px

from .utils import Population, Encoding


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
        encoder: Encoding,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        minimize: bool = True,
        elitism: float = None,  # Percentage of best individuals to preserve
        population_size: int = 100,
    ):
        self.fitness_function = fitness_function
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        self.selection_function = selection_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.minimize = minimize
        self.elitism = elitism

        self.population = Population(population_size, encoder)

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
        #table.add_row("Encoding", str(self.encoding))
        table.add_row("Selection Function", self.selection_function.__name__)
        table.add_row("Crossover Function", self.crossover_function.__name__)
        table.add_row("Mutation Function", self.mutation_function.__name__)
        table.add_row("Fitness Function", self.fitness_function.__name__)
        table.add_row("Dimension", str(self.dim))
        table.add_row(
            "Elitism", str(self.elitism) if self.elitism is not None else "None"
        )
        #if self.encoding == "binary":
        #    table.add_row("Epsilon", str(self.epsilon))
        #    table.add_row("Bits Per Var", str(self.bits_per_var))
        #    table.add_row("Genome Length", str(self.genome_length))

        table.add_row("Minimize", str(self.minimize))
        console = Console()
        console.print(table)

    def eval(self):
        """
        Evaluate the fitness of the current population.
        """
        
        # Raw Fitness.

        raw_fitness = np.array(
            [
                self.fitness_function(self.population.encoding.decode(individual))
                for individual in self.population.individuals
            ]
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

        bestcandidate = self.population.individuals[best_idx]
        self.best_individual = self.population.encoding.decode(bestcandidate)

        return fitness, fitness_transformed

    def evolve(self):
        """
        Perform one generation of evolution.
        """
        fitness, fitness_transformed = self.eval()
        self.fitness_history.append(fitness.mean())
        self.best_history.append(self.best_fitness)
        selected_indices = self.selection_function(self.population.individuals, fitness_transformed)
        selected_parents = self.population.individuals[selected_indices]
        next_generation = self.create_descendants(selected_parents)

        if self.elitism is not None:
            # Elitism
            # Its a stategy to preserve the best individuals from one generation to the next.
            # This is done to ensure that the best solutions found so far are not lost due to the stochastic nature of genetic algorithms.
            # Here, we simply copy the best individual from the current population to the next generation.
            # This garantees that the best solution found so far is always preserved.
            # Think that in nature, the best individuals are more likely to survive and reproduce, passing their genes to the next generation.
            # Preserve N% of the best individuals
            N = max(1, int(self.elitism * self.population.size))
            if self.minimize:
                elite_indices = np.argsort(fitness)[
                    :N
                ]  # Indices of the N best individuals (minimization)
            else:
                elite_indices = np.argsort(fitness)[
                    -N:
                ]  # Indices of the N best individuals (maximization)
            elites = self.population.individuals[elite_indices]
            next_generation[:N] = (
                elites  # Replace the first N individuals with the elites
            )

        self.population.individuals = next_generation

    def create_descendants(self, parents: np.ndarray) -> np.ndarray:
        """
        Create descendants from the selected parents using crossover and mutation.
        """
        # Validate
        if len(parents) < 2:
            raise ValueError("Not enough parents to create descendants.")
        next_generation = []
        for _ in range(self.population.size // 2):
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

    def run(
        self, generations: int, history: bool = False, verbose: bool = True
    ) -> list[np.ndarray]:
        pop_history = []
        if verbose:
            with Progress() as progress:
                task = progress.add_task("Evolving...", total=generations)
                for _ in range(generations):
                    self.evolve()
                    pop_history.append(self.population.individuals.copy())
                    progress.advance(task)
        else:
            for _ in range(generations):
                self.evolve()
                pop_history.append(self.population.individuals.copy())
        if history:
            pop_history = [[self.population.encoding.decode(ind) for ind in gen] for gen in pop_history]
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
