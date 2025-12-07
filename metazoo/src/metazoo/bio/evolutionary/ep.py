# Evolutionary Programming

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from plotly import express as px
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..utils import Encoding, Population


class EvolutionaryProgramming:
    """A Simple Evolutionary Programming (EP) implementation."""

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        encoder: Encoding,
        mutation_strategy: str = "self-adaptive",  # 'fixed' or 'self-adaptive'
        mutation_rate: float = 0.1,  # Used if strategy is 'fixed'
        minimize: bool = True,
        population_size: int = 100,
        tau: Optional[float] = None,
        tau_prime: Optional[float] = None,
        selection_strategy: str = "plus",  # "plus" for (μ+λ), "comma" for (μ,λ)
    ):
        self.fitness_function = fitness_function
        self.population = Population(population_size, encoder)

        if mutation_strategy not in ["fixed", "self-adaptive"]:
            raise ValueError("mutation_strategy must be 'fixed' or 'self-adaptive'")
        else:
            self.mutation_strategy = mutation_strategy
            if mutation_strategy == "fixed":
                self.mutation_function = self._fixed_mutation
            else:
                self.mutation_function = self._self_adaptive_mutation
                # Initialize strategies for self-adaptive mutation
                self.strategies = np.random.lognormal(
                    mean=0.0,
                    sigma=1.0,
                    size=(self.population.size, self.population.encoding.genome_length),
                )
                # Ensure minimum strategy values to avoid premature convergence
                self.strategies = np.maximum(self.strategies, 1e-6)

        if selection_strategy not in ["plus", "comma"]:
            raise ValueError("selection_strategy must be 'plus' or 'comma'")
        else:
            if selection_strategy == "plus":
                self.selection_function = self._plus_selection
            else:
                self.selection_function = self._comma_selection

        self.mutation_rate = mutation_rate
        self.minimize = minimize

        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.best_history = []

        # Auto-adaptation parameters
        dimension = encoder.genome_length
        self.tau = tau if tau is not None else 1.0 / np.sqrt(2.0 * np.sqrt(dimension))
        self.tau_prime = (
            tau_prime if tau_prime is not None else 1.0 / np.sqrt(2.0 * dimension)
        )

    def _fixed_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply fixed mutation to an individual.
        """
        mutation_mask = np.random.rand(len(individual)) < self.mutation_rate
        mutation_values = np.random.normal(0, 1, size=len(individual))
        mutated_individual = np.where(
            mutation_mask, individual + mutation_values, individual
        )
        return mutated_individual

    def _self_adaptive_mutation(
        self, individual: np.ndarray, strategy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply self-adaptive mutation to an individual.
        """
        dimension = len(individual)
        global_noise = np.random.normal(0, 1)
        local_noise = np.random.normal(0, 1, size=dimension)

        # Update strategies
        new_strategy = strategy * np.exp(
            self.tau_prime * global_noise + self.tau * local_noise
        )
        new_strategy = np.clip(
            new_strategy, 1e-10, None
        )  # Prevent strategies from becoming too small

        # Mutate individual
        mutation_values = np.random.normal(0, new_strategy)
        mutated_individual = individual + mutation_values

        return mutated_individual, new_strategy

    def _plus_selection(
        self,
        combined_population: Sequence[np.ndarray],
        fitness: np.ndarray,
        num_survivors: int,
    ) -> Sequence[np.ndarray]:
        """
        (μ + λ) Selection: Select the best individuals from the combined parent and offspring populations.
        """
        sorted_indices = np.argsort(fitness) if self.minimize else np.argsort(-fitness)
        selected_indices = sorted_indices[:num_survivors]
        return [combined_population[i] for i in selected_indices]

    def _comma_selection(
        self,
        offspring_population: Sequence[np.ndarray],
        fitness: np.ndarray,
        num_survivors: int,
    ) -> Sequence[np.ndarray]:
        """
        (μ, λ) Selection: Select the best individuals only from the offspring population.
        """
        sorted_indices = np.argsort(fitness) if self.minimize else np.argsort(-fitness)
        selected_indices = sorted_indices[:num_survivors]
        return [offspring_population[i] for i in selected_indices]

    def summary(self):
        """
        Print all relevant information about the EP instance using rich.
        """
        table = Table(title="Evolutionary Programming Summary")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="bold magenta")
        table.add_row("Population Size", str(self.population.size))
        table.add_row("Genome Length", str(self.population.genome_length))
        table.add_row("Mutation Strategy", str(self.mutation_strategy))
        if self.mutation_strategy == "fixed":
            table.add_row("Mutation Rate", str(self.mutation_rate))
        table.add_row("Selection Function", self.selection_function.__name__)
        table.add_row("Mutation Function", self.mutation_function.__name__)

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

    def mutate(self, parents: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        Create descendants from selected parents only using mutation.
        """
        offspring = []
        for i, parent in enumerate(parents):
            if self.mutation_strategy == "fixed":
                child = self.mutation_function(parent)
                offspring.append(child)
            else:
                strategy = self.strategies[i]
                child, new_strategy = self._self_adaptive_mutation(parent, strategy)
                offspring.append(child)
                self.strategies[i] = new_strategy
        return np.array(offspring)

    def select_survivors(self, population, fitness) -> Sequence[np.ndarray]:
        """
        Select survivors from the combined population (parents + offspring) or only offspring.
        """
        if self.selection_function == self._plus_selection:
            combined_population = np.vstack((self.population.individuals, population))
            combined_fitness = np.hstack(
                (
                    fitness,
                    np.array(
                        [
                            self.fitness_function(self.population.encoding.decode(ind))
                            for ind in population
                        ]
                    ),
                )
            )
            survivors = self.selection_function(
                combined_population, combined_fitness, self.population.size
            )
        else:
            descendants_fitness = np.array(
                [
                    self.fitness_function(self.population.encoding.decode(ind))
                    for ind in population
                ]
            )
            if self.minimize:
                descendants_fitness = np.max(descendants_fitness) - descendants_fitness
            survivors = self.selection_function(
                population, descendants_fitness, self.population.size
            )
        return np.array(survivors)

    def evolve(self):
        """
        Perform one generation of evolution.
        """

        # Evaluate fitness
        fitness, fitness_transformed = self.eval()
        self.fitness_history.append(fitness.mean())
        self.best_history.append(self.best_fitness)

        # Mutate
        descendants = self.mutate(self.population.individuals)

        # Select Survivors
        survivors = self.select_survivors(descendants, fitness_transformed)

        # Update population
        self.population.individuals = np.array(survivors)

    def run(
        self, generations: int, history: bool = False, verbose: bool = True
    ) -> list[np.ndarray]:
        pop_history = []
        best_history = []
        if verbose:
            with Progress() as progress:
                task = progress.add_task("Evolving...", total=generations)
                for _ in range(generations):
                    self.evolve()
                    pop_history.append(self.population.individuals.copy())
                    best_history.append(self.best_individual)
                    progress.advance(task)
        else:
            for _ in range(generations):
                self.evolve()
                pop_history.append(self.population.individuals.copy())
                best_history.append(self.best_individual)
        if history:
            pop_history = [
                [self.population.encoding.decode(ind) for ind in gen]
                for gen in pop_history
            ]
            return (best_history, pop_history)

    def fitness_plot(self, best=False) -> None:
        if self.fitness_history:
            if not best:
                fig = px.line(
                    y=np.array(self.fitness_history),
                    labels={"x": "Generation", "y": "Fitness"},
                    title="Fitness History",
                )
                return fig
            else:
                fig = px.line(
                    y=np.array(self.best_history),
                    labels={"x": "Generation", "y": "Best Fitness"},
                    title="Best Fitness History",
                )
                return fig
        else:
            raise ValueError("No fitness history to plot. Run the algorithm first.")
