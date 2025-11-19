# Differential Evolution

import numpy as np

from metazoo.bio.evolutionary.operators import crossover, mutation
from metazoo.bio.utils import Encoding, Permutation, Population


class DifferentialEvolution:
    def __init__(
        self,
        fitness_function,
        encoder: Encoding,
        population_size=50,
        F=0.8,
        CR=0.9,
        minimize=True,
    ):
        self.fitness_function = fitness_function
        self.F = F  # Scale factor
        self.CR = CR  # Recombination factor
        self.minimize = minimize
        self.population = Population(population_size, encoder)
        self.best_individual = None
        self.best_fitness = np.inf if minimize else -np.inf
        self.fitness_history = []
        self.best_history = []

    def eval(self):
        raw_fitness = np.array(
            [
                self.fitness_function(self.population.encoding.decode(ind))
                for ind in self.population.individuals
            ]
        )
        if self.minimize:
            fitness = np.nan_to_num(raw_fitness, nan=1e10, posinf=1e10, neginf=1e10)
            self.best_fitness = float(fitness.min())
            best_idx = int(fitness.argmin())
        else:
            fitness = np.nan_to_num(raw_fitness, nan=-1e10, posinf=-1e10, neginf=-1e10)
            self.best_fitness = float(fitness.max())
            best_idx = int(fitness.argmax())
        self.best_individual = self.population.individuals[best_idx]
        return fitness

    def evolve(self):
        fitness = self.eval()
        self.fitness_history.append(fitness.mean())
        self.best_history.append(self.best_fitness)
        next_gen = np.empty_like(self.population.individuals)
        pop = self.population.individuals
        NP, D = pop.shape

        for i in range(NP):
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            if isinstance(self.population.encoding, Permutation):
                # PMX Crossover between b and c to create the mutant
                mutant, _ = crossover.PMX(b.copy(), c.copy())
                # Swap mutation over the mutant, with F as mutation rate
                mutant = mutation.swap(mutant, mutation_rate=self.F)
                # Recombination: with probability CR, take mutant gen, otherwise take the fathers
                cross_points = np.random.rand(D) < self.CR
                trial = np.where(cross_points, mutant, pop[i])
                # Validate trial
                # Repair the mutation
                if len(np.unique(trial)) != D:
                    # Repair permutation
                    n = D
                    seen = set()
                    missing = [x for x in range(n) if x not in trial]
                    result = []
                    for x in trial:
                        if x not in seen:
                            result.append(x)
                            seen.add(x)
                        else:
                            result.append(missing.pop(0))
                    trial = np.array(result)
            else:
                # Adjust limits
                if (
                    hasattr(self.population.encoding, "bounds")
                    and self.population.encoding.bounds is not None
                ):
                    lower, upper = np.array(self.population.encoding.bounds).T
                    mutant = np.clip(a + self.F * (b - c), lower, upper)
                else:
                    mutant = a + self.F * (b - c)
                cross_points = np.random.rand(D) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, D)] = True
                trial = np.where(cross_points, mutant, pop[i])
            # Decode
            trial_fitness = self.fitness_function(
                self.population.encoding.decode(trial)
            )
            if (self.minimize and trial_fitness < fitness[i]) or (
                not self.minimize and trial_fitness > fitness[i]
            ):
                next_gen[i] = trial
            else:
                next_gen[i] = pop[i]
        self.population.individuals = next_gen

    def run(self, generations=100, verbose=True):
        for _ in range(generations):
            self.evolve()
