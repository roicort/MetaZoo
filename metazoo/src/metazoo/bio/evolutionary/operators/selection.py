# Selection Operators

import numpy as np

def expected_values(fitness: np.array) -> np.array:
    average_fitness = np.sum(fitness)
    number_individuals = len(fitness)
    return fitness / average_fitness * number_individuals

def roulette(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    expected = expected_values(fitness)
    selected_indices = np.random.choice(len(population), size=len(population), p=expected/expected.sum())
    return selected_indices