# Mutation operators for genetic algorithms

import numpy as np

def gaussian(individual, mutation_rate=0.01, mutation_strength=0.1):
    if np.random.rand() < mutation_rate:
        noise = np.random.normal(0, mutation_strength, size=individual.shape)
        individual += noise
    return individual

def flip_bit(individual, mutation_rate=0.01):
    mask = np.random.rand(individual.shape[0]) < mutation_rate
    individual[mask] = 1 - individual[mask]
    return individual