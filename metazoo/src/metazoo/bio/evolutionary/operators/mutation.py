# Mutation operators for genetic algorithms

import numpy as np

# Continuous - Real

def gaussian(individual, mutation_rate=0.01, mutation_strength=0.1):
    """
    Gaussian mutation for real-valued encoding.
    Each gene in the individual has a certain probability of being mutated by adding Gaussian noise.
    """
    # Apply Gaussian noise to each gene with a certain probability
    if np.random.rand() < mutation_rate:
        # Gaussian Noise with mean 0 and standard deviation mutation_strength
        noise = np.random.normal(0, mutation_strength, size=individual.shape)
        # Create the mutated individual (Individual + Noise)
        individual += noise
        # Clip to keep within bounds [0, 1]
        individual = np.clip(individual, 0, 1)
    return individual

# Continuous - Binary

def flip_bit(individual, mutation_rate=0.01):
    """
    Bit-flip mutation for binary encoding.
    Each bit in the individual has a certain probability of being flipped (0 -> 1 or 1 -> 0).
    """
    # Flip each bit with a certain probability
    if np.random.rand() < mutation_rate:
        # Create a mask for the bits to flip
        mask = np.random.rand(individual.shape[0]) < mutation_rate
        # Flip the bits
        individual[mask] = 1 - individual[mask]
    return individual

# Combinatorial

def swap(individual, mutation_rate=0.01):
    """
    Swap mutation for permutation encoding.
    Two positions in the permutation are selected at random and their values are swapped.
    """
    if np.random.rand() < mutation_rate:
        # Swap two random positions in the permutation
        idx1, idx2 = np.random.choice(range(individual.shape[0]), size=2, replace=False)
        # Swap the elements at idx1 and idx2
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual