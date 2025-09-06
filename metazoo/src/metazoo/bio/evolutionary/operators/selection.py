# Selection Operators
# NOTE: These operators assume that higher fitness values are better.

import numpy as np

def expected_values(fitness: np.array) -> np.array:
    """
    Calculate expected values for a given fitness array.
    
    The expected value for each individual is calculated as:
        expected_value = (fitness / average_fitness) * number_of_individuals
    
    This function normalizes the fitness values to ensure that they sum up to the population size.
    """
    average_fitness = np.sum(fitness)
    number_of_individuals = len(fitness)
    return fitness / average_fitness * number_of_individuals

def roulette(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Roulette Wheel Selection

    Info: In the roulette wheel selection method, a.k.a fitness proportionate selection (FPS), 
    the probability for selecting an individual is directly proportionate to its fitness value.

    Think of this as a roulette wheel where each individual has a slice of the wheel sized according to its fitness. 
    The wheel is spun, and the individual on which the wheel stops is selected. 
    This process is repeated until the desired number of individuals is selected.
    """
    expected = expected_values(fitness)
    selected_indices = np.random.choice(len(population), size=len(population), p=expected/expected.sum())
    return selected_indices

def stochastic_universal_sampling(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Stochastic Universal Sampling (SUS)

    Info: The Stochastic Universal Sampling (SUS) ensures a more uniform selection of individuals 
    based on their fitness values. It is an improvement over the traditional roulette wheel selection method.

    Think of SUS as a roulette wheel where multiple equally spaced pointers are used to select individuals from the population. 
    This method helps to reduce the stochastic noise associated with single-pointer methods like roulette wheel selection.
    """

    expected = expected_values(fitness)
    total_fitness = np.sum(expected)
    point_distance = total_fitness / len(population)
    start_point = np.random.uniform(0, point_distance)
    pointers = [start_point + i * point_distance for i in range(len(population))]

    selected_indices = []
    cumulative_fitness = np.cumsum(expected)
    for pointer in pointers:
        for idx, cum_fit in enumerate(cumulative_fitness):
            if cum_fit >= pointer:
                selected_indices.append(idx)
                break

    return np.array(selected_indices)

def rank(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Rank Selection
    Info: The rank-based selection method is similar to the roulette wheel selection, 
    but instead of directly using the fitness values to calculate the probabilities 
    for selecting each individual, we use the fitness values to order the individuals and assign selection probabilities based on their positions.
    The actual selection probabilities are then assigned based on the rank of each individual.
    """
    ranks = np.argsort(np.argsort(fitness)) # Get ranks
    # Convert ranks to probabilities
    total_rank = np.sum(ranks) # Sum of ranks
    probabilities = ranks / total_rank if total_rank > 0 else np.ones_like(ranks) / len(ranks) # Avoid division by zero
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities) # Select individuals based on probabilities
    return selected_indices

def tournament(population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    """
    Tournament Selection
    Info: In tournament selection, a subset of individuals is randomly chosen from the population, 
    and the individual with the highest fitness in this subset is selected. 
    This process is repeated until the desired number of individuals is selected.
    """
    selected_indices = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)
    return np.array(selected_indices)