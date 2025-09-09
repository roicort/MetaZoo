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

def roulette(population: np.ndarray, fitness: np.ndarray, shift: bool = True) -> np.ndarray:
    """
    Roulette Wheel Selection

    Info: In the roulette wheel selection method, a.k.a fitness proportionate selection (FPS), 
    the probability for selecting an individual is directly proportionate to its fitness value.

    Think of this as a roulette wheel where each individual has a slice of the wheel sized according to its fitness. 
    The wheel is spun, and the individual on which the wheel stops is selected. 
    This process is repeated until the desired number of individuals is selected.

    shift: If True, shifts fitness values to be non-negative.
    """
    expected = expected_values(fitness) # This normalizes fitness values to sum to population size
    if shift:
        # Shift expected values to be non-negative
        # Shift works by subtracting the minimum expected value from all expected values
        # and adding a small constant (1e-6) to avoid zero probabilities
        expected = expected - np.min(expected) + 1e-6
    probabilities = expected / np.sum(expected)
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    assert len(selected_indices) == len(population)
    return selected_indices

def stochastic_universal_sampling(population: np.ndarray, fitness: np.ndarray, shift: bool = True) -> np.ndarray:
    """
    Stochastic Universal Sampling (SUS)

    Info: The Stochastic Universal Sampling (SUS) ensures a more uniform selection of individuals 
    based on their fitness values. It is an improvement over the traditional roulette wheel selection method.

    Think of SUS as a roulette wheel where multiple equally spaced pointers are used to select individuals from the population. 
    This method helps to reduce the stochastic noise associated with single-pointer methods like roulette wheel selection.
    """

    expected = expected_values(fitness)
    if shift:
        expected = expected - np.min(expected) + 1e-6
    total_fitness = np.sum(expected)
    point_distance = total_fitness / len(population)
    start_point = np.random.uniform(0, point_distance)
    # Generate pointers for selection
    # These pointers are equally spaced around the roulette wheel, 
    # in other words, they are spaced by point_distance and are evenly distributed.
    pointers = [start_point + i * point_distance for i in range(len(population))]

    selected_indices = []
    cumulative_fitness = np.cumsum(expected)
    for pointer in pointers:
        for idx, cum_fit in enumerate(cumulative_fitness):
            if cum_fit >= pointer:
                selected_indices.append(idx)
                break

    assert len(selected_indices) == len(population)
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
    assert len(selected_indices) == len(population)
    return selected_indices

def tournament(population: np.ndarray, fitness: np.ndarray, K: int = 3, type: str = "standard", replace: bool = True) -> np.ndarray:
    """
    Tournament Selection
    Info: In tournament selection, a subset of individuals is randomly chosen from the population, 
    and the individual with the highest fitness in this subset is selected. 
    This process is repeated until the desired number of individuals is selected.

    K is the tournament size, which determines how many individuals are randomly chosen for each tournament.
    """

    if not replace and K > len(population):
        raise ValueError(f"K={K} is larger than population size={len(population)}")
    
    selected_indices = []
    
    if type == "standard":
        for _ in range(len(population)):
            # Randomly select K individuals for the tournament
            # IF replace is True, individuals can be selected multiple times
            tournament_indices = np.random.choice(len(population), size=K, replace=replace)
            # Select the individual with the highest fitness from the tournament
            # Tie-breaking is handled by np.argmax which returns the first occurrence
            best_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
            selected_indices.append(best_idx)

    elif type == "proportional":
        for _ in range(len(population)):
            # Randomly select K individuals for the tournament
            # IF replace is True, individuals can be selected multiple times
            tournament_indices = np.random.choice(len(population), size=K, replace=replace)
            # Select an individual based on fitness-proportionate probabilities
            tournament_fitness = fitness[tournament_indices]
            # Here, ties are handled by the probability distribution.
            # If multiple individuals have the same fitness, they will have the same probability of being selected.
            total = np.sum(tournament_fitness)
            probabilities = tournament_fitness / total if not np.isnan(total) and not total == 0 else np.full_like(tournament_fitness, 1.0 / len(tournament_fitness)) # Avoid division by zero
            selected_idx = np.random.choice(tournament_indices, p=probabilities)
            selected_indices.append(selected_idx)
    else:
        raise ValueError("Invalid tournament type. Choose 'standard' or 'proportional'.")
    assert len(selected_indices) == len(population)
    return np.array(selected_indices)
    
def uniform(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Uniform Random Selection
    Info: In uniform random selection, individuals are selected randomly from the population with equal probability, 
    regardless of their fitness values. This method does not consider the fitness of individuals and treats all individuals equally.
    """
    selected_indices = np.random.choice(len(population), size=len(population), replace=True)
    assert len(selected_indices) == len(population)
    return selected_indices