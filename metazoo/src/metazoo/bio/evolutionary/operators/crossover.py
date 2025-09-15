# Crossover operators for genetic algorithms

import numpy as np

# Continuous

def onepoint(parent1, parent2):
    """
    One-point crossover between two parents.

    A crossover point is selected at random, and the segments after this point are swapped
    between the two parents to create two children.
    """
    point = np.random.randint(1, parent1.shape[0])
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def two_point_crossover(parent1, parent2):
    """
    Two-point crossover between two parents.

    Two crossover points are selected at random, and the segments between these points are swapped
    between the two parents to create two children.
    """
    point1 = np.random.randint(1, parent1.shape[0])
    point2 = np.random.randint(1, parent1.shape[0])
    if point1 > point2:
        point1, point2 = point2, point1
    child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return child1, child2

def k_points_crossover(parent1, parent2, k):
    """
    K-point crossover between two parents.

    K crossover points are selected at random, and the segments between these points are swapped
    between the two parents to create two children.
    """
    
    points = np.random.choice(range(1, parent1.shape[0]), size=k, replace=False)
    points.sort()
    child1 = np.concatenate([parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]])
    child2 = np.concatenate([parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]])
    return child1, child2

# Combinatorial

def PMX(parent1, parent2):
    """
    Partially Mapped Crossover (PMX) for permutation encoding.

    It consists of choosing a subsegment from one of the parents and crossing 
    them while preserving the order and position of as many genes as possible 
    from the other parent, maintaining consistency.
    """

    size = len(parent1) # Length of the permutation
    # Position mapping is used to keep track of the positions of elements in the parents
    p1, p2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int)

    # Initialize the position of each index in the individuals
    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i

    # Choose crossover points
    cxpoint1 = np.random.randint(0, size) # First crossover point
    cxpoint2 = np.random.randint(0, size - 1) # Second crossover point
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the values to be swapped
        temp1 = parent1[i]
        temp2 = parent2[i]
        # Swap the matched value
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2
        # Update the position of the swapped values
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return parent1, parent2