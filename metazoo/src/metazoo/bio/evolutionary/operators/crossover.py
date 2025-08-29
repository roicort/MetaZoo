# Crossover operators for genetic algorithms

import numpy as np

def onepoint(parent1, parent2):
    point = np.random.randint(1, parent1.shape[0])
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(1, parent1.shape[0])
    point2 = np.random.randint(1, parent1.shape[0])
    if point1 > point2:
        point1, point2 = point2, point1
    child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return child1, child2

def k_points_crossover(parent1, parent2, k):
    points = np.random.choice(range(1, parent1.shape[0]), size=k, replace=False)
    points.sort()
    child1 = np.concatenate([parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]])
    child2 = np.concatenate([parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]])
    return child1, child2