# Operators

This section documents the available genetic algorithm operators in MetaZoo.

Evolutionary operators are essential components of genetic algorithms that mimic the process of natural selection and evolution. 
They are used to generate new candidate solutions (offspring) from existing ones (parents) in order to explore the solution space and optimize a given objective function. 
The main types of evolutionary operators include crossover, mutation, and selection.


## Crossover

Crossover operators combine the genetic information of two parent solutions to produce one or more offspring. 
This process is inspired by biological reproduction and is crucial for introducing diversity into the population. 

Common crossover techniques include:

For parametric problems:

::: metazoo.bio.evolutionary.operators.crossover.onepoint
::: metazoo.bio.evolutionary.operators.crossover.two_point_crossover
::: metazoo.bio.evolutionary.operators.crossover.k_points_crossover

For permutation-based representations:

::: metazoo.bio.evolutionary.operators.crossover.PMX

## Mutation

Mutation operators introduce random changes to individual solutions, helping to maintain genetic diversity within the population and preventing premature convergence to suboptimal solutions.

Common mutation techniques include:

For parametric problems:

::: metazoo.bio.evolutionary.operators.mutation.gaussian
::: metazoo.bio.evolutionary.operators.mutation.flip_bit

For permutation-based representations:

::: metazoo.bio.evolutionary.operators.mutation.swap

## Selection

Selection operators determine which individuals from the current population are chosen to contribute to the next generation. 
This process is inspired by the concept of "survival of the fittest" in natural selection

::: metazoo.bio.evolutionary.operators.selection.uniform
::: metazoo.bio.evolutionary.operators.selection.expected_values
::: metazoo.bio.evolutionary.operators.selection.roulette
::: metazoo.bio.evolutionary.operators.selection.stochastic_universal_sampling
::: metazoo.bio.evolutionary.operators.selection.rank
::: metazoo.bio.evolutionary.operators.selection.tournament

