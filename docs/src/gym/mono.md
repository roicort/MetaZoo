# Mono

This module provides mono-objective optimization test functions for evolutionary experiments.

## Example
```python
from metazoo.gym.mono import Function

available = Function().available_functions
print(available)

#['Rastrigin', 'Ackley', 'Sphere', 'Rosenbrock', 'Beale', 'GoldsteinPrice', 'Booth', 'Bukin', 'Matyas', 'Levi_N13', 'Griewank', 'Himmelblau', 'ThreeHumpCamel', 'Easom', 'Cross_In_Tray', 'EggHolder', 'HolderTable', 'McCormick', 'Schaffer_N2', 'StyblinskiTang', 'Shekel']

fitness_function = Function('Ackley', reverse=False)
print(fitness_function.bounds)

# [(-5, 5), (-5, 5)]

fitness_function.plot(bounds=fitness_function.bounds, dim=2, num_points=100, mode='surface')
fitness_function.plot(bounds=fitness_function.bounds, dim=2, num_points=100, mode='contour')
```

## Reference

### Class: Function

::: metazoo.gym.mono.Function

### Methods

#### available_functions

Returns a list of available functions.

::: metazoo.gym.mono.Function.available_functions

#### bounds

Gets the bounds of the function.

::: metazoo.gym.mono.Function.bounds

#### plot

Plots the function in 2D or 3D.

::: metazoo.gym.mono.Function.plot

---
