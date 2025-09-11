
# Combinatorial Optimization Problems

## TSP

::: metazoo.gym.combinatorial.TSP

### Usage Example

```python
from metazoo.gym.combinatorial import TSP
tsp = TSP.Berlin52()
tsp.plot(show_optimal=True)
```

## N-Queens

::: metazoo.gym.combinatorial.NQueens

### Usage Example

```python
nqueens = NQueens(8)
sol = [0, 4, 7, 4, 2, 6, 1, 3]
attacks = nqueens.attacks(sol)
fitness = nqueens(sol)
print(f"Solution: {sol}, Attacks: {attacks}, Fitness: {fitness}")
fig = nqueens.plot(solution=sol, attacks=attacks)
fig.show()
```

