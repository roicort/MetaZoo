# Particle Swarm Optimization (PSO)

import numpy as np
from metazoo.bio.utils import Encoding, Real, Permutation


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, minimize=True):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_value =  np.inf if minimize else -np.inf
class Swarm:
    def __init__(self, population_size: int, encoding: Encoding, minimize: bool = False):
        self.size = population_size
        self.encoding = encoding
        self.particles = []
        for _ in range(population_size):
            position = encoding.encode(1).flatten()
            # Real encoding
            if isinstance(encoding, Real):
                velocity = np.zeros(encoding.dim)
            # Permutation encoding (velocity as list of swaps)
            elif isinstance(encoding, Permutation):
                velocity = []  # List of swaps or moves
            else:
                raise ValueError("Unknown encoding type")
            self.particles.append(Particle(position=position, velocity=velocity, minimize=minimize))

    def __len__(self):
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, index):
        return self.particles[index]

    def __setitem__(self, index, value):
        self.particles[index] = value

def get_permutation_swaps(current: np.ndarray, target: np.ndarray):
    """Get the list of swaps needed to convert current permutation to target permutation."""
    swaps = []
    curr = current.copy()
    for i in range(len(curr)):
        if curr[i] != target[i]:
            swap_idx = np.where(curr == target[i])[0][0]
            swaps.append((i, swap_idx))
            # Perform swap
            curr[i], curr[swap_idx] = curr[swap_idx], curr[i]
    return swaps


def apply_permutation_swaps(current: np.ndarray, swaps: list):
    """Apply a list of swaps to a permutation."""
    perm = current.copy()
    for i, j in swaps:
        perm[i], perm[j] = perm[j], perm[i]
    return perm


class ParticleSwarmOptimizer:
    """A simple Particle Swarm Optimization (PSO) implementation."""

    def __init__(
        self,
        fitness_function,
        encoder: Encoding,
        population_size=30,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        minimize=True,
    ):
        """
        Initialize the PSO optimizer.
        """
        self.fitness_function = fitness_function
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.minimize = minimize

        self.swarm = Swarm(population_size, encoder, minimize=minimize)
        self.global_best_position = None
        self.global_best_fitness = np.inf if minimize else -np.inf
        self.fitness_history = []
        self.best_history = []

    def summary(self):
        """
        Print all relevant information about the PSO instance using rich.
        """
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Particle Swarm Optimizer Summary")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="bold magenta")
        table.add_row("Particles", str(self.swarm.size))
        table.add_row("Dimensions", str(self.swarm.encoding.genome_length))
        table.add_row("Inertia", str(self.inertia))
        table.add_row("Cognitive", str(self.cognitive))
        table.add_row("Social", str(self.social))
        table.add_row("Bounds", str(getattr(self.swarm.encoding, "bounds", "None")))
        table.add_row("Fitness Function", self.fitness_function.__name__)
        table.add_row("Minimize", str(self.minimize))
        console = Console()
        console.print(table)

    def eval(self):
        """
        Evaluate the fitness of all particles and update the global best.
        """

        raw_fitness = np.array(
            [
                self.fitness_function(self.swarm.encoding.decode(particle.position))
                for particle in self.swarm
            ]
        )

        if self.minimize:
            fitness = np.nan_to_num(raw_fitness, nan=1e10, posinf=1e10, neginf=1e10)
            self.best_fitness = float(fitness.min())
            best_idx = int(fitness.argmin())
            fitness_transformed = np.max(fitness) - fitness
        else:
            fitness = np.nan_to_num(raw_fitness, nan=-1e10, posinf=-1e10, neginf=-1e10)
            self.best_fitness = float(fitness.max())
            best_idx = int(fitness.argmax())
            fitness_transformed = fitness

        self.best_individual = self.swarm[best_idx]

        return fitness, fitness_transformed

    def step(self):
        """
        Perform one PSO iteration: update velocities and positions.
        """

        fitness, fitness_transformed = self.eval()
        self.fitness_history.append(fitness.mean())
        self.best_history.append(self.best_fitness)

        # Update swarm for Real encoding
        if isinstance(self.swarm.encoding, Real):
            for i, particle in enumerate(self.swarm):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = (
                    self.cognitive
                    * r1
                    * (particle.best_position - particle.position)
                )
                social_velocity = (
                    self.social
                    * r2
                    * (self.best_individual.position - particle.position)
                )
                particle.velocity = (
                    self.inertia * particle.velocity
                    + cognitive_velocity
                    + social_velocity
                )
                particle.position += particle.velocity

                # Clip position to bounds if defined
                if hasattr(self.swarm.encoding, "bounds") and self.swarm.encoding.bounds is not None:
                    lower, upper = np.array(self.swarm.encoding.bounds).T
                    particle.position = np.clip(particle.position, lower, upper)

        if isinstance(self.swarm.encoding, Permutation):
            for i, particle in enumerate(self.swarm):
                # Update velocity as list of swaps
                new_velocity = []

                # Cognitive component
                cognitive_swaps = get_permutation_swaps(
                    particle.position, particle.best_position
                )
                num_cognitive_swaps = int(self.cognitive * np.random.rand() * len(cognitive_swaps))
                new_velocity.extend(cognitive_swaps[:num_cognitive_swaps])

                # Social component
                social_swaps = get_permutation_swaps(
                    particle.position, self.best_individual.position
                )
                num_social_swaps = int(self.social * np.random.rand() * len(social_swaps))
                new_velocity.extend(social_swaps[:num_social_swaps])

                # Update particle velocity and position
                particle.velocity = new_velocity
                particle.position = apply_permutation_swaps(
                    particle.position, particle.velocity
                )

            # Update personal best
            if (self.minimize and fitness[i] < particle.best_value) or (
                not self.minimize and fitness[i] > particle.best_value
            ):
                particle.best_position = particle.position.copy()
                particle.best_value = fitness[i]

        # Update global best
        if (self.minimize and self.best_fitness < self.global_best_fitness) or (
            not self.minimize and self.best_fitness > self.global_best_fitness
        ):
            self.global_best_fitness = self.best_fitness
            self.global_best_position = self.best_individual.position.copy()

    def run(self, iterations=100, history=False, verbose=True):
        """
        Run the main PSO loop.
        """
        pop_history = []
        best_history = []
        if verbose:
            from rich.progress import Progress

            with Progress() as progress:
                task = progress.add_task("Optimizing...", total=iterations)
                for _ in range(iterations):
                    self.step()
                    pop_history.append([p.position.copy() for p in self.swarm])
                    best_history.append(self.global_best_position.copy())
                    progress.advance(task)
        else:
            for _ in range(iterations):
                self.step()
                pop_history.append([p.position.copy() for p in self.swarm])
                best_history.append(self.global_best_position.copy())
        if history:
            return best_history, pop_history

    def fitness_plot(self, best=False):
        """
        Plot the fitness history using plotly.
        """
        import plotly.express as px

        if self.fitness_history:
            if not best:
                fig = px.line(
                    y=np.array(self.fitness_history),
                    labels={"x": "Iteration", "y": "Fitness"},
                    title="Fitness History",
                )
                return fig
            else:
                fig = px.line(
                    y=np.array(self.best_history),
                    labels={"x": "Iteration", "y": "Best Fitness"},
                    title="Best Fitness History",
                )
                return fig
        else:
            raise ValueError("No fitness history to plot. Run the algorithm first.")
