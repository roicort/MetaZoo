from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, Tuple

class Encoding(ABC):
    @abstractmethod
    def encode(self, population_size: int) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, individual: np.ndarray) -> np.ndarray:
        pass

class Binary(Encoding):
    """
    Solves real-valued problems using binary encoding.
    Each variable is represented by a fixed number of bits, determined by the desired precision and the variable bounds.
    """
    def __init__(self, precision: int = 3, bounds: Sequence[Tuple[float, float]] = None):
        self.precision = precision
        self.bounds: Sequence[Tuple[float, float]] = bounds
        if bounds is None:
            raise ValueError("Bounds must be provided for binary encoding.")
        self.dim = len(bounds)  # Number of variables

        self.epsilon = 10 ** (
            -precision
        )  # ε = Obtaianable accuracy for binary encoding

        # Bits per variable for binary encoding
        # This is calculated based on the precision and the range of each variable
        # Using the formula: n = ceil(log2((XU - XL) / ε))
        # where XU and XL are the upper and lower bounds of the variable, and ε is the desired precision
        # We can understand ε as the smallest difference we want to be able to represent between two values of the variable
        self.bits_per_var = max(
            int(np.ceil(np.log2((xu - xl) / self.epsilon))) for xl, xu in self.bounds
        )
        # We use max to ensure we have enough bits for the most constrained variable

        self.genome_length = self.bits_per_var * self.dim

    def decode(self, individual: np.ndarray) -> np.ndarray:
        """
        Decodes a binary individual into its real-valued representation.
        """
        bits_per_var = self.bits_per_var
        decoded = []
        for i, (a, b) in enumerate(self.bounds):
            bits = individual[i * bits_per_var : (i + 1) * bits_per_var]
            value = int("".join(str(int(bit)) for bit in bits), 2)
            max_value = 2**bits_per_var - 1
            real_value = a + (b - a) * value / max_value
            decoded.append(real_value)
        return np.array(decoded)

    def encode(self, population_size: int) -> np.ndarray:
        return np.random.randint(0, 2, size=(population_size, self.genome_length))

class Real(Encoding):
    """
    Solves real-valued problems using real encoding.
    Each variable is represented directly by a real number within the specified bounds.
    """
    def __init__(self, bounds: Sequence[Tuple[float, float]] = None):
        self.bounds: Sequence[Tuple[float, float]] = bounds
        if bounds is None:
            raise ValueError("Bounds must be provided for real encoding.")
        self.dim = len(bounds)  # Number of variables
        self.genome_length = self.dim

    def decode(self, individual: np.ndarray) -> np.ndarray:
        return individual  # For real encoding, the individual is already in real values

    def encode(self, population_size: int):
        arr = np.empty((population_size, self.genome_length))
        for j, (low, high) in enumerate(self.bounds):
            arr[:, j] = np.random.uniform(low, high, size=population_size)
        return arr

class Permutation(Encoding):
    """
    Solves combinatorial problems using permutation encoding.
    Each individual is represented as a permutation of integers from 0 to n-1, where n is the size of the permutation.
    """
    def __init__(self, permutation_size: int):
        self.genome_length = permutation_size

    def decode(self, individual: np.ndarray) -> np.ndarray:
        return individual  # For permutation encoding, the individual is already in permutation form

    def encode(self, population_size: int) -> np.ndarray:
        return np.array([np.random.permutation(self.genome_length) for _ in range(population_size)])


class encoding:
    Binary = Binary
    Real = Real
    Permutation = Permutation

class Population:
    def __init__(self, population_size: int, encoding: Encoding):
        self.size = population_size
        self.encoding = encoding
        self.individuals = encoding.encode(population_size)