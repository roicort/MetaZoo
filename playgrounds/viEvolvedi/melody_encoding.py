from typing import Tuple
import numpy as np
from metazoo.bio.utils import Encoding

class MelodyEncoding(Encoding):
    """
    Codifica una melodía como una secuencia de notas MIDI y duraciones rítmicas, incluyendo silencios.
    Usa 0 como nota para indicar silencio.
    """
    def __init__(self, note_range: Tuple[int, int] = (60, 72), durations: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0), length: int = 64, silence_prob: float = 0.2):
        self.note_range = note_range
        self.durations = durations
        self.length = length
        self.genome_length = length
        self.silence_prob = silence_prob

    def encode(self, population_size: int) -> np.ndarray:
        notes = np.random.randint(self.note_range[0], self.note_range[1], size=(population_size, self.length))
        # Asigna silencios aleatoriamente
        silence_mask = np.random.rand(population_size, self.length) < self.silence_prob
        notes[silence_mask] = 0
        durations = np.random.choice(self.durations, size=(population_size, self.length))
        return np.stack((notes, durations), axis=-1)

    def decode(self, individual: np.ndarray) -> np.ndarray:
        return individual