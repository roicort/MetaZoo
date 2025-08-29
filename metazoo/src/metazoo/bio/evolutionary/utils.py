import numpy as np

class Population:
	def __init__(self, population_size, genome_length, encoding='real', bounds=(-1, 1)):
		self.population_size = population_size
		self.genome_length = genome_length
		self.encoding = encoding
		self.bounds = bounds

	def initialize(self):
		if self.encoding == 'binary':
			return np.random.randint(0, 2, size=(self.population_size, self.genome_length))
		elif self.encoding == 'boolean':
			return np.random.choice([True, False], size=(self.population_size, self.genome_length))
		elif self.encoding == 'real':
			if isinstance(self.bounds, (list, tuple)) and isinstance(self.bounds[0], (list, tuple)):
				arr = np.empty((self.population_size, self.genome_length))
				for j, (low, high) in enumerate(self.bounds):
					arr[:, j] = np.random.uniform(low, high, size=self.population_size)
				return arr
			else:
				low, high = self.bounds
				return np.random.uniform(low, high, size=(self.population_size, self.genome_length))
		else:
			raise ValueError("Encoding must be 'binary', 'boolean' o 'real'")
