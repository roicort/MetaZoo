# Usage

Here is a basic example of how to use MetaZoo:

```python
from metazoo.bio.evolutionary.ga import GeneticAlgorithm
from metazoo.bio.utils import encoding

# Create a Genetic Algorithm instance
encoder = encoding.Binary(...)

ag = GeneticAlgorithm(encoder=encoder, ...)
result = ag.run(...)
print(result)
```

For more advanced usage, see the module documentation below.
