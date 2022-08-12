import ray
import numpy as np

from es.algorithm.noise import SharedNoiseTable


@ray.remote(num_cpus=1)
class Subworker:
    def __init__(self, seed, noise, Solution, instance):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.noise = SharedNoiseTable(noise)
        self.solution = Solution()
        self.instance = instance

    def evaluate(self, params, noise_index=None, noise_std=None):
        if noise_index is not None and noise_std is not None:
            perturbation = noise_std * self.noise.get(i=noise_index, dim=self.solution.size())
            self.solution.set_params(params + perturbation)
        else:
            self.solution.set_params(params)
        return self.solution.evaluate(instance=self.instance)

    def save(self, params, path):
        self.solution.set_params(params)
        self.solution.save(path)