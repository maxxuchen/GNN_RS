import numpy as np
import ray

@ray.remote
def create_shared_noise(count, seed=123):
    """
    Create a large array of noise to be shared by all workers.
    Ray lets us distribute the noise table across a cluster.
    The trainer will create this table and stores it in the rays object store.

    The seed is fixed for this table. The seed for sampling from the table can
    be specified per run.

    Args:
        count(`int`): Size of the shared noise table

    Returns:
        noise(`np.array`): The noise array
    """
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise

class SharedNoiseTable:
    def __init__(self, noise):
        """
        Construct Noise table.

        Args:
            noise(`np.array`): The noise array
        """
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        """
        Get some noise based from the array at i:i+dim

        Args:
            i(`int`): Starting index. From this position until `dim` we take noise.
            dim(`int`): Size of the array which will be returned

        Returns:
            noise(`np.array`)
        """
        return self.noise[i:i + dim]

    def sample_index(self, dim, rng):
        """
        Get an index. It will be sampled randomly without seeding.

        Args:
            dim(`int`): This parameter makes sure we don't sample an index to close to the end of the array
                        to make sure we can take out single chunck of size dim using the returned index.
            rng(`np.random.RandomState`): RNG passed in by each worker to make reproducible
        Returns:
            index(`int`)
        """
        return rng.randint(0, len(self.noise) - dim + 1)
