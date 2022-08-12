import ray
import numpy as np
import time

from es.algorithm.noise import SharedNoiseTable


@ray.remote(num_cpus=1)
class Worker:
    """ ES Worker. Distributed with the help of Ray """

    def __init__(self,
                 seed,
                 noise,
                 Solution,
                 subworkers,
                 min_task_runtime=0.2):
        """
        Args:
            seed(`int`): Identifier of the worker
            noise(`np.array`): The noise array
            Solution(`Solution`): The solution class used to instantiate a solution
            min_task_runtime(`float`): Min runtime for a rollout in seconds
        """
        assert seed.dtype == np.int64, "Worker id must be int"

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.noise = SharedNoiseTable(noise)
        self.solution = Solution()
        self.subworkers = subworkers
        self.min_task_runtime = min_task_runtime

    def do_rollouts(self, params, noise_std, n):
        """
        Evalaute params with peturbations.

        Args:
            params(`np.array`): The parameters this worker should use for evaluation
            noise_std(`float`): Gaussian noise standard deviation
            n(`int`): maximum number of rollouts

        Returns:
            scores(`list`), noise_indices(`list`), episodes(`int`), transitions(`int`)
        """
        scores, noise_indices, episodes, transitions = [], [], 0, 0

        # Perform some rollouts with noise.
        task_t_start = time.time()
        while episodes < n:  # and time.time()-task_t_start < self.min_task_runtime:
            noise_index = self.noise.sample_index(dim=self.solution.size(), rng=self.rng)

            ##
            scores_, episodes_, transitions_ = [], 0, 0
            rollout_ids = [subworker.evaluate.remote(params=params, noise_index=noise_index, noise_std=noise_std) for
                           subworker in self.subworkers]
            for eval_info in ray.get(rollout_ids):
                s, e, t = eval_info
                scores_.append(s)
                # episodes_ += e
                # transitions_ += t
            ##

            episodes_ += 1
            transitions_ += 1
            scores_ = np.mean(scores_)

            scores.append(scores_)
            noise_indices.append(noise_index)
            episodes += episodes_
            transitions += transitions_

        return scores, noise_indices, episodes, transitions

    def evaluate(self, params):
        """
        Evalaute params without peturbations.

        Args:
            params(`np.array`): The parameters this worker should use for evaluation
            n(`int`): maximum number of rollouts

        Returns:
            scores(`list`), episodes(`int`), transitions(`int`)
        """
        scores, episodes, transitions = [], 0, 0
        rollout_ids = [subworker.evaluate.remote(params=params) for subworker in self.subworkers]
        for eval_info in ray.get(rollout_ids):
            s, e, t = eval_info
            scores.append(s)
            # episodes += e
            # transitions += t
        episodes += 1
        transitions += 1

        return scores, episodes, transitions
