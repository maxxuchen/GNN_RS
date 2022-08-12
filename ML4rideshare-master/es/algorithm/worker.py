import ray
import numpy as np
import time

from es.algorithm.noise import SharedNoiseTable
from es.algorithm.subworker import Subworker


@ray.remote(num_cpus=2)
class Worker:
    """ ES Worker. Distributed with the help of Ray """

    def __init__(self,
                 seed,
                 noise_id,
                 Solution,
                 instances,
                 min_task_runtime=0.2):
        """
        Args:
            seed(`int`): Identifier of the worker
            noise_id: The noise_id, SharedNoiseTable(noise_id) is the noise table
            Solution(`Solution`): The solution class used to instantiate a solution
            min_task_runtime(`float`): Min runtime for a rollout in seconds
        """
        assert seed.dtype == np.int64, "Worker id must be int"

        # self.seed = seed
        # self.rng = np.random.RandomState(self.seed)
        # if we have seed for rng, in every epoch, the permutation will be same
        self.rng = np.random.RandomState()
        self.noise = SharedNoiseTable(noise_id)
        self.solution = Solution()
        self.instances = instances
        self.min_task_runtime = min_task_runtime

        self.subworkers = [
            Subworker(noise_id=noise_id, Solution=Solution, instance=instance)
            for instance in self.instances
        ]

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
            rollout = [subworker.evaluate(params=params, noise_index=noise_index, noise_std=noise_std) for
                           subworker in self.subworkers]
            for eval_info in rollout:
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
        rollout = [subworker.evaluate(params=params) for subworker in self.subworkers]
        for eval_info in rollout:
            s, e, t = eval_info
            scores.append(s)
            # episodes += e
            # transitions += t
        episodes += 1
        transitions += 1

        return scores, episodes, transitions

    def save(self, params, path):
        self.solution.set_params(params)
        self.solution.save(path)
