import os
import sys
import ray
import time
import wandb
import torch
import numpy as np

from es.algorithm.noise import create_shared_noise, SharedNoiseTable
from es.algorithm.optimizer import Adam
from es.algorithm.worker import Worker
from es.algorithm.utils import compute_centered_ranks
from es.algorithm.subworker import Subworker

from es.es_config import ROLL_OUT_TIMES
from es.es_config import NUM_INSTNACES


class Trainer:

    def __init__(self, Policy, policy_path, instances, seed=1, num_workers=10, step_size=0.01, count=100,
                 min_task_runtime=0.02):
        self.Policy = Policy
        self.solution = self.Policy()
        self.solution_path = policy_path
        self.solution.load(path=self.solution_path)


        self.optimizer = Adam(solution=self.solution, step_size=step_size)

        self.noise_id = create_shared_noise.remote(count=count)
        self.noise = SharedNoiseTable(ray.get(self.noise_id))

        rng = np.random.RandomState(seed)
        self.worker_seeds = rng.randint(0, 100000000, size=num_workers)
        self.instances = instances[:NUM_INSTNACES]
        self.min_task_runtime = min_task_runtime

        # bookkeeping
        self.episodes_so_far = 0
        self.reward_list = 0
        self.t0 = time.time()
        self.best_score = float("inf")

    def train(self, epochs=100, min_evaluations=10, noise_std=0.01):
        for e in range(epochs):
            # current solution
            theta = self.solution.get_params()
            theta_id = ray.put(theta)
            self.step(theta_id=theta_id, min_evaluations=min_evaluations, noise_std=noise_std)

    def step(self, theta_id, min_evaluations, noise_std):
        """
        Perform one training step. This means one parameter update.
            1. store the current solution in the object store
            2. evaluate perturbed solutions with workers
            3. calculate parameter update
            4. update parameters
            5. return results for statistic collection

        Returns:
            results(`dict`): Information about the training step.
        """

        t0 = time.time()

        num_episodes = 0
        num_transitions = 0
        scores = []
        noise_indices = []

        while num_episodes < min_evaluations:
            # subworkers = []
            workers = []
            for worker_seed in self.worker_seeds:
                # subworker = [
                #     Subworker.remote(seed=worker_seed, noise=self.noise_id, Solution=self.Policy, instance=instance) for
                #     instance in self.instances]
                worker = Worker.remote(seed=worker_seed, noise_id=self.noise_id, Solution=self.Policy,
                                       instances=self.instances, min_task_runtime=self.min_task_runtime)
                # subworkers.append(subworker)
                workers.append(worker)

            # workers[0:1] for evaluation without permutation, workers[1:] for update with permutation
            rollout_ids = [worker.evaluate.remote(params=theta_id) for worker in workers[0:1]] + [
                worker.do_rollouts.remote(params=theta_id, noise_std=noise_std, n=ROLL_OUT_TIMES) for worker in
                workers[1:]]
            results = ray.get(rollout_ids)

            # EVALUATE
            score, _, _ = results[0]
            # we want to minimize the score
            if np.mean(score) < self.best_score:
                self.best_score = np.mean(score)
                self.solution.save(path=self.solution_path)
                for worker in workers:
                    ray.get(worker.save.remote(params=theta_id, path=self.solution_path))
                # for subworker in subworkers:
                #     for s in subworker:
                #         ray.get(s.save.remote(params=theta_id, path=self.solution_path))

            # ROLLOUTS
            for worker_info in results[1:]:
                worker_scores, worker_noise_indices, worker_episodes, worker_transitions = worker_info
                scores += worker_scores
                noise_indices += worker_noise_indices
                num_episodes += worker_episodes
                num_transitions += worker_transitions
        assert len(scores) == len(noise_indices)

        # Record current score
        wandb.log({"solution score": np.mean(score)})
        print("currrent solution score: ", np.mean(score))
        # print("current best solution score: ", self.best_score)

        # UPDATE
        noises = np.array([self.noise.get(i=index, dim=self.solution.size()) for index in noise_indices])
        scores = np.array(scores)
        # mean_scores = np.mean(scores, axis=1)

        # _mina, _maxa, _meana = np.min(mean_scores), np.max(mean_scores), np.mean(mean_scores)
        _min, _max, _mean, _std = np.min(scores), np.max(scores), np.mean(scores), np.std(scores)

        # normalized_scores = (scores-_min)/(_max-_min)
        # scores_ = np.mean(normalized_scores, axis=1)
        # scores_ = np.mean(scores, axis=1)

        # scores_shaped = compute_centered_ranks(scores)
        # scores_shaped0 = compute_centered_ranks(scores[:, 0])
        # scores_shaped1 = compute_centered_ranks(scores[:, 1])
        # scores_ = np.concatenate([np.expand_dims(scores_shaped0, axis=1), np.expand_dims(scores_shaped1, axis=1)], axis=1)
        # scores_ = np.mean(scores_, axis=1)
        scores_shaped = compute_centered_ranks(scores)

        gradient = np.mean(scores_shaped[:, np.newaxis] * noises, axis=0) / noise_std
        assert gradient.shape[0] == self.solution.size()
        self.optimizer.step(gradient)
        self.episodes_so_far += num_episodes

        tf = time.time()
        print("%d rollouts, time: %.2f, min_score: %.3f, max_score: %.3f, average_score: %.3f, std_scores: %.3f" % (num_episodes, tf-t0, _min, _max, _mean, _std))

    def kill(self):
        """ Terminate all workers. """
        for w in self.workers:
            w.__ray_terminate__.remote()
