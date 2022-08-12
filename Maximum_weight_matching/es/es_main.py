import os
import ray
import glob
import wandb

from es.algorithm.match_policy import Match_Policy as Policy
from es.algorithm.trainer import Trainer
from es.es_config import NUM_WORKER, STEP_SIZE, EPOCHS, MIN_EVAL, NOISE_STD, SEED

if __name__ == "__main__":
    # set content root path
    DIR = os.path.dirname(os.path.dirname(__file__))
    policy_path = os.path.join(DIR, 'bc/trained_models/best_params.pkl')
    instances_path = os.path.join(DIR, 'bc/samples/train/sample_*.pkl')
    instances_valid = glob.glob(instances_path)

    # initialize wandb
    wandb.init(project='MWM-es', entity='wz2543', mode="offline")

    ray.init()
    trainer = Trainer(Policy=Policy, policy_path=policy_path, instances=instances_valid, seed=SEED,
                      num_workers=NUM_WORKER,
                      step_size=STEP_SIZE, count=1000000,
                      min_task_runtime=100000)
    trainer.train(epochs=EPOCHS, min_evaluations=MIN_EVAL, noise_std=NOISE_STD)
