import os
import ray
import glob
import torch
import wandb
import argparse
from es.algorithm.trainer import Trainer
from es.es_config import NUM_WORKER, STEP_SIZE, EPOCHS, MIN_EVAL, NOISE_STD, SEED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='RR match problem or VR match problem.',
        choices=['rr_graph', 'vr_graph'],
    )
    args = parser.parse_args()

    # set content root path
    DIR = os.path.dirname(os.path.dirname(__file__))
    if args.problem == 'rr':
        # train rr_match_policy
        from es.utility.rr_match_policy import RR_Match_Policy as Policy
        from es.es_config import RR_net_config as Net_config
        from models.RGGCN import ResidualGatedGCNModel as Model
        policy_path = os.path.join(DIR, 'bc/trained_models/rr_match/best_params.pkl')
    else:
        # train vr_match_policy
        from es.utility.rr_match_policy import VR_Match_Policy as Policy
        from es.es_config import VR_net_config as Net_config
        from models.Bipartite_RGGCN import Bipartite_RGGCNModel as Model
        policy_path = os.path.join(DIR, 'bc/trained_models/vr_match/best_params.pkl')

    # initialize empty network parameter
    # if you have already trained the network via BC, you can comment two lines below
    network = Model(Net_config)
    torch.save(network.state_dict(), policy_path)

    # load data
    instances_path = os.path.join(DIR, 'es/samples/sample*.csv')
    instances = glob.glob(instances_path)

    # initialize wandb
    wandb.init(project='RideSharing_ES', entity='wz2543', mode="offline")

    ray.init()
    trainer = Trainer(Policy=Policy, policy_path=policy_path, instances=instances, seed=SEED,
                      num_workers=NUM_WORKER,
                      step_size=STEP_SIZE, count=1000000,
                      min_task_runtime=100000)
    trainer.train(epochs=EPOCHS, min_evaluations=MIN_EVAL, noise_std=NOISE_STD)
