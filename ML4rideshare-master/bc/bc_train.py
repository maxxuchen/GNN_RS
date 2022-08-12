import os
import glob
import wandb
import pathlib
import argparse
import numpy as np

from env.simulator import Simulator
from es.utility.matcher import RR_Matcher, VR_Matcher
from bc.bc_config import EPOCH_SAMPLE_NUM, MAX_EPOCHS, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, LR, RR_net_config, VR_net_config, NUM_VEHICLE
from es.utility.model_eval import Eval_Model


def process(policy, data_loader, batch_size, device, optimizer=None):
    """
    Process samples. If an optimizer is given, also train on those samples.
    Parameters
    ----------
    policy : torch.nn.Module
        Model to train/evaluate.
    data_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of training samples.
    device: 'cpu' or 'cuda'
    optimizer : torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    Returns
    -------
    mean_loss : float
        Mean cross entropy loss.
    """
    mean_loss = 0
    n_samples_processed = 0
    count = 0
    batch_loss = 0

    with torch.set_grad_enabled(optimizer is not None):
        for data in data_loader:
            data = data.to(device)
            y_pred_edges, loss = policy(data)
            count += 1
            batch_loss += loss

            # if an optimizer is provided, update parameters
            if optimizer is not None and count == batch_size-1:
                optimizer.zero_grad()
                batch_loss = batch_loss / batch_size
                batch_loss.backward()
                optimizer.step()
                count = 0
                batch_loss = 0

            mean_loss += loss.item() * data_loader.batch_size
            n_samples_processed += data_loader.batch_size

    mean_loss /= n_samples_processed
    return mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='RR match problem or VR match problem.',
        choices=['rr_graph', 'vr_graph'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # initialize wandb
    wandb.init(project='RideSharing_BC', entity='wz2543', mode='offline')

    # hyper parameters
    max_epochs = MAX_EPOCHS
    train_batch_size = TRAIN_BATCH_SIZE  # large batch size will run out of GPU memory
    valid_batch_size = VALID_BATCH_SIZE
    lr = LR

    # set content root path
    DIR = os.path.dirname(os.path.dirname(__file__))
    if args.problem == 'rr_graph':
        train_files_path = os.path.join(DIR, 'bc/samples/train/rr_match/sample_*.pkl')
        valid_files_path = os.path.join(DIR, 'bc/samples/valid/rr_match/sample_*.pkl')
        trained_model_dir = os.path.join(DIR, 'bc/trained_models/rr_match')
        graph_type = 'rr_graph'
    elif args.problem == 'vr_graph':
        train_files_path = os.path.join(DIR, 'bc/samples/train/vr_match/sample_*.pkl')
        valid_files_path = os.path.join(DIR, 'bc/samples/valid/vr_match/sample_*.pkl')
        trained_model_dir = os.path.join(DIR, 'bc/trained_models/vr_match')
        graph_type = 'vr_graph'
    else:
        raise NotImplementedError

    # working directory setup
    os.makedirs(trained_model_dir, exist_ok=True)

    # get files
    train_files = glob.glob(train_files_path)
    valid_files = glob.glob(valid_files_path)

    # cuda setup
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # import pytorch after cuda setup
    import torch
    import torch_geometric
    from models.RGGCN import ResidualGatedGCNModel
    from models.Bipartite_RGGCN import Bipartite_RGGCNModel
    from bc.graph_dataset.dataset import GraphDataset

    # randomization setup
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # valid data setup
    valid_data = GraphDataset(valid_files, graph_type)
    valid_loader = torch_geometric.data.DataLoader(valid_data, 1, shuffle=False)

    # set up network and optimizer
    if graph_type == 'rr_graph':
        policy = ResidualGatedGCNModel(RR_net_config).to(device)
    elif graph_type == 'vr_graph':
        policy = Bipartite_RGGCNModel(VR_net_config).to(device)
    else:
        policy = None

    # set up eval env
    data_path = os.path.join(DIR, 'data/test.csv')
    env = Simulator(NUM_VEHICLE, data_path)
    env.reset()
    selected_day = env.get_time()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_valid_loss = 99999

    for epoch in range(max_epochs + 1):
        # train
        epoch_train_files = rng.choice(train_files, int(np.floor(EPOCH_SAMPLE_NUM / train_batch_size)) * train_batch_size,
                                       replace=True)
        train_data = GraphDataset(epoch_train_files, graph_type)
        train_loader = torch_geometric.data.DataLoader(train_data, 1, shuffle=True)
        train_loss = process(policy, train_loader, train_batch_size, device, optimizer)
        wandb.log({"train_loss": train_loss})

        # validate
        valid_loss = process(policy, valid_loader, valid_batch_size, device, None)
        wandb.log({"valid_loss": valid_loss})

        # store best model parameters
        if valid_loss < best_valid_loss:
            torch.save(policy.state_dict(), pathlib.Path(trained_model_dir) / 'best_params.pkl')

            # once update the model parameters, do model eval
            if graph_type == 'rr_graph':
                rr_model = ResidualGatedGCNModel(RR_net_config)
                rr_policy_path = 'bc/trained_models/rr_match/best_params.pkl'
                checkpoint = torch.load(rr_policy_path, map_location=torch.device('cpu'))
                rr_model.load_state_dict(checkpoint)
                rr_matcher = RR_Matcher('nn', rr_model)
                vr_matcher = VR_Matcher('heu')
            else:
                vr_model = Bipartite_RGGCNModel(VR_net_config)
                vr_policy_path = 'bc/trained_models/vr_match/best_params.pkl'
                checkpoint = torch.load(vr_policy_path, map_location=torch.device('cpu'))
                vr_model.load_state_dict(checkpoint)
                rr_matcher = RR_Matcher('heu')
                vr_matcher = VR_Matcher('nn', vr_model)

            # do evaluation
            Assigned_Request, Distance_Driven, Waiting_Time, Revenue \
                = Eval_Model(env, rr_matcher, vr_matcher, selected_day)
            wandb.log({"assigned_request": Assigned_Request})
            wandb.log({"distance_driven": Distance_Driven})
            wandb.log({"waiting_time": Waiting_Time})
            wandb.log({"revenue": Revenue})

        print(f'Epoch: {epoch}, Train Loss: {train_loss:0.3f}, Valid Loss: {valid_loss:0.3f}. ')

