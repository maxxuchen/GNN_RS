import os
import sys
import glob
import wandb
import pathlib
import argparse
import numpy as np


from common.greedy_search import Greedy_Search
from bc_config import EPOCH_SAMPLE_NUM, MAX_EPOCHS, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, LR


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
    wandb.init(project='MWM-bc', entity='wz2543', mode='online')

    # hyper parameters
    max_epochs = MAX_EPOCHS
    train_batch_size = TRAIN_BATCH_SIZE  # large batch size will run out of GPU memory
    valid_batch_size = VALID_BATCH_SIZE
    lr = LR

    net_config = {
        'node_dim': 2,
        'voc_edges_in': 2,
        'voc_edges_out': 2,
        'hidden_dim': 300,
        'num_layers': 15,
        'mlp_layers': 2,
        'aggregation': 'mean'
    }

    # get dir
    DIR = os.path.dirname(__file__)
    train_files_path = os.path.join(DIR, 'samples/train/sample_*.pkl')
    valid_files_path = os.path.join(DIR, 'samples/valid/sample_*.pkl')
    trained_model_dir = os.path.join(DIR, 'trained_models')

    train_files = glob.glob(train_files_path)[:10000]
    valid_files = glob.glob(valid_files_path)[:2000]

    # working directory setup
    os.makedirs(trained_model_dir, exist_ok=True)

    # cuda setup
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # import pytorch **after** cuda setup
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from model.RGGCN import ResidualGatedGCNModel
    from common.dataset_utility import GraphDataset

    # randomization setup
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # data setup
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, 1, shuffle=False)

    policy = ResidualGatedGCNModel(net_config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    best_loss = 99999

    for epoch in range(max_epochs + 1):
        # train
        epoch_train_files = rng.choice(train_files, int(np.floor(EPOCH_SAMPLE_NUM / train_batch_size)) * train_batch_size,
                                       replace=True)
        train_data = GraphDataset(epoch_train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, 1, shuffle=True)
        train_loss = process(policy, train_loader, train_batch_size, device, optimizer)

        wandb.log({"train_loss": train_loss})

        # validate
        valid_loss = process(policy, valid_loader, valid_batch_size, device, None)
        wandb.log({"valid_loss": valid_loss})

        # store best model parameters
        if valid_loss < best_loss:
            torch.save(policy.state_dict(), pathlib.Path(trained_model_dir) / 'best_params.pkl')

        print(f'Epoch: {epoch}, Train Loss: {train_loss:0.3f}, Valid Loss: {valid_loss:0.3f}. ')
