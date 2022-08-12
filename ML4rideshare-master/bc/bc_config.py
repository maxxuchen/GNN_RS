# Model Setting
# NUM_NODES = 1
NUM_VEHICLE = 20


# Training Setting
EPOCH_SAMPLE_NUM = 100  # the number of samples trained in one epoch
MAX_EPOCHS = 50
TRAIN_BATCH_SIZE = 16  # large batch size will run out of GPU memory
VALID_BATCH_SIZE = 16
LR = 5e-4

# Network config
RR_net_config = {
    'node_dim': 5,
    'edge_dim': 3,
    'voc_edges_in': 2,
    'voc_edges_out': 2,
    'hidden_dim': 64,
    'num_layers': 10,
    'mlp_layers': 2,
    'aggregation': 'mean'
}

VR_net_config = {
    'left_node_dim': 2,
    'right_node_dim': 9,
    'edge_dim': 2,
    'voc_edges_in': 2,
    'voc_edges_out': 2,
    'hidden_dim': 64,
    'num_layers': 10,
    'mlp_layers': 2,
    'aggregation': 'mean'
}