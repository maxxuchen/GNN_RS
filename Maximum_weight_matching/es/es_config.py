# Network config
net_config = {
    'node_dim': 2,
    'voc_edges_in': 2,
    'voc_edges_out': 2,
    'hidden_dim': 64,
    'num_layers': 8,
    'mlp_layers': 2,
    'aggregation': 'mean'
}

# test config
ROLL_OUT_TIMES = 20  # the number of rolls out in es (originally 50), i.e., how many different permutations

NUM_INSTNACES = 5  # the number of instances used in es (originally 100), i.e., the number of subworkers in one worker

NUM_WORKER = 2  # (originally 24)
STEP_SIZE = 1e-4

SEED = 1
EPOCHS = 50  # (originally 20)
MIN_EVAL = 100  # (originally 1150), the number of episode per update
NOISE_STD = 0.1

#
#
# # suggest config
# EVAL_TIME_LIMT = 5*60  # the limit of time to eval the model (originally 5*60)
# ROLL_OUT_TIMES = 50  # the number of rolls out in es (originally 50)
# NUM_INSTNACES = 100  # the number of instances used in es (originally 100)
#
# NUM_WORKER = 24  # (originally 24)
# STEP_SIZE = 1e-4
#
# SEED = 1
# EPOCHS = 20  # (originally 20)
# MIN_EVAL = 1150  # (originally 1150)
# NOISE_STD = 0.1