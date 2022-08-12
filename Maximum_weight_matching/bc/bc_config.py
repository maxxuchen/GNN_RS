# Graph setting
NUM_NODES = [30, 35, 40, 45, 50]  # number of vertex
RADIUS = 0.3  # the minimum radius that will create edge

# Training setting
EPOCH_SAMPLE_NUM = 100  # the number of samples trained in one epoch
MAX_EPOCHS = 150
TRAIN_BATCH_SIZE = 16  # large batch size will run out of GPU memory
VALID_BATCH_SIZE = 8
LR = 1e-3


