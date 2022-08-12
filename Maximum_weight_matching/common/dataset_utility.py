import gzip
import math
import torch
import pickle
import numpy as np
import torch_geometric
import networkx as nx
from torch_geometric.utils.convert import from_networkx


class GraphDataset(torch_geometric.data.Dataset):
    """
    Dataset class implementing the basic methods to read samples from a file.

    Parameters
    ----------
    sample_files : list
        List containing the path to the sample files.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        Reads and returns sample at position <index> of the dataset.

        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        graph = sample['graph']
        MWM = sample['MWM']

        # change networkx graph into GNN input
        data = networkx2torch(graph, MWM)
        return data


# change networkx graph into GNN input format
def networkx2torch(G, MWM=None):
    data = from_networkx(G)
    data.x_nodes_coord = data.pos  # Input node coordinates (batch_size, num_nodes, node_dim)

    num_node = len(G.nodes)
    x_edges_values = np.zeros([num_node, num_node])
    x_edges = np.zeros([num_node, num_node])

    for i in range(num_node):
        for j in range(i, num_node):
            x1, y1 = G.nodes[i]['pos']
            x2, y2 = G.nodes[j]['pos']
            x_edges_values[i, j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            x_edges_values[j, i] = x_edges_values[i, j]
            if G.has_edge(i, j):
                x_edges[i, j] = 1
                x_edges[j, i] = 1
            else:
                x_edges[i, j] = 0
                x_edges[j, i] = 0

    # Input edge distance matrix (batch_size, num_nodes, num_nodes)
    data.x_edges_values = torch.from_numpy(x_edges_values).float()

    # Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
    data.x_edges = torch.from_numpy(x_edges).long()

    # add label
    if MWM is not None:
        data.y_edges = get_target(G, MWM)  # get binary selection matrix

    return data


# get imitation learning target
def get_target(G, MWM):
    num_node = len(G.nodes)
    target = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(i, num_node):
            if (i, j) in MWM:
                target[i, j] = 1
                target[j, i] = 1
            else:
                target[i, j] = 0
                target[j, i] = 0
    return torch.from_numpy(target).long()  # Targets for edges (batch_size, num_nodes, num_nodes)


# assign weight
def cal_weight(G, edge):
    node1 = edge[0]
    node2 = edge[1]
    x1, y1 = G.nodes[node1]['pos']
    x2, y2 = G.nodes[node2]['pos']
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)






