import torch
import numpy as np
from env.utility.utility import manhattan_distance
from torch_geometric.utils.convert import from_networkx
from bc.bc_config import RR_net_config


def rr_graph2torch(G, MWM=None):
    """
    change rr_graph graph (networkx) into GNN input format
    """
    data = from_networkx(G)
    num_node = len(G.nodes)
    node_dim = RR_net_config['node_dim']
    edge_dim = RR_net_config['edge_dim']

    x_nodes_values = np.zeros([num_node, node_dim])
    x_edges_values = np.zeros([num_node, num_node, edge_dim - 1])
    x_edges = np.zeros([num_node, num_node])

    for i in range(num_node):
        x_nodes_values[i, :] = G.nodes[i]['feature']
        for j in range(i + 1, num_node):
            x_edges_values[i, j, :] = rr_graph_edge_feature(G, i, j)[:-1]
            x_edges_values[j, i, :] = x_edges_values[i, j]
            x_edges[i, j] = rr_graph_edge_feature(G, i, j)[-1]
            x_edges[j, i] = x_edges[i, j]

    # Input node feature matrix (num_nodes, node_dim)
    data.x_nodes_values = torch.from_numpy(x_nodes_values).float()

    # Input edge feature matrix (num_nodes, num_nodes, edge_dim)
    data.x_edges_values = torch.from_numpy(x_edges_values).float()

    # Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
    data.x_edges = torch.from_numpy(x_edges).long()

    # add label
    if MWM is not None:
        data.y_edges = get_rr_target(G, MWM)  # get binary selection matrix

    return data


def rr_graph_edge_feature(G, i, j):
    """
    Given networkx graph G, node i and j, calculate the edge feature of i, j
    Edge feature:
        start_distance: distance between start locations of two requests,
        end_distance: distance between end locations of two requests,
        exist: whether the edge exists,

    """
    start_lat1, start_lon1, end_lat1, end_lon1, exist_time1 = G.nodes[i]['feature']
    start_lat2, start_lon2, end_lat2, end_lon2, exist_time2 = G.nodes[j]['feature']
    start_distance = manhattan_distance([start_lat1, start_lon1], [start_lat2, start_lon2])
    end_distance = manhattan_distance([end_lat1, end_lon1], [end_lat2, end_lon2])
    if G.has_edge(i, j):
        exist = 1
    else:
        exist = 0
    return np.array([start_distance, end_distance, exist])


def get_rr_target(G, MWM):
    """
    Given networkx graph and MWM solution, get the GNN training target
    """
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
    return torch.from_numpy(target).long()