import torch
import numpy as np
from torch_geometric.utils.convert import from_networkx
from bc.bc_config import VR_net_config
from env.utility.utility import manhattan_distance, find_shortest_path


def vr_graph2torch(G, MWM=None):
    """
    change vr_graph graph (networkx) into GNN input format
    """
    # count left, right node num
    num_left_node = 0
    num_right_node = 0

    for i in range(len(G.nodes)):
        if G.nodes[i]['type'] == 'vehicle':
            num_left_node += 1
        elif G.nodes[i]['type'] == 'ride':
            num_right_node += 1

    data = from_networkx(G)
    left_node_dim = VR_net_config['left_node_dim']
    right_node_dim = VR_net_config['right_node_dim']
    edge_dim = VR_net_config['edge_dim']

    x_left_nodes_values = np.zeros([num_left_node, left_node_dim])
    x_right_nodes_values = np.zeros([num_right_node, right_node_dim])
    x_edges_values = np.zeros([num_left_node, num_right_node, edge_dim - 1])
    x_edges = np.zeros([num_left_node, num_right_node])

    # vehicle features
    for i in range(num_left_node):
        x_left_nodes_values[i, :] = G.nodes[i]['feature']

    # ride features
    for j in range(num_right_node):
        x_right_nodes_values[j, :] = G.nodes[num_left_node + j]['feature']

    for i in range(num_left_node):
        for j in range(num_left_node, num_left_node + num_right_node):
            x_edges_values[i, j - num_left_node, :] = vr_graph_edge_feature(G, i, j)[:-1]
            x_edges[i, j - num_left_node] = vr_graph_edge_feature(G, i, j)[-1]

    # Input node feature matrix (num_nodes, node_dim)
    data.x_left_nodes_values = torch.from_numpy(x_left_nodes_values).float()
    data.x_right_nodes_values = torch.from_numpy(x_right_nodes_values).float()

    # Input edge feature matrix (num_nodes, num_nodes, edge_dim)
    data.x_edges_values = torch.from_numpy(x_edges_values).float()

    # Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
    data.x_edges = torch.from_numpy(x_edges).long()

    # add label
    if MWM is not None:
        data.y_edges = get_vr_target(G, MWM)  # get binary selection matrix

    return data


def vr_graph_edge_feature(G, i, j):
    """
    Given networkx G, and nodes i, j
    if the edge between i, j exists, then i is vehicle node, j is the ride node
    Edge feature:
        total_distance: the total driving distance if the vehicle i serves the ride j
        exist: whether the edge i, j exists
    """
    assert G.nodes[i]['type'] == 'vehicle'
    assert G.nodes[j]['type'] == 'ride'
    v_loc = G.nodes[i]['feature'][:2]
    r1_start_loc = G.nodes[j]['feature'][:2]
    r1_end_loc = G.nodes[j]['feature'][2:4]
    r2_start_loc = G.nodes[j]['feature'][4:6]
    r2_end_loc = G.nodes[j]['feature'][6:8]
    total_distance, _, _, _, _ = find_shortest_path(v_loc, r1_start_loc, r1_end_loc, r2_start_loc, r2_end_loc)

    if G.has_edge(i, j):
        exist = 1
    else:
        exist = 0
    return np.array([total_distance, exist])


def get_vr_target(G, MWM):
    """
    Given networkx graph and MWM solution, get the GNN training target
    """
    num_left_node = 0
    num_right_node = 0

    for i in range(len(G.nodes)):
        if G.nodes[i]['type'] == 'vehicle':
            num_left_node += 1
        elif G.nodes[i]['type'] == 'ride':
            num_right_node += 1

    target = np.zeros([num_left_node, num_right_node])
    for i in range(num_left_node):
        for j in range(num_right_node):
            if (i, num_left_node+j) in MWM:
                target[i, j] = 1
            else:
                target[i, j] = 0
    return torch.from_numpy(target).long()