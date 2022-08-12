import gzip
import pickle
import torch
import numpy as np
import torch.nn.functional as F

from common.dataset_utility import networkx2torch


def Greedy_Search(policy, instance_path):
    """
    :param policy: GNN model
    :param instance_path: one path to instance file
    :return: selection, (num_edge), 1 for selected, 0 for not
    """
    with gzip.open(instance_path, 'rb') as f:
        instance = pickle.load(f)

    G = instance['graph']
    data = networkx2torch(G)

    # process the output of the GNN
    y_pred_edges = policy.predict(data)  # (batch_size, num_nodes, num_nodes, voc_edges_out)

    # greedy search
    temp_graph = G.copy()
    edges = [edge for edge in G.edges]
    selection = np.zeros(len(edges))
    done = False

    while not done:
        # choose the edge with the highest score
        select_edge = find_max_edge(y_pred_edges, temp_graph)
        selection[edges.index(select_edge)] = 1

        # Removes the node_chosen and all adjacent edges
        for node_chosen in select_edge:
            temp_graph.remove_node(node_chosen)

        # check whether done
        if len(temp_graph.edges) == 0:
            done = True

    # calculate the total weights
    weights = cal_total_weight(selection, G)

    return selection, weights


def find_max_edge(y_pred_edges, G):
    """
    :param y_pred_edges: the prob of being part of MWM (num_nodes, num_nodes)
    :param G: Networkx graph
    :return: edge with the highest prob.
    """
    select_edge = None
    select_prob = -999
    edges = G.edges
    for edge in edges:
        i, j = edge
        prob = (y_pred_edges[i, j] + y_pred_edges[j, i]) / 2
        if prob > select_prob:
            select_edge = tuple(edge)
            select_prob = prob

    assert select_edge is not None
    return select_edge


def cal_total_weight(selection, G):
    """
    :param selection: (num_edge), 1 for selected, 0 for not
    :param G: Networkx graph
    :return: total weight of selected edges
    """
    weight = 0
    edges = [edge for edge in G.edges]
    for i in range(len(edges)):
        if selection[i] == 1:
            n1, n2 = edges[i]
            weight += G[n1][n2]['weight']
    return weight

