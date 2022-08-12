import gzip
import pickle
import torch
import numpy as np
import torch.nn.functional as F

from bc.graph_dataset.vr_graph2torch import vr_graph2torch


def VR_Greedy_Search_Match(policy, graph):
    """
    :param policy: GNN network
    :param graph: networkx graph
    :return: match, i.e., set() in python
     (same type as the solution of maximum weight matching)
    """
    match = set()
    edges = [edge for edge in graph.edges]
    selection, _ = VR_Greedy_Search(policy, graph)
    for i in range(len(edges)):
        if selection[i] == 1:
            edge_chosen = tuple(edges[i])
            match.add(edge_chosen)
    return match


def VR_Greedy_Search(policy, G):
    """
    :param policy: GNN model
    :param G: networkx graph
    :return: selection, (num_edge), 1 for selected, 0 for not
    :return: weights: the total weights of the matching
    """
    data = vr_graph2torch(G)
    y_pred_edges = policy.predict(data)  # (batch_size, num_nodes, num_nodes, voc_edges_out)

    # greedy search
    temp_graph = G.copy()
    edges = [edge for edge in G.edges]
    selection = np.zeros(len(edges))
    done = False

    while not done:
        # choose the edge with the highest score
        select_edge = find_max_edge(y_pred_edges, temp_graph, G)
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


def find_max_edge(y_pred_edges, G, original_G):
    """
    :param y_pred_edges: the prob of being part of MWM (num_nodes, num_nodes)
    :param G: Networkx graph
    :param original_G
    :return: edge with the highest prob.
    """
    assert len(G.edges) > 0 and len(G.nodes) > 0
    select_edge = None
    select_prob = -999
    edges = G.edges
    num_left_node, num_right_node = count_left_right_node(original_G)

    for edge in edges:
        i, j = edge
        prob = y_pred_edges[i, j-num_left_node]
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


def count_left_right_node(G):
    # count left, right node num
    num_left_node = 0
    num_right_node = 0

    for i in G.nodes:
        if G.nodes[i]['type'] == 'vehicle':
            num_left_node += 1
        elif G.nodes[i]['type'] == 'ride':
            num_right_node += 1
    return num_left_node, num_right_node
