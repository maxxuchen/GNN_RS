import gzip
import pickle
import torch
import numpy as np

from common.dataset_utility import get_target, networkx2torch

import torch.nn.functional as F


def Guided_Search_Eval(policy, instance, device):
    """
    :param policy: GCN network
    :param instance: one path to instance file
    :return: cross entropy
    """
    with gzip.open(instance, 'rb') as f:
        data = pickle.load(f)

    G = data['graph']
    MWM = data['MWM']
    mwm = get_target(G, MWM)  # get target matrix

    # guided search
    selection = Guided_Search(policy, G, device)
    selection = torch.from_numpy(selection)

    target = torch.from_numpy(mwm)
    gs_cross_entropy = F.binary_cross_entropy(selection.float(), target.float())

    score = gs_cross_entropy  # we want minimize the cross entropy
    return score


def Guided_Search(policy, graph, device):
    '''
    :param policy: GCN model
    :param graph: networkx graph
    :return: a match
    '''
    edges = [edge for edge in graph.edges]
    selection = np.zeros(len(edges))
    done = False

    temp_graph = graph.copy()
    while not done:
        # choose the edge with the highest score
        GCN_input = networkx2torch(temp_graph).to(device)
        temp_edges = GCN_input.edges  # list
        score = policy.eval(GCN_input).detach().numpy()  # tensor to numpy array
        edge_chosen = tuple(temp_edges[np.argmax(score)])
        selection[edges.index(edge_chosen)] = 1

        # delete edges and nodes
        for node_chosen in edge_chosen:
            # Removes the node_chosen and all adjacent edges
            temp_graph.remove_node(node_chosen)

        # check whether done
        if len(temp_graph.edges) == 0:
            done = True

    return selection