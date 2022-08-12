import torch
import pathlib
import networkx as nx
from bc.bc_config import RR_net_config, VR_net_config
from es.utility.rr_greedy_search import RR_Greedy_Search_Match
from es.utility.vr_greedy_search import VR_Greedy_Search_Match
from models.RGGCN import ResidualGatedGCNModel
from models.Bipartite_RGGCN import Bipartite_RGGCNModel


class RR_Matcher(object):
    def __init__(self, type, model=None):
        self.type = type
        if self.type == 'heu':
            self.solver = nx.algorithms.matching.max_weight_matching
        elif self.type == 'nn' and model is not None:
            self.solver = model

    def get_rr_match_decision(self, graph):
        if len(graph.edges) > 0:
            if self.type == 'heu':
                rr_match = self.solver(graph)
            elif self.type == 'nn':
                rr_match = RR_Greedy_Search_Match(self.solver, graph)
            else:
                rr_match = None
        else:
            rr_match = None
        rr_match, decision = rr_match2decision(rr_match, graph)
        return rr_match, decision

    def update_model(self, new_model):
        # update gnn parameters
        assert self.type == 'nn'
        self.solver = new_model


class VR_Matcher(object):
    def __init__(self, type, model=None):
        self.type = type
        if self.type == 'heu':
            self.solver = nx.algorithms.matching.max_weight_matching
        elif self.type == 'nn' and model is not None:
            self.solver = model

    def get_vr_match_decision(self, graph):
        if self.type == 'heu':
            vr_match = self.solver(graph)
        elif self.type == 'nn':
            vr_match = VR_Greedy_Search_Match(self.solver, graph)
        else:
            vr_match = None
        vr_match, decision = vr_match2decision(vr_match, graph)
        return vr_match, decision

    def update_model(self, new_model):
        # update gnn parameters
        self.solver = new_model


def rr_match2decision(rr_match, graph):
    """
    Input:
        rr_match, python set
            obtained from heuristic alog or GCN
        graph, networkx graph
            rr match observe graph,
    Output:
        rr_decisions, list of lists
        [[r1_id, r2_id], [r3_id], [r4_id, r5_id], ...]
    """
    assert len(graph.nodes) > 0
    nodes_table = [i for i in range(len(graph.nodes))]
    decision = []

    if rr_match is not None:
        for pair in rr_match:
            node1, node2 = pair
            nodes_table.remove(node1)
            nodes_table.remove(node2)
            decision.append([graph.nodes[node1]['id'], graph.nodes[node2]['id']])

    # unmatched nodes
    for node in nodes_table:
        decision.append([graph.nodes[node]['id']])

    return rr_match, decision


def vr_match2decision(vr_match, graph):
    """
    Input:
        vr_match, python set
            obtained from heuristic alog or GCN
        graph, networkx graph
            vr match observe graph,
    Output:
        vr_decisions, dictionary
        'vehicle_id': ride_id
    """
    assert len(graph.nodes) > 0
    decision = {}
    for pair in vr_match:
        vehicle = min(pair)
        ride = max(pair)
        decision[graph.nodes[vehicle]['id']] = graph.nodes[ride]['id']
    # unmatched rides will be omitted
    return vr_match, decision
