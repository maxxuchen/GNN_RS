import numpy as np
import networkx as nx

from env.utility.utility import manhattan_distance
from env.env_config import MAXIMUM_SEATS


def request_request_observe(requests):
    """
    Given requests, create the graph for request-request match
    Input:
     requests: list, [request1, request2, ...]
    Output:
     networkx graph
    """
    num = len(requests)
    G = nx.Graph()

    # add node
    G.add_nodes_from([i for i in range(num)])

    # set node attributes
    attrs = {i: requests[i].get_attrs() for i in range(num)}
    nx.set_node_attributes(G, attrs)

    # add edges
    for i in range(num):
        for j in range(i + 1, num):
            if rr_whether_exist(requests[i], requests[j]):
                G.add_edge(i, j)

    # set edge attribute
    nx.set_edge_attributes(G, {e: {'weight': rr_edge_weight(requests[e[0]], requests[e[1]])} for e in G.edges})
    # nx.set_edge_attributes(G, {e: {'weight': rr_edge_weight(requests[e[0]], requests[e[1]]),
    #                                'feature': rr_edge_feature(requests[e[0]], requests[e[1]])} for e in G.edges})

    return G


def rr_whether_exist(r1, r2):
    """
    Under request-request match graph,
    given two requests, check whether there is an edge between them
    """
    exist = True

    # criterion 1: enough seats
    if r1.n_passenger + r2.n_passenger > MAXIMUM_SEATS:
        exist = False

    # criterion 2: enough overlap (saved_distance > 0)
    if rr_edge_weight(r1, r2) <= 0:
        exist = False

    return exist


def rr_edge_weight(r1, r2):
    """
    Under request-request match graph,
    given one edge (two requests), calculate their weight
    """
    distance1 = manhattan_distance(r1.start_loc, r1.end_loc)
    distance2 = manhattan_distance(r2.start_loc, r2.end_loc)

    distance = estimate_ride_distance(r1, r2)

    return distance1 + distance2 - distance  # saved distance if combine r1 and r2


def estimate_ride_distance(r1, r2):
    """
    When the location of vehicle is unknown, estimate the saved distance by combining two requests
    case1, case2: assume that the vehicle_loc is near r1.start_loc
    case3, case4: assume that the vehicle_loc is near r2.start_loc
    """
    case1 = manhattan_distance(r1.start_loc, r2.start_loc) \
            + manhattan_distance(r2.start_loc, r1.end_loc) + manhattan_distance(r1.end_loc, r2.end_loc)

    case2 = manhattan_distance(r1.start_loc, r2.start_loc) \
            + manhattan_distance(r2.start_loc, r2.end_loc) + manhattan_distance(r2.end_loc, r1.end_loc)

    case3 = manhattan_distance(r2.start_loc, r1.start_loc) \
            + manhattan_distance(r1.start_loc, r1.end_loc) + manhattan_distance(r1.end_loc, r2.end_loc)

    case4 = manhattan_distance(r2.start_loc, r1.start_loc) \
            + manhattan_distance(r1.start_loc, r2.end_loc) + manhattan_distance(r2.end_loc, r1.end_loc)

    return (min(case1, case2) + min(case3, case4)) / 2


def get_requests_table(requests):
    table = []
    for request in requests:
        table.append(request.request_id)
    return table


def rr_edge_feature(r1, r2):
    start_distance = manhattan_distance(r1.start_loc, r2.start_loc)
    end_distance = manhattan_distance(r1.end_loc, r2.end_loc)
    exist = rr_whether_exist(r1, r2)
    return np.array([start_distance, end_distance, exist])