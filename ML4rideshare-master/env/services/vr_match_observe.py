import networkx as nx

from env.env_config import SPEED, MAXIMUM_PICKUP_TIME
from env.utility.utility import manhattan_distance, find_shortest_path


def vehicle_ride_observe(vehicles, rides):
    """
    Given rides and vehicles' state, get the graph for vehicle-ride match
    Input:
     rides: list, [request1, request2, ...]
     vehicles: list, [vehicle1, vehicle2, ...]
        all idle vehicles
    Output:
     networkx graph
    """
    G = nx.Graph()
    num = len(vehicles) + len(rides)

    # add node
    G.add_nodes_from([i for i in range(num)])

    # set node attributes
    vehicles_attrs = {i: vehicles[i].get_attrs() for i in range(len(vehicles))}
    rides_attrs = {j + len(vehicles): rides[j].get_attrs() for j in range(len(rides))}
    attrs = {**vehicles_attrs, **rides_attrs}  # merge two dictionaries
    nx.set_node_attributes(G, attrs)

    # add edges
    for i in range(len(vehicles)):
        for j in range(len(rides)):
            if vr_whether_exist_edge(vehicles[i], rides[j]):
                G.add_edge(i, j + len(vehicles))

    # set edge attribute
    nx.set_edge_attributes(G, {e: {'weight':
                                       vr_edge_weight(vehicles[e[0]], rides[e[1] - len(vehicles)])} for e in G.edges})

    return G


def vr_whether_exist_edge(vehicle, ride):
    """
    Under vehicle-ride match graph,
    given one vehicle and one ride, check whether there is an edge between them
    """
    exist = True

    # criterion 1: pickup distance not too large
    if len(ride.requests) == 1:
        r = ride.requests[0]
        distance = manhattan_distance(r.start_loc, vehicle.loc)
    else:
        r1 = ride.requests[0]
        r2 = ride.requests[1]

        # the middle point of r1.start_loc and r2.start_loc
        mid_loc = [(a + b) / 2 for a, b in zip(r1.start_loc, r2.start_loc)]
        distance = manhattan_distance(mid_loc, vehicle.loc)

    if distance / SPEED > MAXIMUM_PICKUP_TIME:
        exist = False
    return exist


def vr_edge_weight(vehicle, ride):
    """
    Under vehicle-ride match graph,
    given one edge, calculate their weight
    """
    o = vehicle.loc
    if len(ride.requests) == 1:
        r = ride.requests[0]
        distance = manhattan_distance(o, r.start_loc) + manhattan_distance(r.start_loc, r.end_loc)
    else:
        r1 = ride.requests[0]
        r2 = ride.requests[1]
        distance, _, _, _, _ = find_shortest_path(o, r1.start_loc, r1.end_loc, r2.start_loc, r2.end_loc)

    return 1 / (distance + 0.001)  # distance is the total driving distance if vehicle serve this ride


def get_id_table(vehicles, rides):
    id_table = []
    for vehicle in vehicles:
        id_table.append(vehicle.vehicle_id)
    for ride in rides:
        id_table.append(ride.ride_id)
    return id_table
