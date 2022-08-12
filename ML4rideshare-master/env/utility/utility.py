from math import cos, asin, sqrt, pi

from env.env_config import SPEED


def haversine_distance(lat1, lon1, lat2, lon2):
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))  # distance in km


def manhattan_distance(loc_a, loc_b):
    lat1, lon1 = loc_a
    lat2, lon2 = loc_b
    return haversine_distance(lat1, 0, lat2, 0) + haversine_distance(0, lon1, 0, lon2)


def cal_runtime(loc_a, loc_b):
    distance = manhattan_distance(loc_a, loc_b)
    return distance / SPEED


def cal_total_distance(loc1, loc2, loc3, loc4, loc5):
    total_distance = manhattan_distance(loc1, loc2) + manhattan_distance(loc2, loc3) \
                     + manhattan_distance(loc3, loc4) + manhattan_distance(loc4, loc5)
    return total_distance


def avg(list):
    return sum(list) / len(list)


def normalize(max, min, value):
    return (value - min) / (max - min)


def find_shortest_path(o, s1, e1, s2, e2):
    """
    Given original loc o and 2 requests, s1->e1, s2->e2
    We want to find the shortest path
    """
    shortest_distance = 99999999
    start_loc = None
    end_loc = None

    # case1: o -> s1 -> s2 -> e1 -> e2
    distance = cal_total_distance(o, s1, s2, e1, e2)
    if distance < shortest_distance:
        shortest_distance = distance
        start_loc = s1
        end_loc = e2

    # case2: o -> s1 -> s2 -> e2 -> e1
    distance = cal_total_distance(o, s1, s2, e2, e1)
    if distance < shortest_distance:
        shortest_distance = distance
        start_loc = s1
        end_loc = e1

    # case3: o -> s2 -> s1 -> e1 -> e2
    distance = cal_total_distance(o, s2, s1, e1, e2)
    if distance < shortest_distance:
        shortest_distance = distance
        start_loc = s2
        end_loc = e2

    # case4: o -> s2 -> s1 -> e2 -> e1
    distance = cal_total_distance(o, s2, s1, e2, e1)
    if distance < shortest_distance:
        shortest_distance = distance
        start_loc = s2
        end_loc = e1

    pickup_time = cal_runtime(o, start_loc)  # the time to pick up the first customer
    total_pickup_time = 2 * pickup_time + cal_runtime(s1, s2)
    total_delivery_distance = 2 * (shortest_distance - manhattan_distance(o, start_loc)) \
                              - manhattan_distance(e1, e2) - manhattan_distance(s1, s2)
    total_delivery_time = total_delivery_distance / SPEED
    return shortest_distance, total_pickup_time, total_delivery_time, start_loc, end_loc
