import datetime
import numpy as np

from env.utility.utility import manhattan_distance as cal_dis
from env.utility.utility import cal_runtime, find_shortest_path, avg
from env.env_config import DROP_FEE, SINGLE_DISTANCE_FEE, SHARE_DISTANCE_FEE, COST_FEE, SPEED


class Ride(object):
    def __init__(self, requests, ride_id):
        self.requests = requests  # a list of orders, can be 1 or 2 orders
        self.id = ride_id
        self.state = 'unassigned'

        # cannot decide the items below until match the ride to the vehicle
        self.start_loc = None  # the start loc of the ride
        self.end_loc = None  # the end loc of the ride
        self.total_distance = 0
        self.pickup_time = None
        self.delivery_time = None
        self.revenue = 0

    def get_feature(self):
        """
        get features for vehicle-ride matching
        """
        if len(self.requests) == 2:
            lat1, lon1 = self.requests[0].start_loc
            lat2, lon2 = self.requests[0].end_loc
            lat3, lon3 = self.requests[1].start_loc
            lat4, lon4 = self.requests[1].end_loc
            exist_time = max(self.requests[0].exist_time, self.requests[1].exist_time)
        else:
            lat1, lon1 = self.requests[0].start_loc
            lat2, lon2 = self.requests[0].end_loc
            lat3, lon3 = self.requests[0].start_loc
            lat4, lon4 = self.requests[0].end_loc
            exist_time = self.requests[0].exist_time

        return np.array([lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4, exist_time])

    def get_attrs(self):
        """
        get a dictionary that includes ride id and features for vehicle-ride matching
        """
        attrs = {'id': self.id, 'type': 'ride', 'feature': self.get_feature()}
        return attrs

    def estimate_time(self, vehicle_loc):
        """
        Given vehicle_loc, estimate the start_loc, end_loc, pickup_time, delivery_time
        It is used for the criterion of whether to create an edge between a vehicle and a ride
        """
        if len(self.requests) == 1:
            request = self.requests[0]
            start_loc = request.start_loc
            end_loc = request.end_loc
            pickup_time = cal_runtime(vehicle_loc, start_loc)
            delivery_time = cal_runtime(start_loc, end_loc)
            distance = cal_dis(vehicle_loc, start_loc) + cal_dis(start_loc, end_loc)
        else:
            r1 = self.requests[0]
            r2 = self.requests[1]
            distance, pickup_time, delivery_time, start_loc, end_loc = \
                find_shortest_path(vehicle_loc, r1.start_loc, r1.end_loc, r2.start_loc, r2.end_loc)

        return distance, start_loc, end_loc, pickup_time, delivery_time

    def route_plan(self, vehicle_loc):
        """
        find the shortest path to serve the ride based on current vehicle loc
        set up the start_loc, end_loc, pickup_time, delivery time, revenue
        """
        self.state = 'assigned'
        if len(self.requests) == 1:
            r = self.requests[0]
            self.start_loc = r.start_loc
            self.end_loc = r.end_loc
            pickup_time = cal_runtime(vehicle_loc, self.start_loc)
            delivery_time = cal_runtime(self.start_loc, self.end_loc)
            self.pickup_time = datetime.timedelta(minutes=pickup_time)
            self.delivery_time = datetime.timedelta(minutes=delivery_time)
            self.total_distance = cal_dis(vehicle_loc, self.start_loc) + cal_dis(self.start_loc, self.end_loc)
            self.revenue = DROP_FEE + SINGLE_DISTANCE_FEE * cal_dis(r.start_loc, r.end_loc) \
                           - COST_FEE * (cal_dis(vehicle_loc, r.start_loc) + cal_dis(r.start_loc, r.end_loc))

        elif len(self.requests) == 2:
            r1 = self.requests[0]
            r2 = self.requests[1]
            total_distance, pickup_time, delivery_time, self.start_loc, self.end_loc = \
                find_shortest_path(vehicle_loc, r1.start_loc, r1.end_loc, r2.start_loc, r2.end_loc)
            self.pickup_time = datetime.timedelta(minutes=pickup_time)
            self.delivery_time = datetime.timedelta(minutes=delivery_time)
            self.total_distance = total_distance
            self.revenue = 2 * DROP_FEE + SHARE_DISTANCE_FEE * (delivery_time * SPEED) \
                           - COST_FEE * total_distance
