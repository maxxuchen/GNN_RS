import datetime
import pandas as pd

from env.models.ride import Ride
from env.models.fleet import Fleet
from env.services.data_loader import Data_Loader
from env.services.data_container import Data_Container
from env.services.rr_match_observe import request_request_observe
from env.services.vr_match_observe import vehicle_ride_observe

from env.env_config import TIME_INTERVAL, SPEED


class Simulator(object):
    def __init__(self, n_vehicles, data_path, seed=0):
        self.seed = seed
        self.n_vehicles = n_vehicles
        self.data_path = data_path
        self.time_slot = datetime.timedelta(minutes=TIME_INTERVAL)
        self.time = None
        self.timestep = 0  # the number of pass time step
        self.end_time = None

        # initialize order_data_loader (transfer raw data)
        self.data_loader = Data_Loader(self.data_path, self.seed)

        # initialize the data container (temporal data container)
        self.request_container = Data_Container()
        self.ride_container = Data_Container()
        self.vehicle_container = Data_Container()

        # initialize the fleet
        self.fleet = Fleet(self.n_vehicles, self.seed)

    def reset(self):
        self.time = self.data_loader.reset()  # get the record from a new day
        self.time = self.time + datetime.timedelta(hours=8)  # start at 8:00
        self.timestep = 0  # the number of pass time step
        self.end_time = self.time + datetime.timedelta(days=0, minutes=0, hours=2)
        self.request_container.reset()  # clear all stored requests
        self.ride_container.reset()  # clear all stored rides
        self.vehicle_container.reset()
        self.fleet.reset(self.time)

        # get new information from env
        idle_vehicles = self.fleet.get_idle_vehicles()
        self.vehicle_container.renew(idle_vehicles)
        new_requests = self.data_loader.get_requests(self.time)
        self.request_container.renew(new_requests)

    def repeat(self, time):
        """
        repeat the data on a given day
        """
        self.time = time
        self.timestep = 0
        self.end_time = self.time + datetime.timedelta(days=1, hours=0)
        self.request_container.reset()  # clear all stored requests
        self.ride_container.reset()  # clear all stored rides
        self.vehicle_container.reset()
        self.fleet.reset(self.time)

        # get new information from env
        idle_vehicles = self.fleet.get_idle_vehicles()
        self.vehicle_container.renew(idle_vehicles)
        new_requests = self.data_loader.get_requests(self.time)
        self.request_container.renew(new_requests)

    def step(self, dispatch_action):
        """
        Only update time here.
        Do request-request match and vehicle-ride match before calling step().
        """
        done = False
        loss_request, assigned_request = 0, 0
        distance_driven, waiting_time, revenue = 0, 0, 0
        self.time = self.time + self.time_slot
        self.timestep += 1
        self.fleet.step(self.time, dispatch_action)  # dispatch vehicles

        if self.time >= self.end_time:
            done = True
        else:
            # delete assigned requests
            for ride in dispatch_action.values():
                for request in ride.requests:
                    self.request_container.delete(request)
                    assigned_request += 1

            # delete expired requests
            for request in self.request_container.get_list():
                if not request.whether_exist():
                    self.request_container.delete(request)
                    loss_request += 1

            # get new information from env
            idle_vehicles = self.fleet.get_idle_vehicles()
            self.vehicle_container.renew(idle_vehicles)
            new_requests = self.data_loader.get_requests(self.time)
            self.request_container.add(new_requests)

            # calculate the reward
            for ride in self.ride_container.get_list():
                if ride.state == 'assigned':
                    distance_driven += ride.total_distance
                    waiting_time += ride.pickup_time.total_seconds()/60
                    revenue += ride.revenue

        return done, assigned_request, loss_request, distance_driven, waiting_time, revenue

    def get_rr_match_graph(self):
        """
        Get the networkx graph for request-request matching
        If there is no request, return None
        """
        if self.request_container.check_none():
            return None  # no new orders
        else:
            rr_graph = request_request_observe(self.request_container.get_list())
            return rr_graph

    def do_rr_match(self, rr_decisions):
        """
        Combine requests into rides and store them in ride_container
        Input:
        rr_decisions, list of tuples
            [(r1_id, r2_id), (r3_id), (r4_id, r5_id), ...]
        Output: rides based on the rr match decisions
        """
        assert not self.request_container.check_none()
        rides = []
        for decision in rr_decisions:
            requests = []
            ride_id = 'ride'
            for request_id in decision:
                requests.append(self.request_container.find_id(request_id))
                ride_id = ride_id + request_id
            rides.append(Ride(requests, ride_id))
        self.ride_container.renew(rides)

    def get_vr_match_graph(self):
        """
        Return the networkx graph for the vehicle ride matching
        If there is no idle vehicle, return None
        """
        assert not self.ride_container.check_none()
        if self.vehicle_container.check_none():
            return None  # no idle vehicles
        else:
            vr_graph = vehicle_ride_observe(self.vehicle_container.get_list(), self.ride_container.get_list())
            return vr_graph

    def do_vr_match(self, vr_decisions):
        """
        Input: vr_decisions, dictionary
            'vehicle_id': ride_id
        Return: dispatch_action, dictionary
            'vehicle_id': ride objective
        """
        # return dispatch action (i.e., assign rides)
        dispatch_action = {}
        for vehicle_id in vr_decisions.keys():
            ride = self.ride_container.find_id(vr_decisions[vehicle_id])
            dispatch_action[vehicle_id] = ride
        return dispatch_action

    def get_time(self):
        return self.time


