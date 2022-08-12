import numpy as np
from env.models.vehicle import Vehicle


class Fleet(object):
    def __init__(self, n_vehicle, init_time, seed=0):
        self.n_vehicle = n_vehicle
        self.fleet = []
        self.time = init_time
        self.seed = seed

    def step(self, next_time, decisions):
        """
        input:
            next_time, datetime
                next time step
            decisions, dictionary
                'vehicle_id': ride object
                Receive the matching decisions from the vehicle-ride match
        """
        self.time = next_time
        for vehicle in self.fleet:
            if vehicle.id in decisions:
                vehicle.step(next_time, decisions[vehicle.id])
            else:
                vehicle.step(next_time)

    def reset(self, init_time):
        self.fleet = []
        id_list = self.initialize_id()
        loc_list = self.initialize_loc()
        for i in range(self.n_vehicle):
            vehicle_id = id_list[i]
            loc = loc_list[i]
            self.fleet.append(Vehicle(vehicle_id, init_time, loc))

    def get_idle_vehicles(self):
        """
        return all idle vehicles as a list for vehicle-ride match
        """
        idle_vehicles = []
        for vehicle in self.fleet:
            if vehicle.state == 'idle':
                idle_vehicles.append(vehicle)
        return idle_vehicles

    def initialize_loc(self):
        np.random.seed(self.seed)
        loc_list = []
        for i in range(self.n_vehicle):
            # loc = np.random.rand(2).tolist()
            loc = [40.75092741194779, -73.97361432244571]  # average lat and lon of orders
            loc_list.append(loc)
        return loc_list

    def initialize_id(self):
        np.random.seed(self.seed)
        id_list = []
        for i in range(self.n_vehicle):
            vehicle_id = np.random.randint(low=10000, high=99999)
            while vehicle_id in id_list:
                vehicle_id = np.random.randint(low=10000, high=99999)
            id_list.append(vehicle_id)
        return id_list

    def find_vehicle(self, vehicle_id):
        find = None
        for vehicle in self.fleet:
            if vehicle.id == vehicle_id:
                find = vehicle
        return find

    def get_ids(self):
        fleet_ids = []
        for vehicle in self.fleet:
            fleet_ids.append(vehicle.id)
        return fleet_ids
