import numpy as np


class Vehicle(object):
    def __init__(self, vehicle_id, init_time, loc):
        self.id = 'id' + str(vehicle_id)
        self.time = init_time
        self.loc = loc  # initial location [lat, lon]
        self.ride = None  # ongoing ride
        self.ride_end_time = None  # ride end time
        self.state = 'idle'  # vehicle state include 'idle' and 'occupied'
        # self.reward = []  # profit of this vehicle

    def get_feature(self):
        """
        get features for vehicle-ride matching
        """
        lat, lon = self.loc
        return np.array([lat, lon])

    def get_attrs(self):
        """
        get a dictionary that includes vehicle id and feature
        """
        attrs = {'id': self.id, 'type': 'vehicle', 'feature': self.get_feature()}
        return attrs

    def step(self, next_time, new_ride=None):
        """
        Only update time here!!
        idle vehicle: assign new ride or stay unchanged
        occupied vehicle: arrive destination or stay serving
        """
        self.time = next_time
        if self.state == 'occupied' and next_time >= self.ride_end_time:
            self.set_idle()
        elif self.state == 'idle' and new_ride is not None:
            self.set_occupied(new_ride)

    def set_idle(self):
        """
        vehicle finish the ride
        """
        assert self.ride is not None
        # self.reward.append(self.ride.get_reward())
        self.loc = self.ride.end_loc
        self.state = 'idle'
        self.ride = None
        self.ride_end_time = None

    def set_occupied(self, new_ride):
        """
        assign the new ride to the vehicle
        """
        assert self.ride is None
        self.state = 'occupied'
        self.ride = new_ride

        # find the shortest path
        self.ride.route_plan(self.loc)
        self.ride_end_time = self.time + self.ride.pickup_time + self.ride.delivery_time

