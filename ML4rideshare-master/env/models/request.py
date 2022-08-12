import numpy as np
from env.env_config import REQUEST_EXIST_TIME_STEP


class Request(object):
    def __init__(self, request_id, start_time, start_loc, end_loc, n_passenger):
        self.id = request_id
        self.start_time = start_time
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.n_passenger = n_passenger
        self.exist_time = 0

    def whether_exist(self):
        self.exist_time += 1
        if self.exist_time >= REQUEST_EXIST_TIME_STEP:
            return False
        else:
            return True

    def get_feature(self):
        # get the node feature of this request for request-request match
        # node feature: [start_lat, start_lon, end_lat, end_lon, exist_time]
        lat1, lon1 = self.start_loc
        lat2, lon2 = self.end_loc
        return np.array([lat1, lon1, lat2, lon2, self.exist_time])

    def get_attrs(self):
        """
        get a dictionary that includes request_id and features for request-request-matching
        """
        attrs = {'id': self.id, 'type': 'request', 'feature': self.get_feature()}
        return attrs

