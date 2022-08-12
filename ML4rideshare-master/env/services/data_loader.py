import datetime
import numpy as np
import pandas as pd

from env.models.request import Request
from env.env_config import TIME_INTERVAL


class Data_Loader(object):
    def __init__(self, data_path, seed=0):
        self.data_path = data_path
        self.seed = seed
        with open(data_path) as f:
            self.data = pd.read_csv(f)

        self.preprocess()
        self.date_list = self.get_date_list(self.data)  # a list of all possible date
        self.time_slot = datetime.timedelta(minutes=TIME_INTERVAL)
        self.id_table = []

    def get_requests(self, time):
        """
        Given next time, return new requests
        Output:
            requests, list
            [request1, request2, ...]
        """
        requests = []
        tmp = self.data[time <= self.data['pickup_datetime']]
        chosen_data = tmp[tmp['pickup_datetime'] < time+self.time_slot]
        for index, row in chosen_data.iterrows():
            request_id = self.generate_new_id()
            start_time = row['pickup_datetime']
            start_loc = (row['pickup_latitude'], row['pickup_longitude'])
            end_loc = (row['dropoff_latitude'], row['dropoff_longitude'])
            n_passenger = row['passenger_count']
            requests.append(Request(request_id, start_time, start_loc, end_loc, n_passenger))

        return requests

    def reset(self):
        """
        randomly choose a day from the dataset
        """
        np.random.seed(self.seed)
        day = np.random.choice(self.date_list)  # choose a random day
        begin_time = pd.to_datetime(day, format='%Y-%m-%d')  # Timestamp('chosen day 00:00:00')
        self.id_table = []
        return begin_time

    def preprocess(self):
        # convert into datetime
        self.data['pickup_datetime'] = pd.to_datetime(self.data['tpep_pickup_datetime'])
        self.data['pickup_day'] = self.data['pickup_datetime'].dt.strftime('%Y-%m-%d')

        # # normalize the lat and lon
        # max_lat = max(max(self.data['pickup_latitude']), max(self.data['dropoff_latitude']))
        # min_lat = min(min(self.data['pickup_latitude']), min(self.data['dropoff_latitude']))
        # self.data['pickup_latitude'] = self.data['pickup_latitude'].apply(lambda x: (x - min_lat)/(max_lat - min_lat))
        # self.data['dropoff_latitude'] = self.data['dropoff_latitude'].apply(lambda x: (x - min_lat) / (max_lat - min_lat))
        #
        # max_lon = max(max(self.data['pickup_longitude']), max(self.data['dropoff_longitude']))
        # min_lon = min(min(self.data['pickup_longitude']), min(self.data['dropoff_longitude']))
        # self.data['pickup_longitude'] = self.data['pickup_longitude'].apply(lambda x: (x - min_lon)/(max_lon - min_lon))
        # self.data['dropoff_longitude'] = self.data['dropoff_longitude'].apply(lambda x: (x - min_lon) / (max_lon - min_lon))


    def generate_new_id(self):
        new_id = np.random.randint(low=1000000, high=9999999)
        while new_id in self.id_table:
            new_id = np.random.randint(low=1000000, high=9999999)
        self.id_table.append(new_id)
        return str(new_id)

    @staticmethod
    def get_date_list(data):
        """
        return a list of all possible date
        """
        tmp = data.drop_duplicates('pickup_day', keep='last')
        date_list = tmp['pickup_day'].tolist()
        return date_list