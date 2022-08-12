import gzip
import torch
import pickle
import numpy as np
import torch_geometric
from bc.graph_dataset.rr_graph2torch import rr_graph2torch
from bc.graph_dataset.vr_graph2torch import vr_graph2torch


class GraphDataset(torch_geometric.data.Dataset):
    """
    Dataset class implementing the basic methods to read samples from a file.

    Parameters
    ----------
    sample_files : list
        List containing the path to the sample files.
    """

    def __init__(self, sample_files, graph_type):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.graph_type = graph_type  # 'rr_graph' or 'vr_graph'

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        Reads and returns sample at position <index> of the dataset.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        graph = sample['graph']
        MWM = sample['MWM']

        # change networkx graph into GNN input
        if self.graph_type == 'rr_graph':
            data = rr_graph2torch(graph, MWM)
        elif self.graph_type == 'vr_graph':
            data = vr_graph2torch(graph, MWM)
        else:
            data = None

        return data





