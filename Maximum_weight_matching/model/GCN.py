import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from bc.bc_config import NODE_FEATURE_NUM


class GraphConvolution(torch.nn.Module):
    def __init__(self):
        super(GraphConvolution, self).__init__()
        self.project = torch.nn.Linear(NODE_FEATURE_NUM, 128)
        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 64)
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(64 * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        x = self.project(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        # edge regression
        batch_size = len(data.edges)
        total_edges = sum(len(data.edges[u]) for u in range(batch_size))
        output = torch.zeros(total_edges)

        for i in range(batch_size):
            _edges = data.edges[i]  # edge set of the i-th graph
            _nodes = data.nodes[i]  # node set of the i-th graph
            num_edge = len(_edges)
            num_node = len(_nodes)

            # the total number of nodes before i-th graph
            nodes_before = sum(len(data.nodes[u]) for u in range(i))
            edges_before = sum(len(data.edges[u]) for u in range(i))

            # get the node feature x of i-th graph
            _x = x[nodes_before: nodes_before + num_node]

            score = torch.zeros(num_edge)
            for j in range(num_edge):
                v1 = _edges[j][0]
                v2 = _edges[j][1]
                v1_index = _nodes.index(v1)  # get index of node
                v2_index = _nodes.index(v2)
                score[j] = self.seq(torch.cat((_x[v1_index, :], _x[v2_index, :])))
            score = F.softmax(score, dim=0)
            output[edges_before: edges_before + num_edge] = score

        return output

    def eval(self, data):
        torch.set_grad_enabled(False)
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.weight.float()

        x = self.project(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        # edge regression
        batch_size = len(data.edges)
        total_edges = sum(len(data.edges[u]) for u in range(batch_size))

        _edges = data.edges
        _nodes = data.nodes
        num_edge = len(_edges)
        num_node = len(_nodes)

        # get the node feature x of i-th graph
        _x = x
        score = torch.zeros(num_edge)
        for j in range(num_edge):
            v1 = _edges[j][0]
            v2 = _edges[j][1]
            v1_index = _nodes.index(v1)  # get index of node
            v2_index = _nodes.index(v2)
            score[j] = self.seq(torch.cat((_x[v1_index, :], _x[v2_index, :])))
        score = F.softmax(score, dim=0)

        return score
