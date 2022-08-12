import torch
import torch.nn.functional as F


class FullyConnected(torch.nn.Module):
    def __init__(self, NODE_FEATURE_NUM):
        super(FullyConnected, self).__init__()
        self.NODE_FEATURE_NUM = NODE_FEATURE_NUM
        self.fc1 = torch.nn.Linear(2 * self.NODE_FEATURE_NUM, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.7)

    def forward(self, data):
        inp = self.node2edge(data)
        x = torch.relu(self.fc1(inp))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return torch.squeeze(x, -1)

    def node2edge(self, data):
        """
        :param data: torch_geometric.data.Data
            data.x: node feature
            data.edges: edge index set
            data.nodes: node index set
        :return:
            output: torch [total_edges, 2 * NODE_FEATURE_NUM]
                the edge features
        """
        # change node feature into edge feature
        device = data.x.device
        batch_size = len(data.edges)
        total_edges = sum(len(data.edges[u]) for u in range(batch_size))
        edge_feature = torch.zeros(total_edges, 2 * self.NODE_FEATURE_NUM).to(device)

        for i in range(batch_size):
            _edges = data.edges[i]  # edge set of the i-th graph
            _nodes = data.nodes[i]  # node set of the i-th graph
            num_edge = len(_edges)
            num_node = len(_nodes)

            # the total number of nodes before i-th graph
            nodes_before = sum(len(data.nodes[u]) for u in range(i))
            edges_before = sum(len(data.edges[u]) for u in range(i))

            # get the node feature x of i-th graph
            _x = data.x[nodes_before: nodes_before + num_node]

            for j in range(num_edge):
                v1 = _edges[j][0]
                v2 = _edges[j][1]
                v1_index = _nodes.index(v1)  # get index of node
                v2_index = _nodes.index(v2)
                edge_feature[edges_before + j, :] = torch.cat((_x[v1_index, :], _x[v2_index, :]))

        return edge_feature
