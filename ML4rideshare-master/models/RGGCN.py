import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RGGCN_layer import ResidualGatedGCNLayer, MLP


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.
    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/chaitjo/graph-convnet-tsp
    """

    def __init__(self, config):
        super(ResidualGatedGCNModel, self).__init__()

        # Define net parameters
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']

        # Node and edge embedding layers/lookups
        self.nodes_values_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(self.edge_dim-1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)

        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, data):
        """
        Args:
            data.x_edges: Input edge indicator value (num_nodes, num_nodes)
            data.x_edges_values: Input edge feature (num_nodes, num_nodes, edge_dim)
            data.x_nodes_values: Input node feature (num_nodes, node_dim)
            data.y_edges: Targets for edges (num_nodes, num_nodes)
        Returns:
            y_pred_edges: Predictions for edges (batch_size=1, num_nodes, num_nodes, voc_edges_out=2)
            loss: Value of loss function
        """
        # Node and edge embedding
        x = self.nodes_values_embedding(data.x_nodes_values.unsqueeze(0))  # B x V x H
        e_vals = self.edges_values_embedding(data.x_edges_values.unsqueeze(0))  # B x V x V x H
        e_tags = self.edges_embedding(data.x_edges.unsqueeze(0))  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)

        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out

        # Compute loss
        y = y_pred_edges.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss = nn.CrossEntropyLoss()(y, data.y_edges.unsqueeze(0))

        return y_pred_edges, loss

    def predict(self, data):
        """
        Args:
            data.x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            data.x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            data.x_nodes_values: Input node feature (batch_size, num_nodes, node_dim)
        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, voc_edges_out)
        """
        # Node and edge embedding
        x = self.nodes_values_embedding(data.x_nodes_values.unsqueeze(0))  # B x V x H
        e_vals = self.edges_values_embedding(data.x_edges_values.unsqueeze(0))  # B x V x V x H
        e_tags = self.edges_embedding(data.x_edges.unsqueeze(0))  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)

        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        y_pred_edges = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y_pred_edges = y_pred_edges[:, :, :, 1].squeeze(0)  # V x V

        return y_pred_edges
