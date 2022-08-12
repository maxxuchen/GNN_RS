import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Bipartite_RGGCN_layer import Bipartite_Layer, MLP



class Bipartite_RGGCNModel(nn.Module):
    """Bipartite Residual Gated GCN Model for outputting predictions as edge adjacency matrices.
    References:
        Code: https://github.com/chaitjo/graph-convnet-tsp
    """

    def __init__(self, config):
        super(Bipartite_RGGCNModel, self).__init__()

        # Define net parameters
        self.left_node_dim = config['left_node_dim']
        self.right_node_dim = config['right_node_dim']
        self.edge_dim = config['edge_dim']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']

        # Node and edge embedding layers/lookups
        self.left_nodes_values_embedding = nn.Linear(self.left_node_dim, self.hidden_dim, bias=False)
        self.right_nodes_values_embedding = nn.Linear(self.right_node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(self.edge_dim-1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)

        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(Bipartite_Layer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, data):
        """
        Args:
            data.x_edges: Input edge indicator value (left_num_nodes, right_num_nodes)
            data.x_edges_values: Input edge feature (left_num_nodes, right_num_nodes, edge_dim)
            data.x_left_nodes_values: Input left node feature (num_left_nodes, left_node_dim)
            data.x_right_nodes_values: Input right node feature (num_right_nodes, right_node_dim)
            data.y_edges: Targets for edges (left_num_nodes, right_num_nodes)
        Returns:
            y_pred_edges: Predictions for edges (batch_size=1, left_num_nodes, right_num_nodes, voc_edges_out=2)
            loss: Value of loss function
        Notation:
            B: batch_size
            VL: the number of left nodes
            VR: the number of right nodes
            H: the hidden dimension
        """
        # Node and edge embedding
        xl = self.left_nodes_values_embedding(data.x_left_nodes_values.unsqueeze(0))  # B x VL x H
        xr = self.right_nodes_values_embedding(data.x_right_nodes_values.unsqueeze(0))  # B x VR x H

        e_vals = self.edges_values_embedding(data.x_edges_values.unsqueeze(0))  # B x VL x VR x H/2
        e_tags = self.edges_embedding(data.x_edges.unsqueeze(0))  # B x VL x VR x H/2
        e = torch.cat((e_vals, e_tags), dim=3)  # B x VL x VR x H

        # GCN layers
        for layer in range(self.num_layers):
            xl, xr, e = self.gcn_layers[layer](xl, xr, e)

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x VL x VR x voc_edges_out

        # Compute loss
        y = y_pred_edges.permute(0, 3, 1, 2)  # B x voc_edges x VL x VR
        loss = nn.CrossEntropyLoss()(y, data.y_edges.unsqueeze(0))

        return y_pred_edges, loss
    
    def predict(self, data):
        """
        Args:
            data.x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            data.x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            data.x_left_nodes_values: Input node feature (batch_size, left_num_nodes, left_node_dim)
            data.x_right_nodes_values: Input node feature (batch_size, right_num_nodes, right_node_dim)
        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, voc_edges_out)
        """
        # Node and edge embedding
        xl = self.left_nodes_values_embedding(data.x_left_nodes_values.unsqueeze(0))  # B x VL x H
        xr = self.right_nodes_values_embedding(data.x_right_nodes_values.unsqueeze(0))  # B x VR x H

        e_vals = self.edges_values_embedding(data.x_edges_values.unsqueeze(0))  # B x VL x VR x H/2
        e_tags = self.edges_embedding(data.x_edges.unsqueeze(0))  # B x VL x VR x H/2
        e = torch.cat((e_vals, e_tags), dim=3)  # B x VL x VR x H

        # GCN layers
        for layer in range(self.num_layers):
            xl, xr, e = self.gcn_layers[layer](xl, xr, e) 

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x VL x VR x voc_edges_out
        y_pred_edges = F.log_softmax(y_pred_edges, dim=3)  # B x VL x VR x voc_edges
        y_pred_edges = y_pred_edges[:, :, :, 1].squeeze(0)  # VL x VR

        return y_pred_edges
