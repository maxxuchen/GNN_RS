import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from models.RGGCN_layer import BatchNormNode, BatchNormEdge, MLP


class Bipartite_EdgeFeatures(nn.Module):
    """Convnet features for edges.
    e_ij = U*e_ij + VL*x_i + VR*x_j
    """
    def __init__(self, hidden_dim):
        super(Bipartite_EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.VL = nn.Linear(hidden_dim, hidden_dim, True)
        self.VR = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, xl, xr, e):
        """
        Args:
            xl: Left node features (batch_size, left_num_nodes, hidden_dim)
            xr: Right node feature (batch_size, right_num_nodes, hidden_dim)
            e: Edge features (batch_size, left_num_nodes, right_num_nodes, hidden_dim)
        Returns:
            e_new: Convolved edge features (batch_size, left_num_nodes, right_num_nodes, hidden_dim)
        """
        Ue = self.U(e)  # B x VL x VR x H
        VLx = self.VL(xl).unsqueeze(2)  # Extend from "B x VL x H" to "B x VL x 1 x H"
        VRx = self.VR(xr).unsqueeze(1)  # Extend from "B x VR x H" to "B x 1 x VR x H"
        e_new = Ue + VLx + VRx
        return e_new


class Bipartite_NodeFeatures(nn.Module):
    """Convnet features for nodes.

    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]

    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """

    def __init__(self, hidden_dim, right_to_left=False, aggregation="mean"):
        super(Bipartite_NodeFeatures, self).__init__()
        self.right_to_left = right_to_left
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, xl, xr, edge_gate):
        """
        Args:
            xl: Left node features (batch_size, left_num_nodes, hidden_dim)
            xr: Right node feature (batch_size, right_num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, left_num_nodes, right_num_nodes, hidden_dim)
        Returns:
            xl_new: Convolved left node features (batch_size, num_nodes, hidden_dim)
            xr_new: Convolved right node features (batch_size, num_nodes, hidden_dim)
        """
        if self.right_to_left:
            Uxl = self.U(xl)  # B x VL x H
            Vxr = self.V(xr)  # B x VR x H
            Vxr = Vxr.unsqueeze(1)  # extend Vxr from "B x VR x H" to "B x 1 x VR x H"
            gateVxr = edge_gate * Vxr  # B x VL x VR x H
            if self.aggregation == "mean":
                xl_new = Uxl + torch.sum(gateVxr, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x VL x H
            elif self.aggregation == "sum":
                xl_new = Uxl + torch.sum(gateVxr, dim=2)  # B x VL x H
            return xl_new
        else:
            Uxr = self.U(xr)  # B x VR x H
            Vxl = self.V(xl)  # B x VL x H
            Vxl = Vxl.unsqueeze(2)  # extend Vxr from "B x VL x H" to "B x VL x 1 x H"
            gateVxl = edge_gate * Vxl  # B x VL x VR x H
            if self.aggregation == "mean":
                xr_new = Uxr + torch.sum(gateVxl, dim=1) / (1e-20 + torch.sum(edge_gate, dim=1))  # B x VR x H
            elif self.aggregation == "sum":
                xr_new = Uxr + torch.sum(gateVxl, dim=1)  # B x VL x H
            return xr_new


class Bipartite_Layer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(Bipartite_Layer, self).__init__()
        self.left2right_node_feat = Bipartite_NodeFeatures(hidden_dim, right_to_left=False, aggregation=aggregation)
        self.right2left_node_feat = Bipartite_NodeFeatures(hidden_dim, right_to_left=True, aggregation=aggregation)
        self.edge_feat = Bipartite_EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, xl, xr, e):
        """
        Args:
            xl: Left node features (batch_size, left_num_nodes, hidden_dim)
            xr: Right node feature (batch_size, right_num_nodes, hidden_dim)
            e: Edge feature (batch_size, left_num_nodes, right_num_nodes, hidden_dim)
        Returns:
            xl_new: Convolved left node features (batch_size, left_num_nodes, hidden_dim)
            xr_new: Convolved right node features (batch_size, right_num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, left_num_nodes, right_num_nodes, hidden_dim)
        """
        e_in = e
        xl_in, xr_in = xl, xr

        # Edge convolution
        e_tmp = self.edge_feat(xl_in, xr_in, e_in)  # B x VL x VR x H

        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)

        # Node convolution
        xl_tmp = self.right2left_node_feat(xl_in, xr_in, edge_gate)
        xr_tmp = self.left2right_node_feat(xl_in, xr_in, edge_gate)

        # Batch normalization
        # do we really need batch normalization here? it raises error when the number of node is 1
        # e_tmp = self.bn_edge(e_tmp)
        # xl_tmp = self.bn_node(xl_tmp)
        # xr_tmp = self.bn_node(xr_tmp)

        # ReLU Activation
        e = F.relu(e_tmp)
        xl = F.relu(xl_tmp)
        xr = F.relu(xr_tmp)

        # Residual connection
        xl_new = xl_in + xl
        xr_new = xr_in + xr
        e_new = e_in + e
        return xl_new, xr_new, e_new
