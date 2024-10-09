""" key components of a gnn
1.Graph Representation: graphs are typically represented using an adjacency matrix and node feature
matrix.
2.Message passing: nodes aggregate information from their neighbours.
3.Nodes update: the updated node features are computed using the aggregated messages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__() 
        self.weights = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, adjacency_matrix:torch.Tensor, node_features):
        # normalize the adjacency matrix ? what is normalizing

        D = adjacency_matrix.sum(dim=1)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float("inf")] == 0 # avoid division by zero

        adjacency_matrix_normalized = D_inv_sqrt.view(-1, 1)*adjacency_matrix*D_inv_sqrt.view(-1, 1)

        # Message passing: Multiply the normalized adjacency matrix and the weights
        output = torch.matmul(adjacency_matrix_normalized, node_features@self.weights)
        return output