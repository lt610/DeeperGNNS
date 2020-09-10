from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp_layer import MLPLayer


class GCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.relu, graph_norm=True,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0):
        super(GCNNet, self).__init__()
        self.layers = nn.ModuleList()
        self.inLayer = MLPLayer(num_feats, num_hidden, bias, activation, residual=residual, dropout=dropout)
        for _ in range(num_layers):
            self.layers.append(GCNLayer(num_hidden, num_hidden, bias, activation, graph_norm, batch_norm,pair_norm,
                                        residual, dropout, dropedge))
        self.outLayer = MLPLayer(num_hidden, num_classes, bias, None, residual=residual, dropout=dropout)

    def forward(self, graph, features):
        h = self.inLayer(graph, features)
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        h = self.outLayer(graph, h)
        return h

