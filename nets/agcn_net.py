from layers.mlp_layer import MLPLayer
from layers.agcn_layer import AGCNLayer
import torch.nn as nn
import torch.nn.functional as F


class AGCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False,  graph_norm=True, activation=F.relu,
                 residual=False, dropout=0):
        super(AGCNNet, self).__init__()
        self.layers = nn.ModuleList()
        self.inLayer = MLPLayer(num_feats, num_hidden, bias, activation, residual=residual, dropout=dropout)
        for _ in range(num_layers):
            self.layers.append(AGCNLayer(num_hidden, num_hidden, bias, graph_norm, activation, residual, dropout))
        self.outLayer = MLPLayer(num_hidden, num_classes, bias, None, residual=residual, dropout=dropout)

    def forward(self, graph, features):
        h = self.inLayer(graph, features)
        initial_features = h
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, initial_features)
        h = self.outLayer(graph, h)
        return h

