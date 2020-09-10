from layers.mlp_layer import MLPLayer
from layers.asgc_layer import ASGCLayer
import torch.nn as nn
import torch.nn.functional as F


class ASGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, graph_norm=True, activation=F.relu,
                 dropout=0):
        super(ASGCNet, self).__init__()
        self.layers = nn.ModuleList()
        self.inLayer = MLPLayer(num_feats, num_hidden, bias, activation, dropout=dropout)
        for _ in range(num_layers):
            self.layers.append(ASGCLayer(num_hidden, num_hidden, bias, graph_norm))
        self.outLayer = MLPLayer(num_hidden, num_classes, bias, None, dropout=dropout)

    def forward(self, graph, features):
        h = self.inLayer(graph, features)
        initial_features = h
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, initial_features)
        h = self.outLayer(graph, h)
        return h
