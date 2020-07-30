import torch.nn as nn
from layers.mlp_layer import MLPLayer
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.relu,
                 batch_norm=False, residual=False, dropout=0):
        super(MLPNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            MLPLayer(num_feats, num_hidden, bias, activation, batch_norm, residual, dropout))
        for i in range(1, num_layers - 1):
            self.layers.append(
                MLPLayer(num_hidden, num_hidden, bias, activation, batch_norm, residual, dropout))
        self.layers.append(
            MLPLayer(num_hidden, num_classes, bias, None, batch_norm, residual, 0))

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        return h
