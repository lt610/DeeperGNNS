from layers.gcn_layer import cal_gain
from layers.mlp_layer import MLPLayer
from layers.vgcn_layer import VGCNLayer
import torch.nn as nn
import torch.nn.functional as F


class VGCNLayerNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers=2, bias=False, alpha=1,
                 activation=F.relu, residual=False, dropout=0):
        super(VGCNLayerNet, self).__init__()
        self.inLayer = MLPLayer(num_feats, num_hidden, bias=True, activation=activation, dropout=dropout)

        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            self.layers.append(VGCNLayer(num_hidden, num_hidden, bias, alpha, activation, residual, dropout))
        self.outLayer = MLPLayer(num_hidden, num_classes, bias=True, activation=None,  dropout=dropout)

    def forward(self, graph, features):
        initial_features = self.inLayer(graph, features)
        h = initial_features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, initial_features)
        h = self.outLayer(graph, h)
        return h