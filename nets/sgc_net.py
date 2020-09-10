from layers.sgc_layer import SGCLayer
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv
import torch.nn.functional as F


class SGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False, graph_norm=True,
                 pair_norm=False, cashed=True, dropout=0):
        super(SGCNet, self).__init__()
        self.sgc = SGCLayer(num_feats, num_classes, bias, num_layers, graph_norm, pair_norm, cashed, dropout=dropout)
        # self.sgc = SGConv(num_feats, num_classes, num_layers, bias=bias)

    def forward(self, graph, features):
        h = self.sgc(graph, features)
        return h
