from layers.sgc_layer import SGCLayer
import torch.nn as nn
import torch.nn.functional as F


class SGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False, graph_norm=True, pair_norm=False):
        super(SGCNet, self).__init__()
        self.sgc = SGCLayer(num_feats, num_classes, bias, num_layers, graph_norm, pair_norm)

    def forward(self, graph, features):
        h = self.sgc(graph, features)
        return h
