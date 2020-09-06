from layers.vsgc_layer import VSGCLayer
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv


class VSGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False, graph_norm=True, alpha=1):
        super(VSGCNet, self).__init__()
        self.vsgc = VSGCLayer(num_feats, num_classes, bias, num_layers, graph_norm, alpha)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        return h