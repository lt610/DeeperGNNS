from layers.vsgc_layer import VSGCLayer
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv

from layers.vsgcwo_layer import VSGCWOLayer


class VSGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False, alpha=1, lambd=1, dropout=0):
        super(VSGCNet, self).__init__()
        self.vsgc = VSGCLayer(num_feats, num_classes, bias, num_layers, alpha, lambd, dropout)
        # self.vsgc = VSGCWOLayer(num_feats, num_classes, bias, num_layers, alpha, lambd, dropout)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        return h