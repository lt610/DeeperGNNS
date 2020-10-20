from layers.vsgc_layer_pre import VSGCLayerPre
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv

from layers.vsgcwo_layer import VSGCWOLayer


class VSGCNetPre(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False, alpha=1, lambd=1, dropout=0):
        super(VSGCNetPre, self).__init__()
        self.vsgc = VSGCLayerPre(num_feats, num_classes, bias, num_layers, alpha, lambd, dropout)
        # self.vsgc = VSGCWOLayer(num_feats, num_classes, bias, num_layers, alpha, lambd, dropout)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        return h