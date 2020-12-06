from layers.vsgc_layer_multi import VSGCLayerMulti
import torch.nn as nn
from dgl.nn.pytorch import SGConv

from layers.vsgcwo_layer import VSGCWOLayer


class VSGCNetMulti(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, k=1, dropout=0, propagation=0):
        super(VSGCNetMulti, self).__init__()
        self.vsgc = VSGCLayerMulti(in_dim, out_dim, bias=bias, k=k, dropout=dropout, propagation=propagation)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        return h