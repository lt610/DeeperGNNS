from layers.vsgc_layer_pre import VSGCLayerPre
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv
import torch.nn.functional as F
from layers.vsgcwo_layer import VSGCWOLayer


class VSGCMLPNet(nn.Module):
    def __init__(self, num_feats, num_hidden, num_classes, num_layers, bias=True, alpha=1, lambd=1, dropout=0):
        super(VSGCMLPNet, self).__init__()
        self.mlp = MLPLayer(num_feats, num_hidden, bias=bias, activation=F.relu, dropout=dropout)
        self.vsgc = VSGCLayerPre(num_hidden, num_classes, bias, num_layers, alpha, lambd, dropout)

    def forward(self, graph, features):
        h = self.mlp(graph, features)
        h = self.vsgc(graph, h)
        return h