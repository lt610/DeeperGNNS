from layers.vsgc_layer_pre import VSGCLayerPre
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv
import torch.nn.functional as F
from layers.vsgcwo_layer import VSGCWOLayer


class VSGCMLPNet(nn.Module):
    def __init__(self, num_feats, num_hidden, num_classes, num_layers, bias=True, alpha=1, lambd=1, dropout=0):
        super(VSGCMLPNet, self).__init__()
        self.vsgc = VSGCLayerPre(num_feats, num_hidden, bias, num_layers, alpha, lambd, dropout)
        self.mlp1 = MLPLayer(num_hidden, num_classes, activation=F.relu, dropout=dropout)
        self.mlp2 = MLPLayer(num_classes, num_classes, activation=None, dropout=dropout)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        h = self.mlp1(graph, h)
        h = self.mlp2(graph, h)
        return h