from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp_layer import MLPLayer
from layers.vgcn_layer import VGCNLayer
from layers.vsgc_layer import VSGCLayer


class VMixNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_inLayer=1, num_vsgc=1, bias=False, activation=F.relu, graph_norm=True,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0):
        super(VMixNet, self).__init__()
        self.inLayers = nn.ModuleList()
        self.inLayers.append(MLPLayer(num_feats, num_hidden, True, activation, dropout=dropout))
        if num_inLayer - 2 > 0:
            for _ in range(num_inLayer - 2):
                self.inLayers.append(MLPLayer(num_hidden, num_hidden, True, activation, dropout=dropout))
        self.inLayers.append(MLPLayer(num_hidden, num_hidden, True, None, dropout=dropout))
        self.vsgc = VSGCLayer(num_hidden, num_classes, bias, num_vsgc, dropout=dropout)

    def forward(self, graph, features):
        h = features
        for _, inLayer in enumerate(self.inLayers):
            h = inLayer(graph, h)
        h = self.vsgc(graph, h)
        return h
