from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F

from layers.mlp_layer import MLPLayer
from layers.vgcn_layer import VGCNLayer
from layers.vsgc_layer import VSGCLayer


class VMixNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_gcn=1, num_vsgc=1, bias=False, activation=F.relu, graph_norm=True,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0):
        super(VMixNet, self).__init__()
        self.gcns = nn.ModuleList()
        self.gcns.append(GCNLayer(num_feats, num_hidden, bias, activation, graph_norm, batch_norm, pair_norm,
                             residual, dropout, dropedge))
        for _ in range(num_gcn - 1):
            self.gcns.append(GCNLayer(num_hidden, num_hidden, bias, activation, graph_norm, batch_norm, pair_norm,
                                      residual, dropout, dropedge))
        self.vsgc = VSGCLayer(num_hidden, num_classes, bias, num_vsgc, dropout=dropout)

    def forward(self, graph, features):
        h = features
        for _, gcn in enumerate(self.gcns):
            h = gcn(graph, h)
        h = self.vsgc(graph, h)
        return h
