import math

from layers.gcn_layer import cal_gain
from layers.gcnii_layer import GCNIILayer
import torch.nn as nn
import torch.nn.functional as F


class GCNIINet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False,
                 activation=F.relu, graph_norm=True, dropout=0, alpha=0, lamda=0):
        super(GCNIINet, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            beta = math.log(lamda/(i+1)+1)
            self.convs.append(GCNIILayer(num_feats, num_hidden, num_hidden, bias, activation,
                                         graph_norm, alpha, beta))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_feats, num_hidden))
        self.fcs.append(nn.Linear(num_hidden, num_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def forward(self, graph, features):
        h0 = self.dropout(features)
        h0 = self.activation(self.fcs[0](h0))
        h = h0
        for con in self.convs:
            h = self.dropout(h)
            h = con(graph, h, h0)
        h = self.dropout(h)
        h = self.fcs[-1](h)
        return h
