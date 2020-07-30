import math

from layers.gcn_layer import cal_gain
from layers.gcnii_layer import GCNIILayer
from layers.gcnii_variant_layer import GCNIIVariantLayer
import torch.nn as nn
import torch.nn.functional as F


class GCNIINet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False,
                 activation=F.relu, graph_norm=True, dropout=0, alpha=0, lamda=0, variant=False):
        super(GCNIINet, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            beta = math.log(lamda / (i + 1) + 1)
            if variant:
                self.convs.append(GCNIIVariantLayer(num_hidden, num_hidden, bias, activation,
                                                    graph_norm, alpha, beta))
            else:
                self.convs.append(GCNIILayer(num_hidden, num_hidden, bias, activation,
                                             graph_norm, alpha, beta))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_feats, num_hidden))
        self.fcs.append(nn.Linear(num_hidden, num_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def forward(self, graph, features):
        h0 = F.dropout(features, self.dropout, self.training)
        h0 = self.activation(self.fcs[0](h0))
        h = h0
        for con in self.convs:
            h = F.dropout(h, self.dropout, self.training)
            h = con(graph, h, h0)
        h = F.dropout(h, self.dropout, self.training)
        h = self.fcs[-1](h)
        return h
