from layers.gcn_layer import cal_gain
from layers.vgcn_layer import VGCNLayer
import torch.nn as nn
import torch.nn.functional as F


class VGCNLayerNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers=2, bias=False, graph_norm=True, alpha=1, activation=F.relu, residual=False):
        super(VGCNLayerNet, self).__init__()
        self.linear1 = nn.Linear(num_feats, num_hidden, bias)
        self.linear2 = nn.Linear(num_hidden, num_classes, bias)
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(VGCNLayer(num_hidden, num_hidden, bias, graph_norm, alpha, activation, residual))
        for i in range(1, num_layers - 1):
            self.layers.append(VGCNLayer(num_hidden, num_hidden, bias, graph_norm, alpha, activation, residual))
        self.layers.append(VGCNLayer(num_hidden, num_hidden, bias, graph_norm, alpha, activation, residual))

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        initial_features = self.activation(self.linear1(features))
        h = initial_features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, initial_features)
        h = self.linear2(h)
        return h