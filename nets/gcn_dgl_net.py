import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCNDGLNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.relu):
        super(GCNDGLNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(num_feats, num_hidden, bias=bias, activation=activation))
        for i in range(1, num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, bias=bias, activation=activation))
        self.layers.append(GraphConv(num_hidden, num_classes, bias=bias, activation=None))

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        return h