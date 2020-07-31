from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F


class GCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.relu, graph_norm=True,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(num_feats, num_hidden, bias, activation, graph_norm, batch_norm,
                                    pair_norm, residual, dropout, dropedge))
        for i in range(1, num_layers - 1):
            self.layers.append(GCNLayer(num_hidden, num_hidden, bias, activation, graph_norm, batch_norm,
                                        pair_norm, residual, dropout, dropedge))
        self.layers.append(GCNLayer(num_hidden, num_classes, bias, None, graph_norm, batch_norm,
                                    pair_norm, residual, 0, dropedge))

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        return h

