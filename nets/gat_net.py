import torch.nn as nn
import torch.nn.functional as F
from layers.gat_layer import GATLayer


class GATNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, num_heads, merge='cat',
                 activation=F.elu,batch_norm=False, residual=False, dropout=0):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(num_feats, num_hidden, num_heads, merge,
                                    activation, batch_norm, residual, dropout))
        for i in range(1, num_layers - 1):
            self.layers.append(GATLayer(num_hidden * num_heads, num_hidden, num_heads, merge,
                                        activation,batch_norm, residual, dropout))
        self.layers.append(
            GATLayer(num_hidden * num_heads, num_classes, 1, 'mean',
                     None, batch_norm, residual, dropout))

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        return h
