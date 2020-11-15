from layers.vsgc_layer_after import VSGCLayer
from layers.mlp_layer import MLPLayer
import torch.nn as nn
from dgl.nn.pytorch import SGConv
import torch.nn.functional as F
from layers.vsgcwo_layer import VSGCWOLayer


class VSGCNet(nn.Module):
    def __init__(self, num_feats, num_hidden, num_classes, num_k, num_layers, bias=True, alpha=1, lambd=1,
                 activation=F.relu, batch_norm=False, residual=False, dropout=0):
        super(VSGCNet, self).__init__()
        self.vsgc = VSGCLayer(num_k, alpha, lambd, dropout)
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(MLPLayer(num_feats, num_classes, bias, None, batch_norm=False, residual=False, dropout=dropout))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(MLPLayer(num_feats, num_hidden, bias, activation, batch_norm, residual, dropout))
                elif i == num_layers - 1:
                    self.layers.append(MLPLayer(num_hidden, num_classes, bias, None, False, residual, dropout))
                else:
                    self.layers.append(MLPLayer(num_hidden, num_hidden, bias, activation, batch_norm, residual, dropout))

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
        return h