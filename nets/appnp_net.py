import torch.nn as nn
from dgl.nn.pytorch import APPNPConv
from nets.mlp_net import MLPNet


class APPNPNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, alpha, num_hidden, num_layers, bias=False, activation=None,
                 batch_norm=False, residual=False, dropout=0):
        super(APPNPNet, self).__init__()
        self.mlp = MLPNet(num_feats, num_classes, num_hidden, num_layers, bias,
                          activation, batch_norm, residual, dropout)
        self.appnp = APPNPConv(k, alpha)

    def forward(self, graph, features):
        h = features
        h = self.mlp(graph, h)
        h = self.appnp(h)
        return h
