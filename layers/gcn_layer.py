import torch as th
from torch import nn
import dgl.function as fn
from torch.nn import functional as F
from layers.pair_norm import PairNorm
def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None, graph_norm=True, batch_norm=False,
                 pair_norm=False, residual=False, dropout=0, dropedge=0):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.activation = activation
        self.grahp_norm = graph_norm
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.pair_norm = pair_norm