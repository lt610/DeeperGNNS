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
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.pair_norm = pair_norm
        if pair_norm:
            self.pn = PairNorm(mode='PN-SCS', scale=1)
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.dropedge = nn.Dropout(dropedge)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.zeros_(self.res_fc.bias)

    def forward(self, graph, features):
        h_pre = features
        g = graph.local_var()
        h = self.dropout(features)
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            h = h * norm
        g.ndata['h'] = h
        w = th.ones(g.number_of_edges(), 1).to(features.device)
        g.edata['w'] = self.dropedge(w)
        g.update_all(fn.u_mul_e('h', 'w', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        if self.graph_norm:
            h = h * norm
        h = self.linear(h)
        if self.batch_norm:
            h = self.bn(h)
        if self.pair_norm:
            h = self.pn(h)
        if self.activation is not None:
            h= self.activation(h)
        if self.residual:
            h = h + self.res_fc(h_pre)
        return h
