import torch as th
from torch import nn
import dgl.function as fn

from layers.gcn_layer import cal_gain, Identity
from layers.pair_norm import PairNorm
import dgl
import torch.nn.functional as F


class AGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, graph_norm=True, activation=None, residual=False, dropout=0):
        super(AGCNLayer, self).__init__()
        self.a = nn.Linear(in_dim + out_dim, 1, bias=bias)
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.graph_norm = graph_norm
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.a.weight)
        if self.a.bias is not None:
            nn.init.zeros_(self.a.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.zeros_(self.res_fc.bias)

    def forward(self, graph, features, initial_features):
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        h_pre = features
        h = self.dropout(features)
        if self.graph_norm:
            h = h * norm
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        if self.graph_norm:
            h = h * norm

        hs = th.cat([h_pre, h], dim=1)
        alpha = F.sigmoid(self.a(hs))

        h = alpha * h + initial_features

        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = h + self.res_fc(h_pre)
        return h