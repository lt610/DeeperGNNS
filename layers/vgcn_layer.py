import torch as th
from torch import nn
import dgl.function as fn

from layers.gcn_layer import cal_gain, Identity
from layers.pair_norm import PairNorm
import dgl
import torch.nn.functional as F


class VGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, alpha=1, activation=None, residual=False, dropout=0):
        super(VGCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.alpha = alpha
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
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.zeros_(self.res_fc.bias)

    def forward(self, graph, features, initial_features):
        g = graph.local_var()

        degs = g.in_degrees().float().clamp(min=1)

        norm = th.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)

        norm_1 = th.pow(degs, -1)
        norm_1 = norm_1.to(features.device).unsqueeze(1)

        h_pre = features

        h = self.dropout(features)
        h = self.linear(h)

        # h_pre = h

        # print(h.shape)
        # print(norm.shape)
        ri = initial_features * norm_1
        h = h * norm
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * norm
        h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
        # h = self.linear(h)

        if self.activation is not None:
            h = self.activation(h)

        # h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
        if self.residual:
            h = h + self.res_fc(h_pre)
        return h