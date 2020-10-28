import torch as th
from torch import nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from layers.gcn_layer import cal_gain, Identity
from layers.pair_norm import PairNorm
import dgl
import torch.nn.functional as F


class VGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, k=1, graph_norm=True, alpha=1, lambd=1, activation=None,
                 residual=False, dropout=0, attention=False):
        super(VGCNBlock, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k = k
        self.graph_norm = graph_norm
        self.alpha = alpha
        self.activation = activation
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
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
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs + 1, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        if self.attention:
            g.ndata['h'] = features
            g.apply_edges(fn.u_sub_v('h', 'h', 'l1'))
            l1 = g.edata.pop('l1')
            l1 = th.norm(l1, p=1, dim=1)
            g.edata['att'] = edge_softmax(g, l1)
        else:
            g.edata['att'] = th.ones(g.number_of_edges(), 1).to(features.device)

        h_last = features
        h = self.dropout(features)
        h = self.linear(h)
        h_pre = h
        ri = h * norm * norm

        for _ in range(self.k):

            if self.graph_norm:
                h = h * norm

            g.ndata['h'] = h

            g.update_all(fn.u_mul_e('h', 'att', 'm'), fn.sum('m', 'h'))

            h = g.ndata.pop('h')

            if self.graph_norm:
                h = h * norm

            h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
            h_pre = h

        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = h + self.res_fc(h_last)
        return h