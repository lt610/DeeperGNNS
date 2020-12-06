import torch as th
from torch import nn
import dgl.function as fn
from layers.pair_norm import PairNorm
import dgl


class VSGCLayerMulti(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, k=1, dropout=0, propagation=0):
        super(VSGCLayerMulti, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.propagation = propagation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, graph, features):
        g = graph.local_var()
        degs = g.in_degrees().float()

        norm_1 = th.pow(degs, -1).to(features.device).unsqueeze(1)
        norm_05 = th.pow(degs, -0.5).to(features.device).unsqueeze(1)

        h = self.dropout(features)
        h = self.linear(h)
        if self.propagation == 0:
            h_initial = h * norm_1
            for _ in range(self.k):
                h = h * norm_05
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm_05
                h = h + h_initial

        elif self.propagation == 1:
            h_pre = h
            h_initial = h
            for _ in range(self.k):
                h = h * norm_05
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm_05
                h = h + h_initial - h_pre
                h_pre = h
        elif self.propagation == 2:
            h_pre = h
            h_initial = h * norm_1
            for _ in range(self.k):
                h = h * norm_05
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm_05
                h = h + h_initial - h_pre
                h_pre = h
        elif self.propagation == 3:
            h_initial = h
            for _ in range(self.k):
                h = h * norm_05
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')
                h = h * norm_05
                h = h + h_initial

        return h