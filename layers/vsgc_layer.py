import torch as th
from torch import nn
import dgl.function as fn
from layers.pair_norm import PairNorm
import dgl


class VSGCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, k=1, graph_norm=True, alpha=1, dropout=0):
        super(VSGCLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k = k
        self.graph_norm = graph_norm
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, graph, features):
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
        dgl.remove_self_loop(g)
        # h = self.dropout(features)
        h = self.linear(features)

        h_pre = h
        ri = h * norm * norm
        for _ in range(self.k):
            if self.graph_norm:
                h = h * norm
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            if self.graph_norm:
                h = h * norm
            h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
            h_pre = h
        return h