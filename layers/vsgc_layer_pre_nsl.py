import torch as th
from torch import nn
import dgl.function as fn
from layers.pair_norm import PairNorm
import dgl


class VSGCLayerPre(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, k=1, alpha=1, lambd=1, dropout=0):
        super(VSGCLayerPre, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k = k
        self.alpha = alpha
        self.lambd = lambd
        self.dropout = nn.Dropout(dropout)

        self.exact_solution = False
        self.cache_es = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    # alpha and lambda
    def forward(self, graph, features):
        g = graph.local_var()

        degs = g.in_degrees().float()
        norm_lambd_1 = th.pow(self.lambd * degs + 1.0, -1)
        norm_lambd_1 = norm_lambd_1.to(features.device).unsqueeze(1)

        norm05 = th.pow(degs + 1.0, 0.5)
        norm05 = norm05.to(features.device).unsqueeze(1)
        norm_05 = th.pow(degs + 1.0, -0.5)
        norm_05 = norm_05.to(features.device).unsqueeze(1)

        h = self.dropout(features)
        h = self.linear(h)

        h_pre = h
        h_initial = h * norm_lambd_1
        for _ in range(self.k):
            h = h * norm_05

            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')

            h = h * norm_lambd_1 * norm05

            # h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
            h = self.alpha * self.lambd * h + (1 - self.alpha) * h_pre + self.alpha * h_initial

            h_pre = h

        return h