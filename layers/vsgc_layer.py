import torch as th
from torch import nn
import dgl.function as fn
from layers.pair_norm import PairNorm
import dgl


class VSGCLayer(nn.Module):
    def __init__(self, k=1, alpha=1, lambd=1, dropout=0):
        super(VSGCLayer, self).__init__()
        self.k = k
        self.alpha = alpha
        self.lambd = lambd
        # 后续可能会加进参数列表里
        self.cashed = True
        self.cashed_h = None

    def forward(self, graph, features):
        g = graph.local_var()
        # g = g.remove_self_loop()
        if self.cashed_h is not None:
            h = self.cashed_h
        else:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

            norm_1 = th.pow(degs, -1)
            norm_1 = norm_1.to(features.device).unsqueeze(1)

            # g = g.remove_self_loop()
            h = features
            h_pre = h
            ri = h * norm_1
            for _ in range(self.k):

                h = h * norm
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')

                h = h * norm
                h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
                h_pre = h
            if self.cashed:
                self.cashed_h = h
        return h