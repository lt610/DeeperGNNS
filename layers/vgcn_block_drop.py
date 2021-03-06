import torch as th
from torch import nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from layers.gcn_layer import cal_gain, Identity
from layers.pair_norm import PairNorm
import dgl
import torch.nn.functional as F


class VGCNBlock(nn.Module):
    def __init__(self, k=1, alpha=1, lambd=1, attention=False, epsilon=5e-3, edge_drop=0.5, important=False):
        super(VGCNBlock, self).__init__()

        self.k = k
        self.alpha = alpha
        self.attention = attention
        self.epsilon = epsilon
        self.edge_drop = edge_drop
        self.important = important

    def forward(self, graph, features, initial_features):
        g = graph.local_var()

        if self.attention:
            g.ndata['h'] = features
            g.apply_edges(fn.u_sub_v('h', 'h', 'dif'))
            dif = g.edata.pop('dif')
            l2 = th.norm(dif, p=2, dim=1)
            att = 1 / (2 * l2 + self.epsilon)

            if self.edge_drop > 0:
                k = int(att.shape[0] * self.edge_drop)
                _, drop_idxs = att.topk(k, largest=self.important, sorted=False)
                att[drop_idxs] = 0

            g.edata['att'] = att
            g.update_all(fn.copy_e('att', 'att'), fn.sum('att', 'degree'))
            degs = g.ndata.pop('degree')
            norm = th.pow(degs + 1, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        else:
            degs = g.in_degrees().float()
            norm = th.pow(degs + 1, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            g.edata['att'] = th.ones(g.number_of_edges(), 1).to(features.device)

        # # unweighted graph
        # if self.edge_drop > 0:
        #     g.ndata['h'] = features
        #     g.apply_edges(fn.u_sub_v('h', 'h', 'dif'))
        #     dif = g.edata.pop('dif')
        #     l2 = th.norm(dif, p=2, dim=1)
        #     att = 1 / (2 * l2 + self.epsilon)
        #
        #     if self.edge_drop > 0:
        #         k = int(att.shape[0] * self.edge_drop)
        #         _, drop_idxs = att.topk(k, largest=self.important, sorted=False)
        #         att[:] = 1
        #         att[drop_idxs] = 0
        #
        #     g.edata['att'] = att
        #     g.update_all(fn.copy_e('att', 'att'), fn.sum('att', 'degree'))
        #     degs = g.ndata.pop('degree')
        #     norm = th.pow(degs + 1, -0.5)
        #     norm = norm.to(features.device).unsqueeze(1)
        # else:
        #     degs = g.in_degrees().float()
        #     norm = th.pow(degs + 1, -0.5)
        #     norm = norm.to(features.device).unsqueeze(1)
        #     g.edata['att'] = th.ones(g.number_of_edges(), 1).to(features.device)

        h_pre = initial_features
        h = initial_features
        ri = initial_features * norm * norm
        for _ in range(self.k):
            h = h * norm

            g.ndata['h'] = h

            g.update_all(fn.u_mul_e('h', 'att', 'm'), fn.sum('m', 'h'))

            h = g.ndata.pop('h')

            h = h * norm

            h = self.alpha * h + self.alpha * ri + (1 - self.alpha) * h_pre
            h_pre = h

        return h