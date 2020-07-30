import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F
from layers.pair_norm import PairNorm
from layers.gcn_layer import cal_gain, Identity


class DAGNNLayer(nn.Module):
    def __init__(self, num_classes, num_layers, graph_norm=True):
        super(DAGNNLayer, self).__init__()
        self.s = Parameter(th.FloatTensor(num_classes, 1))
        self.num_layers = num_layers
        self.graph_norm = graph_norm

    def reset_parameters(self):
        gain = cal_gain(F.relu)
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, features):
        g = graph.local_var()
        h = features
        results = [h]

        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        for _ in range(self.num_layers):
            if self.graph_norm:
                h = h * norm
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            if self.graph_norm:
                h = h * norm
            results.append(h)

        H = th.stack(results, dim=1)
        S = F.relu(th.matmul(H, self.s))
        S = S.permute(0, 2, 1)
        H = th.bmm(S, H).squeeze(1)
        return H
