import torch as th
from torch import nn
import dgl.function as fn
from torch.nn import functional as F
from layers.pair_norm import PairNorm
from layers.gcn_layer import cal_gain, Identity


class GCNIIVariantLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None, graph_norm=True, alpha=0, beta=0):
        super(GCNIIVariantLayer, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear2 = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.alpha = alpha
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=gain)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features, initial_features):
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            h = features * norm
        g.ndata['h'] = h
        w = th.ones(g.number_of_edges(), 1).to(features.device)
        g.edata['w'] = w
        g.update_all(fn.u_mul_e('h', 'w', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        if self.graph_norm:
            h = h * norm
        h = (1 - self.alpha) * h
        h = (1 - self.beta) * h + self.beta * self.linear1(h)
        if self.activation is not None:
            h = self.activation(h)
        ifeatures = self.alpha * initial_features
        h = h + (1 - self.beta) * ifeatures + self.beta * self.linear2(ifeatures)
        return h