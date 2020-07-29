import torch as th
from torch import nn
import dgl.function as fn
from torch.nn import functional as F
from layers.pair_norm import PairNorm
from layers.gcn_layer import cal_gain, Identity


class GCNIILayer(nn.Module):
    def __init__(self, num_feats, in_dim, out_dim, bias=False, activation=None, graph_norm=True, alpha=0, beta=0):
        super(GCNIILayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.alpha = alpha
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

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
        h = (1 - self.alpha) * h + self.alpha * initial_features
        h = (1 - self.beta) * h + self.beta * self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h