import torch.nn as nn
import torch.nn.functional as F
from layers.dagnn_layer import DAGNNLayer
from layers.gcn_layer import cal_gain


class DAGNNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=True, activation=F.relu, dropout=0):
        super(DAGNNNet, self).__init__()
        self.linear1 = nn.Linear(num_feats, num_hidden, bias)
        self.linear2 = nn.Linear(num_hidden, num_classes, bias)
        self.dagnn = DAGNNLayer(num_classes, num_layers)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        h = F.dropout(features, self.dropout, training=self.training)
        h = self.activation(self.linear1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear2(h)
        h = self.dagnn(graph, h)
        return h

