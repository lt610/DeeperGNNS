import torch as th
from torch import nn
import dgl.function as fn
from layers.pair_norm import PairNorm


class SGCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, k=1, graph_norm=True, pair_norm=False, cashed=False, dropout=0):
        super(SGCLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.k = k
        self.graph_norm = graph_norm
        self.pair_norm = pair_norm
        self.cashed = cashed
        self.cashed_h = None
        self.dropout = nn.Dropout(dropout)
        if pair_norm:
            self.pn = PairNorm(mode='PN', scale=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, graph, features):
        """创建一个局部副本变量，这样如果有修改操作不会造成交叉影响"""
        g = graph.local_var()
        if self.cashed_h is not None:
            h = self.cashed_h
        else:
            h = features
            if self.graph_norm:
                """保证度数至少为1，事实上添加了自循环后应该可以保证了"""
                degs = g.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                """features的维度是[n,d]，[n]维度的tensor广播后是[d,n]，所以需要unsqueeze将[n]添
                加一个维度变成[n,1]，这样经过广播后也是[n,d]"""
                norm = norm.to(features.device).unsqueeze(1)

            for _ in range(self.k):
                if self.graph_norm:
                    h = h * norm
                g.ndata['h'] = h
                """update_all()相当于register_message，register_reduce，send，recv一步到位，
                message和recv函数也可以自定义"""
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                """ndata内部是一个字典，键值是唯一的，update_all内部的计算应该是在局部副本变量上进行的，
                因此一开始ndata['h']指向features，内部更新后指向新的变量。这里也可以不pop，
                直接将features指向ndata['h']，即引用赋值"""
                h = g.ndata.pop('h')

                """对称归一化的邻接矩阵，聚合前后都有一次norm"""
                if self.graph_norm:
                    h = h * norm

                if self.pair_norm:
                    h = self.pn(h)

            if self.cashed:
                self.cashed_h = h
        h = self.dropout(h)
        h = self.linear(h)

        return h

