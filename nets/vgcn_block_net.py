from layers.vgcn_block import VGCNBlock
import torch.nn as nn
import torch.nn.functional as F


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, k, num_blocks=2, bias=True, graph_norm=True, alpha=1, lambd=1,
                 activation=F.relu, residual=False, dropout=0):
        super(VGCNBlockNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(VGCNBlock(num_feats, num_hidden, bias, k, graph_norm, alpha, lambd, activation, residual, dropout))
        for i in range(0, num_blocks - 2):
            self.blocks.append(VGCNBlock(num_hidden, num_hidden, bias, k, graph_norm, alpha, lambd, activation, residual, dropout))
        self.blocks.append(VGCNBlock(num_hidden, num_classes, bias, k, graph_norm, alpha, lambd, None, residual, dropout))

    def forward(self, graph, features):
        h = features
        for i, block in enumerate(self.blocks):
            h = block(graph, h)
        return h