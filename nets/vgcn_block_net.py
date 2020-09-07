from layers.vgcn_block import VGCNBlock
import torch.nn as nn
import torch.nn.functional as F


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, k, num_blocks=2, bias=False, graph_norm=True, alpha=1, activation=F.relu, residual=False):
        super(VGCNBlockNet, self).__init__()
        self.k = k
        self
        self.blocks = nn.ModuleList()
        self.blocks.append(VGCNBlock(num_feats, num_hidden, bias, k, graph_norm, alpha, activation, residual))
        for i in range(1, num_blocks - 1):
            self.blocks.append(VGCNBlock(num_hidden, num_hidden, bias, k, graph_norm, alpha, activation, residual))
        self.blocks.append(VGCNBlock(num_hidden, num_classes, bias, k, graph_norm, alpha, None, residual))

    def forward(self, graph, features):
        h = features
        for i, block in enumerate(self.blocks):
            h = block(graph, h)
        return h