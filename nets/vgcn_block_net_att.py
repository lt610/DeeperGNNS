from layers.vgcn_block_att import VGCNBlock
from layers.mlp_layer import MLPLayer
import torch.nn as nn
import torch.nn.functional as F


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, num_blocks=2, bias=True, alpha=1, lambd=1,
                 dropout=0, attention=False):
        super(VGCNBlockNet, self).__init__()

        self.mlp1 = MLPLayer(num_feats, 64, bias=True, dropout=dropout)
        self.mlp2 = MLPLayer(64, num_classes, bias=True, dropout=dropout)
        self.block1 = VGCNBlock(k, alpha, lambd)
        self.block2 = VGCNBlock(k, alpha, lambd, attention)
        # self.mlp = MLPLayer(num_feats, num_classes, bias=True, dropout=dropout)
        # self.blocks = nn.ModuleList()
        # self.blocks.append(VGCNBlock(k, alpha, lambd))
        # for i in range(0, num_blocks - 2):
        #     self.blocks.append(VGCNBlock(k, alpha, lambd, attention))
        # self.blocks.append(VGCNBlock(k, alpha, lambd, attention))

    def forward(self, graph, features):
        # initial_features = self.mlp1(graph, features)
        # h = initial_features
        # for i, block in enumerate(self.blocks):
        #     h = block(graph, h, initial_features)

        h = self.mlp1(graph, features)
        h = self.block1(graph, h, h)
        h = self.mlp2(graph, h)
        h = self.block2(graph, h, h)

        return h