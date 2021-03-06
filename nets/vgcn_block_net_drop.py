from layers.vgcn_block_drop import VGCNBlock
from layers.mlp_layer import MLPLayer
import torch.nn as nn
import torch.nn.functional as F


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, num_blocks=2, bias=True, alpha=1, lambd=1,
                 feat_drop=0, attention=False, edge_drop=0, important=False):
        super(VGCNBlockNet, self).__init__()

        # self.mlp = MLPLayer(num_feats, num_classes, bias=True, dropout=dropout)
        # self.blocks = nn.ModuleList()
        # self.blocks.append(VGCNBlock(k, alpha, lambd))
        # for i in range(0, num_blocks - 2):
        #     self.blocks.append(VGCNBlock(k, alpha, lambd, attention))
        # self.blocks.append(VGCNBlock(k, alpha, lambd, attention))

        self.mlp1 = MLPLayer(num_feats, 64, bias=bias, dropout=feat_drop)
        self.block1 = VGCNBlock(k, alpha, lambd)
        self.mlp2 = MLPLayer(64, num_classes, bias=bias, dropout=feat_drop)
        self.block2 = VGCNBlock(k, alpha, lambd, attention, edge_drop=edge_drop, important=important)

    def forward(self, graph, features):
        # initial_features = self.mlp1(graph, features)
        # h = initial_features
        # for i, block in enumerate(self.blocks):
        #     h = block(graph, h, initial_features)

        initial1 = self.mlp1(graph, features)
        h = self.block1(graph, initial1, initial1)
        initial2 = self.mlp2(graph, h)
        h = self.block2(graph, h, initial2)

        return h