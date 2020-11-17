from layers.vgcn_block_att import VGCNBlock
from layers.mlp_layer import MLPLayer
import torch.nn as nn
import torch.nn.functional as F


class VGCNBlockNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, num_blocks=2, bias=True, alpha=1, lambd=1,
                 feat_drop=0, attention=False, att_drop=0):
        super(VGCNBlockNet, self).__init__()
        # # 每个Block共享一个W
        # self.mlp = MLPLayer(num_feats, num_classes, bias=True, dropout=feat_drop)
        # self.block1 = VGCNBlock(k, alpha, lambd)
        # self.block2 = VGCNBlock(k, alpha, lambd, attention, att_drop=att_drop)

        # 每个Block的W都不同
        self.mlp1 = MLPLayer(num_feats, 64, bias=True, dropout=feat_drop)
        self.block1 = VGCNBlock(k, alpha, lambd)
        self.mlp2 = MLPLayer(64, num_classes, bias=True, dropout=feat_drop)
        self.block2 = VGCNBlock(k, alpha, lambd, attention, att_drop=att_drop)

    def forward(self, graph, features):
        # # 每个Block共享一个W
        # initial_features = self.mlp(graph, features)
        # h = self.block1(graph, initial_features, initial_features)
        # h = self.block2(graph, h, initial_features)

        # 每个Block的W都不同
        initial1 = self.mlp1(graph, features)
        h = self.block1(graph, initial1, initial1)
        initial2 = self.mlp2(graph, h)
        h = self.block2(graph, h, initial2)

        return h